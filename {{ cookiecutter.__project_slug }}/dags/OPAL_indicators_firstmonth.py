# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from datetime import datetime, timedelta

from flowpytertask import FlowpyterOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow import DAG
from airflow.providers.common.sql.hooks.sql import fetch_one_handler

import yaml

# We're producing the residents/relocations and presence/trips aggregates/indicators
# in the same DAG, because redaction of the presence/trips indicators depends on
# the resident counts aggregates.
#
# Note: Calculation and redaction of residents indicators in this first-month DAG (run once)
# is different from following months

root_dir = (
    "{{var.value.FLOWBOT_HOST_ROOT_DIR}}/hti-mobility-dashboard-indicators-pipeline"
)

with DAG(
    dag_id="OPAL_indicators_firstmonth",
    schedule="@once",
    start_date=datetime(2020, 1, 1),
    default_args=dict(
        retries=3,
        retry_delay=timedelta(hours=1),
        host_notebook_out_dir=f"{root_dir}/executed_notebooks",
        host_dagrun_data_dir=f"{root_dir}/data/dagruns",
        host_shared_data_dir=f"{root_dir}/data/shared",
    ),
    params=dict(
        presence_reference_date="2020-01-01",
        trips_reference_date="2020-01-01",
        data_version="TEST_20231215",
    ),
) as dag:
    get_clusters_table = SQLExecuteQueryOperator(
        task_id="get_clusters_table",
        sql="SELECT mapping_table FROM geography.latest_clusters ORDER BY date_added DESC LIMIT 1",
        handler=lambda cursor: fetch_one_handler(cursor)[0],
        conn_id="flowdb",
    )

    #  Run the aggregates (required for redaction purposes only, in the first month)
    redacted_residents_aggregates = FlowpyterOperator(
        notebook_name="monthly_residence_aggregates_flowclient.ipynb",
        task_id="redacted_residents_aggregates",
        host_notebook_dir=f"{root_dir}/notebooks/aggregates",
        nb_params=dict(
            start_date="{{ ds }}",
            author="FlowBot",
            mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
            require_latest_data=True,
            calculate_relocations=False,  # This is the first month, so we don't calculate relocations from the previous month
        ),
        requires_flowapi=True,
        requires_flowdb=True,  # Ideally wouldn't need this, but direct flowdb access is required to check for truncated CDR data and to run subset query
        network_mode="flowbot_flowapi_flowmachine",
        pool="flowkit_queries",  # Use a pool to limit the number of concurrent compute-intensive aggregates-calculation tasks
    )
    redacted_presence_aggregates = FlowpyterOperator(
        notebook_name="daily_presence_aggregates_flowclient.ipynb",
        task_id="redacted_presence_aggregates",
        host_notebook_dir=f"{root_dir}/notebooks/aggregates",
        nb_params=dict(
            start_date="{{ ds }}",
            author="FlowBot",
            mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
            include_unsubsetted=True,
            require_latest_data=True,
        ),
        requires_flowapi=True,
        requires_flowdb=True,  # Ideally wouldn't need this, but direct flowdb access is required to check for truncated CDR data and to run subset query
        network_mode="flowbot_flowapi_flowmachine",
        pool="flowkit_queries",  # Use a pool to limit the number of concurrent compute-intensive aggregates-calculation tasks
    )
    # Run aggregates again using flowmachine directly
    # (we don't actually need the unredacted residents aggregates in the first month, but let's run them anyway for completeness)
    unredacted_residents_aggregates = FlowpyterOperator(
        notebook_name="monthly_residence_aggregates_flowmachine.ipynb",
        task_id="unredacted_residents_aggregates",
        host_notebook_dir=f"{root_dir}/notebooks/aggregates",
        nb_params=dict(
            start_date="{{ ds }}",
            author="FlowBot",
            mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
            redact=False,
            require_latest_data=True,
            calculate_relocations=False,  # This is the first month, so we don't calculate relocations from the previous month
        ),
        requires_flowapi=False,
        requires_flowdb=True,
        network_mode="flowbot_flowapi_flowmachine",
        pool="flowkit_queries",  # Use a pool to limit the number of concurrent compute-intensive aggregates-calculation tasks
    )
    unredacted_presence_aggregates = FlowpyterOperator(
        notebook_name="daily_presence_aggregates_flowmachine.ipynb",
        task_id="unredacted_presence_aggregates",
        host_notebook_dir=f"{root_dir}/notebooks/aggregates",
        nb_params=dict(
            start_date="{{ ds }}",
            author="FlowBot",
            mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
            include_unsubsetted=True,
            aggregates_to_calculate=[
                "all-trips",
                "consecutive-trips",
                "home-away-matrix",
            ],  # Only need OD matrices unredacted
            redact=False,
            require_latest_data=True,
        ),
        requires_flowapi=False,
        requires_flowdb=True,
        network_mode="flowbot_flowapi_flowmachine",
        pool="flowkit_queries",  # Use a pool to limit the number of concurrent compute-intensive aggregates-calculation tasks
    )

    # Calculate indicators for first month
    residents_indicators_init = FlowpyterOperator(
        notebook_name="Residents_Relocations_init.ipynb",
        task_id="residents_indicators_init",
        host_notebook_dir=f"{root_dir}/notebooks/indicators",
        read_only_mounts=dict(static_dir=f"{root_dir}/static"),
        nb_params=dict(
            residents_reference_date="{{ ds }}",
            metric_crs_epsg=32618,
        ),
        requires_flowapi=False,
        requires_flowdb=False,
    )
    # Calculate indicators from the aggregates
    presence_indicators = FlowpyterOperator(
        notebook_name="Presence_Movements.ipynb",
        task_id="presence_indicators",
        host_notebook_dir=f"{root_dir}/notebooks/indicators",
        read_only_mounts=dict(static_dir=f"{root_dir}/static"),
        nb_params=dict(
            data_date="{{ ds }}",
            presence_reference_date="{{ params.presence_reference_date }}",
            trips_reference_date="{{ params.trips_reference_date }}",
            metric_crs_epsg=32618,
        ),
        requires_flowapi=False,
        requires_flowdb=False,
    )

    # Upload indicators to the mobility platform
    redact = FlowpyterOperator(
        notebook_name="csv_to_redacted_json.ipynb",
        task_id="redact",
        host_notebook_dir=f"{root_dir}/notebooks/upload",
        read_only_mounts=dict(static_dir=f"{root_dir}/static"),
        nb_params=yaml.safe_dump(
            dict(
                INDICATORS_FILES={
                    "indicators/residents_indicators_{{ dag_run.logical_date.strftime('%Y-%m') }}.csv": [
                        "residents",
                        "residents_perKm2",
                        "residents_diffwithref",
                        "residents_pctchangewithref",
                    ],
                    "indicators/presence_indicators_{{ dag_run.logical_date.strftime('%Y-%m') }}.csv": [
                        "presence",
                        "presence_perKm2",
                        "trips_in",
                        "trips_out",
                        "abnormality",
                        "presence_diffwithref",
                        "presence_pctchangewithref",
                    ],
                    "indicators/movements_indicators_{{ dag_run.logical_date.strftime('%Y-%m') }}.csv": [
                        "travellers",
                        "abnormality",
                        "travellers_diffwithref",
                        "travellers_pctchangewithref",
                    ],
                },
                DATA_VERSION="{{ params.data_version }}",
                CDR_POPULATION_FILE="cdr_subscriber_population.csv",
                ADMIN3_WITH_ACTIVITY_FILE="admin3s_with_cdr_activity.csv",
                IS_FIRST_MONTH=True,
                JSON_OUTPUTS="redacted_jsons",
            )
        ),  # Specifying params as yaml because dict keys cannot be templated
        requires_flowapi=False,
        requires_flowdb=False,
    )

    upload = FlowpyterOperator(
        notebook_name="upload_indicators.ipynb",
        task_id="upload",
        host_notebook_dir=f"{root_dir}/notebooks/upload",
        read_only_mounts=dict(static_dir=f"{root_dir}/static"),
        nb_params=dict(
            JSON_DATA_SUBDIR="redacted_jsons",
            BASE_URL="https://api.dev.haiti.mobility-dashboard.org/v1",
            INDICATORS_TO_UPLOAD=[
                "residents.residents",
                "residents.residents_perKm2",
                "residents.residents_diffwithref",
                "residents.residents_pctchangewithref",
                "presence.presence",
                "presence.presence_perKm2",
                "presence.trips_in",
                "presence.trips_out",
                "presence.abnormality",
                "presence.presence_diffwithref",
                "presence.presence_pctchangewithref",
                "movements.travellers",
                "movements.abnormality",
                "movements.travellers_diffwithref",
                "movements.travellers_pctchangewithref",
            ],
        ),
        # Auth0 settings are set via envs because they include secrets, and params would be visible in the executed notebook
        environment=dict(
            ADMIN_CLIENT="{{ var.value.AUTH0_ADMIN_CLIENT }}",  # Admin client id from Auth0
            UPDATER_CLIENT="{{ var.value.AUTH0_UPDATER_CLIENT }}",  # Updater client id from Auth0
            ADMIN_SECRET="{{ var.value.AUTH0_ADMIN_SECRET }}",  # Admin secret from Auth0
            UPDATER_SECRET="{{ var.value.AUTH0_UPDATER_SECRET }}",  # Updater secret from Auth0
            AUTH0_DOMAIN="{{ var.value.AUTH0_DOMAIN }}",  # Auth0 domain to request tokens from
            AUDIENCE="https://flowkit-ui-backend.flowminder.org",  # Domain to request tokens for
        ),
        requires_flowapi=False,
        requires_flowdb=False,
    )
    (
        redacted_residents_aggregates
        >> unredacted_residents_aggregates
        >> residents_indicators_init
        >> redact
        >> upload
    )
    (
        redacted_presence_aggregates
        >> unredacted_presence_aggregates
        >> presence_indicators
        >> redact
        >> upload
    )
    # Indicators init notebook uses presence aggregates to determine which admin3s have CDR activity
    redacted_presence_aggregates >> residents_indicators_init
