# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from flowpytertask import FlowpyterOperator
from airflow.sensors.time_delta import TimeDeltaSensorAsync
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.common.sql.hooks.sql import fetch_one_handler
from airflow import DAG

import yaml

# We're producing the residents/relocations and presence/trips aggregates/indicators
# in the same DAG, because redaction of the presence/trips indicators depends on
# the resident counts aggregates.
#
# Note: OPAL_indicators firstmonth DAG needs to run (once) before this DAG starts

root_dir = (
    "{{var.value.FLOWBOT_HOST_ROOT_DIR}}/hti-mobility-dashboard-indicators-pipeline"
)

with DAG(
    dag_id="OPAL_indicators",
    schedule=relativedelta(months=1),
    start_date=datetime(2022, 8, 1),
    end_date=None,
    default_args=dict(
        retries=3,
        retry_delay=timedelta(hours=1),
        weight_rule="upstream",  # Prefer to complete one DAG run before proceeding with the next
        host_notebook_out_dir=f"{root_dir}/executed_notebooks",
        host_dagrun_data_dir=f"{root_dir}/data/dagruns",
        host_shared_data_dir=f"{root_dir}/data/shared",
    ),
    params=dict(
        residents_reference_date="2020-01-01",
        relocations_reference_date="2020-02-01",
        presence_reference_date="2020-01-01",
        trips_reference_date="2020-01-01",
        data_version="2023-12-18",
    ),
) as dag:
    # DAG will start as soon as the data interval ends. But data will not all have arrived by then.
    # This task waits for a set time period before starting to run the aggregates notebooks.
    # We could use a sensor to check the latest ingestion date, but ultimately if we don't receive
    # the last day of data we'll eventually still want to run the aggregates.
    # Simplest solution is just to wait as long as we expect it to take and then start running.
    wait_for_ingest = TimeDeltaSensorAsync(
        task_id="wait_for_ingest",
        delta=timedelta(
            hours=30
        ),  # Wait 30 hours after data_interval_end (ingestion usually finishes after ~24h8m)
    )

    get_clusters_table = SQLExecuteQueryOperator(
        task_id="get_clusters_table",
        sql="SELECT mapping_table FROM geography.latest_clusters ORDER BY date_added DESC LIMIT 1",
        handler=lambda cursor: fetch_one_handler(cursor)[0],
        conn_id="flowdb",
    )

    #  Run the aggregates
    redacted_residents_aggregates = FlowpyterOperator(
        notebook_name="monthly_residence_aggregates_flowclient.ipynb",
        task_id="redacted_residents_aggregates",
        host_notebook_dir=f"{root_dir}/notebooks/aggregates",
        nb_params=dict(
            start_date="{{ ds }}",
            author="FlowBot",
            mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
            require_latest_data=True,  # Fail if most recent available CDR data are before the end of this month (implies the notebook is running too early)
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
            require_latest_data=True,  # Fail if most recent available CDR data are before the end of this month (implies the notebook is running too early)
        ),
        requires_flowapi=True,
        requires_flowdb=True,  # Ideally wouldn't need this, but direct flowdb access is required to check for truncated CDR data and to run subset query
        network_mode="flowbot_flowapi_flowmachine",
        pool="flowkit_queries",  # Use a pool to limit the number of concurrent compute-intensive aggregates-calculation tasks
    )

    # Run aggregates again using flowmachine directly, because we need them unredacted
    unredacted_residents_aggregates = FlowpyterOperator(
        notebook_name="monthly_residence_aggregates_flowmachine.ipynb",
        task_id="unredacted_residents_aggregates",
        host_notebook_dir=f"{root_dir}/notebooks/aggregates",
        nb_params=dict(
            start_date="{{ ds }}",
            author="FlowBot",
            mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
            redact=False,
            require_latest_data=True,  # Fail if most recent available CDR data are before the end of this month (implies the notebook is running too early)
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
            require_latest_data=True,  # Fail if most recent available CDR data are before the end of this month (implies the notebook is running too early)
        ),
        requires_flowapi=False,
        requires_flowdb=True,
        network_mode="flowbot_flowapi_flowmachine",
        pool="flowkit_queries",  # Use a pool to limit the number of concurrent compute-intensive aggregates-calculation tasks
    )

    # Calculate indicators from the aggregates
    residents_indicators = FlowpyterOperator(
        notebook_name="Residents_Relocations.ipynb",
        task_id="residents_indicators",
        host_notebook_dir=f"{root_dir}/notebooks/indicators",
        read_only_mounts=dict(static_dir=f"{root_dir}/static"),
        nb_params=dict(
            data_date="{{ ds }}",
            residents_reference_date="{{ params.residents_reference_date }}",
            relocations_reference_date="{{ params.relocations_reference_date }}",
            metric_crs_epsg=32618,
        ),
        requires_flowapi=False,
        requires_flowdb=False,
        depends_on_past=True,  # Indicators for this month depend on indicators for previous months
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
        depends_on_past=True,  # Indicators for this month depend on indicators for previous months
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
                        "arrived",
                        "departed",
                        "delta_arrived",
                        "residents_diffwithref",
                        "abnormality",
                        "residents_pctchangewithref",
                    ],
                    "indicators/relocations_indicators_{{ (dag_run.logical_date - macros.dateutil.relativedelta.relativedelta(months=1)).strftime('%Y-%m') }}to{{ dag_run.logical_date.strftime('%Y-%m') }}.csv": [
                        "relocations",
                        "relocations_diffwithref",
                        "abnormality",
                        "relocations_pctchangewithref",
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
                IS_FIRST_MONTH=False,
                JSON_OUTPUTS="redacted_jsons",
                CDR_POPULATION_FILE="cdr_subscriber_population.csv",
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
                "residents.arrived",
                "residents.departed",
                "residents.delta_arrived",
                "residents.residents_diffwithref",
                "residents.abnormality",
                "residents.residents_pctchangewithref",
                "relocations.relocations",
                "relocations.relocations_diffwithref",
                "relocations.abnormality",
                "relocations.relocations_pctchangewithref",
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

    # Doesn't really matter whether we run the redacted or unredacted aggregates tasks first
    (
        wait_for_ingest
        >> get_clusters_table
        >> redacted_residents_aggregates
        >> unredacted_residents_aggregates
        >> residents_indicators
        >> redact
        >> upload
    )
    (
        wait_for_ingest
        >> get_clusters_table
        >> redacted_presence_aggregates
        >> unredacted_presence_aggregates
        >> presence_indicators
        >> redact
        >> upload
    )
