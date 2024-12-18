# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from flowpytertask import FlowpyterOperator
from airflow import DAG

import yaml

# DAG to produce and upload OPAL indicators _without_ running the aggregates first
# (for months where we have already produced the aggregates)
#
# We're producing the residents/relocations and presence/trips aggregates/indicators
# in the same DAG, because redaction of the presence/trips indicators depends on
# the resident counts aggregates.
#
# Note: OPAL_indicators_firstmonth DAG needs to run (once) before this DAG starts

root_dir = (
    "{{var.value.FLOWBOT_HOST_ROOT_DIR}}/hti-mobility-dashboard-indicators-pipeline"
)

with DAG(
    dag_id="OPAL_indicators_no_aggregates",
    schedule=relativedelta(months=1),
    start_date=datetime(2020, 2, 1),
    end_date=datetime(2022, 7, 1),
    default_args=dict(
        retries=3,
        retry_delay=timedelta(hours=1),
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
                CDR_POPULATION_FILE="cdr_subscriber_population.csv",
                IS_FIRST_MONTH=False,
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

    [residents_indicators, presence_indicators] >> redact >> upload
