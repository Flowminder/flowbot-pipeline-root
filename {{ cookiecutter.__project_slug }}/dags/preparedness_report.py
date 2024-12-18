# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

from flowpytertask import FlowpyterOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.common.sql.hooks.sql import fetch_one_handler
from airflow.decorators import task
from airflow.operators.python import BranchPythonOperator
from airflow import DAG

import sys

sys.path.append(str(Path(__file__).parent.parent))
from preparedness_config import PreparednessConfig

preparedness_config = PreparednessConfig()

PROJECT_ROOT = Path("{{var.value.FLOWBOT_HOST_ROOT_DIR}}", preparedness_config.slug)


with DAG(
    dag_id=f"{preparedness_config.name}_report_v2",
    schedule=relativedelta(months=1),
    start_date=preparedness_config.start_date
    - relativedelta(months=1),  # First dagrun is start_date + schedule
    end_date=preparedness_config.end_date,
    default_args=dict(
        host_notebook_out_dir=str(PROJECT_ROOT / "executed_notebooks"),
        host_notebook_dir=str(PROJECT_ROOT / "notebooks"),
        host_dagrun_data_dir=str(PROJECT_ROOT / "data" / "dagruns"),
        host_shared_data_dir=str(PROJECT_ROOT / "data" / "shared"),
        notebook_uid=preparedness_config.uid,
        notebook_gid=preparedness_config.gid,
    ),
) as dag:

    get_clusters_table = SQLExecuteQueryOperator(
        task_id="get_clusters_table",
        sql="SELECT mapping_table FROM geography.latest_clusters ORDER BY date_added DESC LIMIT 1",
        handler=lambda cursor: fetch_one_handler(cursor)[0],
        conn_id="flowdb",
    )

    get_geom_table = SQLExecuteQueryOperator(
        task_id="get_geom_table",
        sql="SELECT geom_table FROM geography.latest_clusters ORDER BY date_added DESC LIMIT 1",
        handler=lambda cursor: fetch_one_handler(cursor)[0],
        conn_id="flowdb",
    )

    get_pcod_names = SQLExecuteQueryOperator(
        task_id="get_pcod_names",
        sql=preparedness_config.region_query,
        conn_id="flowdb",
    )

    # TODO; This should also spit names in different languages into the geojson
    get_admin_areas_task = FlowpyterOperator(
        task_id="getting_admin_zones",
        image="flowminder/flowpyterlab:api-analyst-latest",
        host_notebook_dir=str(PROJECT_ROOT / "notebooks" / "reports"),
        notebook_name="get_admin_areas.ipynb",
        nb_params=dict(
            names_table="{{ ti.xcom_pull(task_ids='get_pcod_names') }}",
        ),
        requires_flowapi=True,
        network_mode=preparedness_config.internal_network,
    )

    get_pcod_names >> get_admin_areas_task

    redacted_presence_aggregates = FlowpyterOperator(
        notebook_name="daily_presence_aggregates_flowmachine.ipynb",
        task_id="unredacted_presence_aggregates",
        host_notebook_dir=f"{PROJECT_ROOT}/notebooks/aggregates",
        nb_params=dict(
            start_date="{{ ds }}",
            author="FlowBot",
            mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
            include_unsubsetted=True,
            aggregates_to_calculate=[
                "active-cell-counts"
            ],  # Only need OD matrices unredacted
            redact=True,
            require_latest_data=True,  # Fail if most recent available CDR data are before the end of this month (implies the notebook is running too early)
        ),
        requires_flowapi=False,
        requires_flowdb=True,
        network_mode=preparedness_config.internal_network,
        pool="flowkit_queries",  # Use a pool to limit the number of concurrent compute-intensive aggregates-calculation tasks
    )
    [get_clusters_table] >> redacted_presence_aggregates

    redacted_residents_aggregates = FlowpyterOperator(
        notebook_name="monthly_residence_aggregates_flowmachine.ipynb",
        task_id="unredacted_residents_aggregates",
        host_notebook_dir=f"{PROJECT_ROOT}/notebooks/aggregates",
        nb_params=dict(
            start_date="{{ ds }}",
            min_call_days=3,
            window_length=7,
            author="FlowBot",
            mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
            redact=True,  # TODO: Checkme
            require_latest_data=True,  # TODO: Checkme Fail if most recent available CDR data are before the end of this month (implies the notebook is running too early)
            geom_table_join_column="admin3pcod",
        ),
        requires_flowapi=False,
        requires_flowdb=True,
        network_mode=preparedness_config.internal_network,
        pool="flowkit_queries",  # Use a pool to limit the number of concurrent compute-intensive aggregates-calculation tasks
    )

    [get_clusters_table] >> redacted_residents_aggregates

    resident_indicators_common_args = dict(
        data_date="{{ds}}",
        residents_reference_date=(preparedness_config.residents_reference_date_str),
        base_pop_and_growth_rates_filename=str(preparedness_config.base_pop_file),
        scaling_factors_bilateral_pairs_filename=str(
            preparedness_config.flow_weights_file
        ),
    )

    # Template args get passed in as kwargs by BranchPythonOperator, so we just need 'ds' here.
    def is_first_month(*, ds):
        import logging

        logger = logging.getLogger(__name__)
        if datetime.strptime(ds, "%Y-%m-%d") <= preparedness_config.start_date:
            logger.info(
                f"{ds} is before {preparedness_config.start_date}, running init task"
            )
            return init_residents_indicators.task_id
        else:
            logger.info(
                f"{ds} is after {preparedness_config.start_date}, running regular task"
            )
            return residents_indicators.task_id

    do_init_check = BranchPythonOperator(
        task_id="do_init_check", python_callable=is_first_month
    )

    [
        get_geom_table,
        redacted_residents_aggregates,
        get_admin_areas_task,
        redacted_presence_aggregates,
    ] >> do_init_check

    # Init relocations; run only once
    init_residents_indicators = FlowpyterOperator(
        notebook_name="Residents_Relocations_init.ipynb",
        task_id="init_residents_indicators",
        host_notebook_dir=f"{PROJECT_ROOT}/notebooks/indicators",
        read_only_mounts=dict(static_dir=f"{PROJECT_ROOT}/static"),
        nb_params=dict(
            geometry_file="admin3.geojson",  # Check this; it should be dumped by get_admin_areas task
            metric_crs_epsg=32618,
            base_pop_column=f"est_pop_2020_01",
            geometry_admin3_col="admin3pcod",
            **resident_indicators_common_args,
        ),
        requires_flowapi=False,
        requires_flowdb=False,
    )
    do_init_check >> init_residents_indicators

    # Calculate indicators from the aggregates
    residents_indicators = FlowpyterOperator(
        notebook_name="Residents_Relocations.ipynb",
        task_id="residents_indicators",
        host_notebook_dir=f"{PROJECT_ROOT}/notebooks/indicators",
        read_only_mounts=dict(static_dir=f"{PROJECT_ROOT}/static"),
        nb_params=dict(
            metric_crs_epsg=32618,
            base_pop_column=f"est_pop_2020_01",
            geometry_admin3_col="admin3pcod",
            **resident_indicators_common_args,
        ),
        requires_flowapi=False,
        requires_flowdb=False,
        depends_on_past=True,  # Indicators for this month depend on indicators for previous months
    )
    do_init_check >> residents_indicators

    # Plot indicator
    preparedness_indicators = FlowpyterOperator(
        task_id="plotting_preparedness",
        notebook_name="preparedness.ipynb",
        host_notebook_dir=f"{PROJECT_ROOT}/notebooks/reports",
        nb_params=dict(
            country=preparedness_config.country_slug,
            date="{{ ds }}",
            monthly_residents_dir_name="indicators",
            admin_3_shapefile="admin3.geojson",
            projected_population=str(preparedness_config.base_pop_file),
            json_name="report_components.json",
        ),
        read_only_mounts=dict(
            static_dir=str(PROJECT_ROOT / "static"),
        ),
        environment=dict(
            MAPBOX_WMTS_URL="{{ var.value.MAPBOX_WMTS_URL }}",  # Basemap URL
        ),
    )
    residents_indicators >> preparedness_indicators

    # Render report
    rendering_task = FlowpyterOperator(
        host_notebook_dir=str(PROJECT_ROOT / "notebooks" / "reports"),
        notebook_name="render_preparedness_report.ipynb",
        task_id=f"rendering_{preparedness_config.name}",
        depends_on_past=False,
        read_only_mounts=dict(
            static_dir=str(PROJECT_ROOT / "static"),
            key_obs_dir=str(PROJECT_ROOT / "key_obs"),
        ),
        nb_params=dict(
            report_json="report_components/report_components.json",
            html_out_folder=str(Path("html") / "{{ds}}"),
            execution_date="{{ds}}",
            publication_date=datetime.today().strftime("%Y-%m-%d"),
            crisis_name="test_crisis",
            update_freq="1",
            country=preparedness_config.country,
            partners=preparedness_config.partners,
        ),
        image="flowminder/flowpyterlab:api-analyst-latest",
    )
    preparedness_indicators >> rendering_task

    pdf_task = FlowpyterOperator(
        host_notebook_dir=str(PROJECT_ROOT / "notebooks" / "reports"),
        notebook_name="html_to_pdf.ipynb",
        task_id=f"printing_{preparedness_config.name}",
        depends_on_past=False,
        nb_params=dict(
            pdf_dir=str(Path("pdf") / "{{ ds }}"),
            html_dir=str(Path("html") / "{{ ds }}"),
            execution_date="{{ ds }}",
            crisis_name=f"{preparedness_config.country_slug}_preparedness",
            country=preparedness_config.country,
        ),
        read_only_mounts=dict(
            static_dir=str(PROJECT_ROOT / "static"),
        ),
        image="flowminder/flowpyterlab:api-analyst-latest",
    )
    rendering_task >> pdf_task
