# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import timedelta, datetime
from pathlib import Path

from flowpytertask import FlowpyterOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.common.sql.hooks.sql import fetch_one_handler
from airflow.decorators import task
from airflow.models import Variable
from airflow import DAG

from docker.types import Mount
import sys

sys.path.append(str(Path(__file__).parent.parent))
from active_crisis_config import ActiveCrisisConfig

config = ActiveCrisisConfig()

PROJECT_ROOT = Path(Variable.get("FLOWBOT_HOST_ROOT_DIR"), config.slug)


with DAG(
    dag_id=f"{config.name}_report",
    schedule="@daily",
    start_date=config.start_date,
    end_date=config.end_date,
    default_args=dict(
        host_notebook_out_dir=str(PROJECT_ROOT / "executed_notebooks"),
        host_notebook_dir=str(PROJECT_ROOT / "notebooks"),
        host_dagrun_data_dir=str(PROJECT_ROOT / "data" / "dagruns"),
        host_shared_data_dir=str(PROJECT_ROOT / "data" / "shared"),
        notebook_uid=config.notebook_uid,
        notebook_gid=config.notebook_gid,
    ),
    catchup=True,
) as dag:

    data_mount = Mount(
        type="bind",
        read_only=False,
        source=f"{PROJECT_ROOT}/data",
        target="/host_data",
    )

    @task.docker(
        image="python:latest",
        mounts=[data_mount],
        user=f"{config.notebook_uid}:{config.notebook_gid}",
        mount_tmp_dir=False,
    )
    def path_check_operator():
        from pathlib import Path

        Path("/host_data/shared/").mkdir(exist_ok=True, mode=0o774, parents=False)
        Path("/host_data/shared/aggregates/").mkdir(
            exist_ok=True, mode=0o774, parents=False
        )
        Path("/host_data/shared/aggregates/crisis_response/").mkdir(
            exist_ok=True, mode=0o774, parents=False
        )
        Path("/host_data/dagruns").mkdir(exist_ok=True, mode=0o774)

    check_paths_exist = path_check_operator()

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
        sql=config.region_query,
        conn_id="flowdb",
    )

    get_admin_areas_task = FlowpyterOperator(
        task_id="getting_admin_zones",
        image="flowminder/flowpyterlab:api-analyst-latest",
        host_notebook_dir=str(PROJECT_ROOT / "notebooks" / "reports"),
        notebook_name="get_admin_areas.ipynb",
        nb_params=dict(
            admin_file_path="admin3.geojson",
            admin_level="admin3",
            names_table="{{ ti.xcom_pull(task_ids='get_pcod_names') }}",
        ),
        requires_flowapi=True,
        network_mode="container:flowkit-flowapi-1",
    )

    [check_paths_exist, get_pcod_names] >> get_admin_areas_task

    common_agg_params = dict(
        author="FlowBot",
        start_date="{{ macros.ds_add(ds,-93) }}",  # max possible for three months
        logical_date="{{ ds }}",
        mapping_table="{{ ti.xcom_pull(task_ids='get_clusters_table') }}",
        geom_table="{{  ti.xcom_pull(task_ids='get_geom_table')  }}",
    )

    weekly_residence_aggregates_flowmachine = FlowpyterOperator(
        task_id=f"weekly_residence_aggregates_flowmachine",
        image="flowminder/flowpyterlab:api-analyst-latest",
        host_notebook_dir=str(PROJECT_ROOT / "notebooks" / "aggregates"),
        notebook_name="weekly_residence_aggregates_flowmachine.ipynb",
        requires_flowdb=True,
        nb_params=dict(
            **common_agg_params,
            include_subsetted=False,
            aggregates_to_run=[
                "home-relocations_consecutive",
                "home-relocations_disjoint",
            ],
            redact=True,
            require_lastest_data=True,
        ),
        network_mode="container:flowkit-flowmachine-1",
        pool="flowkit_queries",
    )

    [
        get_clusters_table,
        get_geom_table,
        check_paths_exist,
    ] >> weekly_residence_aggregates_flowmachine

    plotting_task = FlowpyterOperator(
        host_notebook_dir=str(PROJECT_ROOT / "notebooks" / "reports"),
        notebook_name="active_crisis.ipynb",
        task_id=f"plotting_{config.name}",
        depends_on_past=False,
        read_only_mounts=dict(static_dir=str(PROJECT_ROOT / "static")),
        nb_params=dict(
            event_date=config.start_date.strftime("%Y-%m-%d"),
            data_file_path="redacted_jsons",
            aggregate_dir="aggregates/crisis_response/",  # Redacted? Shared?
            spatial_geometry_file="admin3.geojson",
            spatial_geometry_unit_column="admin3pcod",
            affected_areas_file=str(config.affected_area),
            report_date="{{ ds }}",
        ),
        image="flowminder/flowpyterlab:api-analyst-latest",
        environment=dict(
            MAPBOX_WMTS_URL="{{ var.value.MAPBOX_WMTS_URL }}",  # Basemap URL
        ),
    )

    rendering_task = FlowpyterOperator(
        host_notebook_dir=str(PROJECT_ROOT / "notebooks" / "reports"),
        notebook_name="render_active_crisis_report.ipynb",
        task_id=f"rendering_{config.name}",
        depends_on_past=False,
        read_only_mounts=dict(
            static_dir=str(PROJECT_ROOT / "static"),
            key_obs_dir=str(PROJECT_ROOT / "key_obs"),
        ),
        nb_params=dict(
            report_json=str(Path("active_crisis") / "active_crisis.json"),
            html_out_folder=str(Path("html") / "{{ds}}"),
            logical_date="{{ds}}",
            incident_date=config.start_date_str,
            crisis_name=config.printable_name,
            update_freq="1",
            country=config.country,
        ),
        image="flowminder/flowpyterlab:api-analyst-latest",
    )

    pdf_task = FlowpyterOperator(
        host_notebook_dir=str(PROJECT_ROOT / "notebooks" / "reports"),
        notebook_name="html_to_pdf.ipynb",
        task_id=f"printing_{config.name}",
        depends_on_past=False,
        nb_params=dict(
            pdf_dir=str(Path("pdf") / "{{ds}}"),
            html_dir=str(Path("html") / "{{ds}}"),
            execution_date="{{ds}}",
            crisis_name=config.name,
        ),
        read_only_mounts=dict(
            static_dir=str(PROJECT_ROOT / "static"),
        ),
        image="flowminder/flowpyterlab:api-analyst-latest",
    )

    (
        [
            get_admin_areas_task,
            weekly_residence_aggregates_flowmachine,
        ]
        >> plotting_task
        >> rendering_task
        >> pdf_task
    )
