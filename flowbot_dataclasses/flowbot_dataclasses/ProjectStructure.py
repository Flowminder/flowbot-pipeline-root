from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ProjectStructure:
    notebooks: List[Path]
    static: List[Path]
    dags: List[Path]
    config: Path


active_crisis = ProjectStructure(
    notebooks=[
        Path("aggregates/*"),
        Path("indicators/*"),
        Path("reports/*"),
        Path("reports/**/*.py"),
    ],
    static=[
        Path("templates/active_crisis_report/*"),
        Path("templates/active_crisis_report/translations/**/*.json"),
        Path("templates/images/*"),
        Path("templates/page_preamble.html"),
    ],
    dags=[Path("active_crisis_report.py"), Path("aggregates/")],
    config=Path("active_crisis_config.py"),
)

preparedness = ProjectStructure(
    notebooks=[
        Path("aggregates/*"),
        Path("indicators/*"),
        Path("reports/preparedness.ipynb"),
        Path("reports/render_preparedness_report.ipynb"),
        Path("reports/html_to_pdf.ipynb"),
        Path("reports/get_admin_areas.ipynb"),
        Path("reports/**/*.py"),
        Path("reports/*.py"),
    ],
    static=[
        Path("templates/preparedness_report/*"),
        Path("templates/images/*"),
        Path("templates/page_preamble.html"),
        Path("data/*"),
    ],
    dags=[Path("preparedness_report.py")],
    config=Path("preparedness_config.py"),
)

dashboard_indicators = ProjectStructure(
    notebooks=[Path("aggregates/*"), Path("indicators/*"), Path("uploads/*")],
    static=[],
    dags=[Path("OPAL_indicators*.py")],
    config=Path("indicator_config.py"),
)
