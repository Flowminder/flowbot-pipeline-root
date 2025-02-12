from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class PreparednessConfig:
    name: str = "{{cookiecutter.crisis_name | slugify}}"
    slug: str = "{{ cookiecutter.__project_slug }}"
    country: str = "{{ cookiecutter.__full_country_name }}"
    country_slug: str = "{{ cookiecutter.__country_iso }}"
    start_date: datetime = datetime.strptime(
        "{{ cookiecutter.start_date }}", "%Y-%m-%d"
    )
    end_date: datetime = datetime.strptime("{{ cookiecutter.end_date }}", "%Y-%m-%d")
    base_pop_file: Path = Path("{{ cookiecutter.__country_iso | pop_estimates_file }}")
    flow_weights_file: Path = Path(
        "{{ cookiecutter.__country_iso | flow_weights_file }}"
    )
    region_query: str = """
       {{ cookiecutter.__country_iso | region_query }} 
    """
    residents_reference_date: datetime = datetime.strptime(
        "{{ cookiecutter.residents_reference_date }}", "%Y-%m-%d"
    )
    relocations_reference_date: datetime = datetime.strptime(
        "{{ cookiecutter.relocations_reference_date }}", "%Y-%m-%d"
    )
    notebook_uid: str = "{{cookiecutter.__country_iso | notebook_uid}}"
    notebook_gid: str = "{{cookiecutter.__country_iso | notebook_gid}}"
    internal_network: str = "{{cookiecutter.__country_iso | internal_network}}"
    partners: List[str] = field(
        default_factory=lambda: [
            "swiss_sponsor_crop.jpeg",
            "hewlett.png",
            "afd.png",
            "digicel_red.jpeg",
        ]
    )

    def __post_init__(self):
        self.start_date_str = self.start_date.strftime("%Y-%m-%d")
        self.end_date_str = self.end_date.strftime("%Y-%m-%d")
        self.residents_reference_date_str = self.residents_reference_date.strftime(
            "%Y-%m-%d"
        )
        self.relocations_reference_date_str = self.relocations_reference_date.strftime(
            "%Y-%m-%d"
        )
