# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Populated by Cookiecutter

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class ActiveCrisisConfig:
    # Cookiecutter set vars
    name: str = "{{ cookiecutter.crisis_name | slugify }}"
    printable_name: str = "{{ cookiecutter.crisis_name }}"
    slug: str = "{{ cookiecutter.__project_slug }}"
    country: str = "{{ cookiecutter.__full_country_name }}"
    country_slug: str = "{{ cookiecutter.__country_iso }}"
    start_date: datetime = datetime.strptime(
        "{{ cookiecutter.start_date }}", "%Y-%m-%d"
    )
    end_date: datetime = datetime.strptime("{{ cookiecutter.end_date }}", "%Y-%m-%d")
    region_query: str = """
    {{ cookiecutter.__country_iso | region_query }}
    """
    notebook_uid: str = "{{cookiecutter.__country_iso | notebook_uid}}"
    notebook_gid: str = "{{cookiecutter.__country_iso | notebook_gid}}"
    internal_network: str = "{{cookiecutter.__country_iso | internal_network}}"

    # Constant and derived vars
    affected_area: Path = Path("affected_areas.json")
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
