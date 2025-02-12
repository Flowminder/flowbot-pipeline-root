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
    name: str = "approvalstest"
    printable_name: str = "approvalstest"
    slug: str = "npl-approvalstest-2016-01-01"
    country: str = "Nepal"
    country_slug: str = "npl"
    start_date: datetime = datetime.strptime(
        "2016-01-01", "%Y-%m-%d"
    )
    end_date: datetime = datetime.strptime("2016-12-31", "%Y-%m-%d")
    region_query: str = """
    
        SELECT admin3pcod, admin3name, admin2name, admin1name
        FROM geography.admin3
        LEFT JOIN geography.admin2 ON admin2pcod = substring(admin3pcod FOR 7) || '_1'
        LEFT JOIN geography.admin1 ON admin1pcod = substring(admin2pcod FOR 5) || '_1';
    
    """
    notebook_uid: str = "1001"
    notebook_gid: str = "1001"
    internal_network: str = "flowbot_flowapi_flowmachine"

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
