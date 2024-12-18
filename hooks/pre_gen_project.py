# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Union
import cookiecutter
import cookiecutter.utils


def to_abs(path: Union[Path, str]) -> Path:
    path = Path(path)
    if not path.is_absolute():
        return Path(os.getenv("PWD")) / path
    else:
        return path


try:
    datetime.strptime("{{ cookiecutter.start_date }}", "%Y-%m-%d")
except ValueError:
    print("start_date must be of the form yyyy-mm-dd")
    sys.exit(1)
try:
    datetime.strptime("{{ cookiecutter.end_date }}", "%Y-%m-%d")
except ValueError:
    print("end_date must be of the form yyyy-mm-dd")
    sys.exit(1)
abs_path = to_abs("{{cookiecutter.affected_area_path}}")
if "{{ cookiecutter.report_type }}" == "active_crisis" and not abs_path.exists():
    print(f"{abs_path} not found")
    sys.exit(1)
