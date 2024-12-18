# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from dataclasses import dataclass
import os
from pathlib import Path
from pprint import pprint
from typing import List, Union
from shutil import copy

from flowbot_dataclasses.ProjectStructure import ProjectStructure


def filter_dir(root: Union[Path, str], to_include: List[Path]):
    root = Path(root)
    to_include = [path for p in to_include for path in root.glob(str(p))]
    print("files_to_include")
    pprint(to_include)
    for file in (p for p in root.rglob("*") if p.is_file()):
        if file not in to_include:
            file.unlink()


def clean_empty_folders(root: Path):
    for subfolder in (p for p in root.iterdir() if p.is_dir()):
        clean_empty_folders(subfolder)
        try:
            subfolder.rmdir()
        except (NotADirectoryError, OSError):
            pass


def to_abs(path: Union[Path, str]) -> Path:
    path = Path(path)
    if not path.is_absolute():
        return Path(os.getenv("PWD")) / path
    else:
        return path


def clean_configs(struct: ProjectStructure):
    for config_file in Path(".").glob("*_config.py"):
        if config_file != struct.config:
            config_file.unlink()


if __name__ == "__main__":
    from flowbot_dataclasses.CountryStaticData import {{cookiecutter.country|to_iso|lower}} as country
    from flowbot_dataclasses.ProjectStructure import {{cookiecutter.report_type}} as repo_type
    from flowbot_dataclasses.ProjectStructure import active_crisis


    print("Populating {{ cookiecutter.report_type }} for {{ cookiecutter.country }}")
    filter_dir("dags", repo_type.dags)
    filter_dir("notebooks", repo_type.notebooks)
    filter_dir("static", [*repo_type.static, country.pop_estimates_file, country.flow_weights_file])
    if repo_type == active_crisis:
        abs_path = to_abs("{{ cookiecutter.affected_area_path }}")
        print(f"Copying {abs_path} to static/affected_areas.json ")
        copy(abs_path, "static/affected_areas.json")
    print("Removing unused config files...")
    clean_configs(repo_type)
    print("Cleaning empty folders...")
    clean_empty_folders(Path("."))
    print("{{ cookiecutter.__project_slug }} ready!")
    print("Copying replay file into new repo...")
    try:
        # raise Exception("Rerunning a pipeline using the stored spec overwrites the spec incorrectly, so we're not doing it for now.")
        # Notes for the future fix: 
        # - When you run a Cookiecutter pipeline _without_ a replay file, it populates (in this case) flowbot_pipeline_root.json
        # - When you run run a Cookiecutter pipeline _with_ a replay file, it leaves this bit alone - but this bit of code 
        #   dropped flowbot_pipeline_root into the re-rendered repo.
        # - So when you reran the pipeline using --replay-file {other_pipeline}/original_spec.json, this would work fine the first time
        #   but overwrite original_spec.json incorrectly, which would then fail when you ran it the second time
        # - EITHER find a way to distinguish from in here if we're running using --replay-file
        # - OR leave it til we swap off Cookiecutter and onto Copier
        raise NotImplementedError("Updating original_spec.json currently disabled")
        if not Path("original_spec.json").exists():
            copy(
                Path(os.getenv("PWD"))
                / "cookiecutter_replay"
                / "flowbot-pipeline-root.json",
                "original_spec.json",
                
            )
    except Exception as e:
        print(e.with_traceback)
        print(
            "Issue with copying reply file - you may not be able to recreate this repo from clean"
        )
