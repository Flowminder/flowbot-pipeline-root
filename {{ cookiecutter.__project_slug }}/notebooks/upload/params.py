# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import dataclasses
import functools
import gzip
import json
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import httpx
import logging

logging.basicConfig(level=logging.INFO)

log = logging.getLogger("upload_notebook")


@dataclasses.dataclass(frozen=True)
class LookupParams:
    endpoint: str
    prefix: str
    index_id: str
    value_id: str


@dataclasses.dataclass(frozen=True)
class SridParams(LookupParams):
    endpoint: str = "spatial_resolutions"
    prefix: str = ""
    index_id: str = "label"
    value_id: str = "index"


@dataclasses.dataclass(frozen=True)
class TridParams(LookupParams):
    endpoint: str = "temporal_resolutions"
    prefix: str = ""
    index_id: str = "relativedelta_unit"
    value_id: str = "index"


@dataclasses.dataclass(frozen=True)
class CategoryParams(LookupParams):
    endpoint: str = "categories"
    prefix: str = ""
    index_id: str = "category_id"
    value_id: str = "type"


def lookup_factory(
    lookup_params: type[LookupParams], param_fetcher: Callable
) -> Callable:
    """Creates a function that, given a"""
    lookup_response = param_fetcher(lookup_params.endpoint)
    lookup_dict = {
        f"{lookup_params.prefix}{parameter[lookup_params.index_id]}": parameter[
            lookup_params.value_id
        ]
        for parameter in lookup_response
    }

    def lookup_func(index):
        return str(lookup_dict[index])

    return lookup_func


def get_remote_parameter(parameter_name, admin_token, base_url):
    response = httpx.get(
        url=f"{base_url}/{parameter_name}",
        headers={"Authorization": f"Bearer {admin_token}"},
        follow_redirects=True,
    )
    log.info(response)
    return response.json()[parameter_name]


def get_local_parameter(parameter_name, config_path):
    with open(config_path) as fp:
        config = json.load(fp)
    return config[parameter_name]


def upload_config(config_file: Path, admin_token, base_url):
    with open(config_file) as fp:
        config = json.load(fp)
    upload_func = functools.partial(
        httpx.post,
        headers={
            "Content-Type": "application/json",
            # "Content-Encoding": "gzip",
            "Authorization": f"Bearer {admin_token}",
        },
        timeout=3600,
        follow_redirects=True,
    )
    log.info("Uploading config")
    upload_func(url=f"{base_url}/setup", json=config)
    log.info("Uploading spatial resolutions")
    for s_res in config["spatial_resolutions"]:
        log.info(str(s_res)[:30])
        response = upload_func(url=f"{base_url}/spatial_resolutions", json=s_res)
        if response.status_code >= 300:
            log.warning(response)
    log.info("Uploading temporal resolutions")
    for t_res in config["temporal_resolutions"]:
        log.info(str(t_res)[:30])
        response = upload_func(url=f"{base_url}/temporal_resolutions", json=t_res)
        if response.status_code >= 300:
            log.warning(response)
