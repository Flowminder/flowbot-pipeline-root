# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import asyncio
import dataclasses
import gzip
import itertools
import json
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import (
    Callable,
    Generator,
    Iterable,
    AsyncIterable,
    Awaitable,
    Coroutine,
    Any,
)

import httpx

import logging

from data_cleaning import JsonDataset

log = logging.getLogger("upload_notebook")


class AttemptState(Enum):
    UNTRIED = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclasses.dataclass
class UploadAttempt:
    dataset: JsonDataset
    state: AttemptState = AttemptState.UNTRIED
    response: httpx.Response = None


def batched(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def ingestor_factory(
    client, base_url, admin_token, local_request_cache_folder=None
) -> Callable[[UploadAttempt], Coroutine[Any, Any, Any]]:
    async def upload_dataset(attempt: UploadAttempt) -> UploadAttempt:
        log.info(
            f"Uploading {attempt.dataset.dt} {attempt.dataset.indicator_id}, hash {attempt.dataset.__hash__()}"
        )
        if local_request_cache_folder:
            (local_request_cache_folder / attempt.dataset.filename).write_text(
                json.dumps(attempt.dataset.request)
            )
        response = await client.patch(
            url=f"{base_url}/data",
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Authorization": f"Bearer {admin_token}",
            },
            data=gzip.compress(
                json.dumps(attempt.dataset.request, default=str).encode("utf-8")
            ),
            timeout=3600,
        )
        if response.status_code in [201, 204]:
            log.info(
                f"{attempt.dataset.__hash__()} successful, code {response.status_code}"
            )
            state = AttemptState.SUCCEEDED
        else:
            log.info(
                f"{attempt.dataset.__hash__()} failed, code {response.status_code}"
            )
            state = AttemptState.FAILED
        return UploadAttempt(attempt.dataset, state, response)

    return upload_dataset


async def ingest_chunked(
    dataset_generator: Iterable, ingestion_func: Callable, chunk_size
):
    responses = []
    for dataset_chunk in batched(dataset_generator, chunk_size):
        chunk_responses = await asyncio.gather(
            *(ingestion_func(f) for f in dataset_chunk)
        )
        responses += chunk_responses
    return responses


async def retry_if_bad(attempt: UploadAttempt, ingestion_func):
    if attempt.state == AttemptState.SUCCEEDED:
        return attempt
    else:
        if attempt.state == AttemptState.FAILED:
            log.info(f"Retrying {attempt.dataset.__hash__()}")
        return await ingestion_func(attempt)


async def ingest_with_retry(
    dataset_generator: Iterable, ingestion_func, chunk_size, retry_count=0
):
    attempts = await ingest_chunked(dataset_generator, ingestion_func, chunk_size)
    if retry_count >= 0 and not all(
        attempt.state == AttemptState.SUCCEEDED for attempt in attempts
    ):
        attempts = await ingest_with_retry(
            attempts, ingestion_func, chunk_size, retry_count - 1
        )
    return attempts


async def do_upload(
    dataset_generator,
    base_url,
    admin_token,
    chunk_size,
    retry_count,
    local_request_cache_folder=None,
) -> list[UploadAttempt]:
    log.info(
        f"Preparing to upload datasets to {base_url} "
        f"chunk_size {chunk_size} retry_count {retry_count}"
    )

    async with httpx.AsyncClient() as client:
        ingestor = ingestor_factory(
            client, base_url, admin_token, local_request_cache_folder
        )
        if retry_count:
            ingestor = partial(retry_if_bad, ingestion_func=ingestor)
        upload_attempts = (UploadAttempt(ds) for ds in dataset_generator)
        final_results = await ingest_with_retry(
            upload_attempts, ingestor, chunk_size, retry_count
        )

    successful_uploads = sum(
        1 for _ in filter(lambda x: x.state == AttemptState.SUCCEEDED, final_results)
    )
    failed_uploads = sum(
        1 for _ in filter(lambda x: x.state == AttemptState.FAILED, final_results)
    )
    log.info(
        f"Upload complete: {successful_uploads} successful, {failed_uploads} failed"
    )
    if failed_uploads:
        log.warning(f"{failed_uploads} failed: see debug for info")
        # TODO more info about failures
    return final_results


async def set_scopes(mdid_list: list[int], scope: str, base_url: str, admin_token: str):
    async with httpx.AsyncClient() as client:
        mdid_request_generator = (
            client.post(
                url=f"{base_url}/scope_mapping",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                },
                json={"scope": scope, "mdid": mdid},
                follow_redirects=True,
            )
            for mdid in mdid_list
        )
        responses = await asyncio.gather(*mdid_request_generator)
    return responses


def store_mdids(attempts: list[UploadAttempt], mdid_folder: Path):
    for attempt in attempts:
        if attempt.state != AttemptState.SUCCEEDED:
            raise Exception(
                "Attempting to save unsuccessful mdid. This shouldn't happen."
            )
        metadata = attempt.dataset.request["metadata"]
        metadata["mdid"] = int(attempt.response.text)
        filename = f"{metadata['indicator_id']}_{metadata['dt']}.json"
        (mdid_folder / filename).write_text(json.dumps(metadata))
