# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import asyncio
import dataclasses
import gzip
import itertools
import json
from functools import partial

import httpx
import pytest

from notebooks.data_cleaning import JsonDataset
from notebooks.upload import (
    ingest_chunked,
    ingest_with_retry,
    UploadAttempt,
    retry_if_bad,
    AttemptState,
    do_upload,
)


class MockDataset(JsonDataset):
    mock_json: dict

    def __init__(self, mock_json):
        self.mock_json = mock_json

        super().__init__(
            revision="",
            date_added="",
            category_id="",
            indicator_id="",
            srid="",
            trid="",
            dt="",
            data_type="",
            data_input=[],
        )
        self.request = mock_json
        self.filename = "mock_file.json"
        self.hash = "12345"

    def request(self):
        return self.mock_json

    def __repr__(self):
        return f"{self.mock_json}"


@pytest.fixture
def dummy_data_generator():
    return (MockDataset({"value": x}) for x in range(10))


@pytest.fixture
def dummy_mixed_data_generator():
    return (MockDataset({"is_good": x % 2}) for x in range(10))


@pytest.fixture()
def good_response():
    return httpx.Response(status_code=201)


@pytest.fixture()
def bad_response():
    return httpx.Response(status_code=401)


@pytest.fixture()
def dummy_ingestion_func(good_response):
    """Emulates a dataset being sent to the backend"""

    async def inner(dataset):
        await asyncio.sleep(0)
        return good_response

    return inner


@pytest.fixture()
def dummy_smart_ingestion_func(good_response, bad_response):
    """Returns good or bad data depending on input"""

    async def inner(attempt: UploadAttempt):
        await asyncio.sleep(0)
        if attempt.dataset.request["is_good"]:
            return UploadAttempt(
                dataset=attempt.dataset,
                state=AttemptState.SUCCEEDED,
                response=good_response,
            )
        else:
            return UploadAttempt(
                dataset=attempt.dataset,
                state=AttemptState.FAILED,
                response=bad_response,
            )

    return inner


@pytest.mark.asyncio
async def test_ingest_chunked(dummy_data_generator, dummy_ingestion_func):
    out = await ingest_chunked(dummy_data_generator, dummy_ingestion_func, 3)
    assert len(out) == 10
    assert all(r.status_code == 201 for r in out)


@pytest.fixture()
def bad_attempt(bad_response):
    return UploadAttempt(
        dataset=MockDataset({"is_good": 0}),
        state=AttemptState.FAILED,
        response=bad_response,
    )


@pytest.fixture()
def good_attempt(good_response):
    return UploadAttempt(
        dataset=MockDataset({"is_good": 1}),
        state=AttemptState.SUCCEEDED,
        response=good_response,
    )


@pytest.fixture
def bad_good_target(bad_attempt, good_attempt):
    return [bad_attempt, good_attempt] * 5


@pytest.mark.parametrize(
    ["chunk_size", "retry_count"], ((1, 1), (5, 1), (1, 5), (5, 5))
)
@pytest.mark.asyncio
async def test_ingest(
    dummy_mixed_data_generator,
    dummy_smart_ingestion_func,
    chunk_size,
    retry_count,
    bad_good_target,
):
    dummy_ingestor = partial(retry_if_bad, ingestion_func=dummy_smart_ingestion_func)
    dummy_attempts = (UploadAttempt(ds) for ds in dummy_mixed_data_generator)
    out = await ingest_with_retry(
        dummy_attempts, dummy_ingestor, chunk_size=chunk_size, retry_count=retry_count
    )
    assert out == bad_good_target


@pytest.fixture
def mock_patch(good_response, bad_response):
    async def dummy_patch(self, url, headers, data, timeout):
        data = json.loads(gzip.decompress(data))
        if data["is_good"]:
            return good_response
        else:
            return bad_response

    return dummy_patch


@pytest.mark.asyncio
async def test_do_upload(
    dummy_mixed_data_generator, monkeypatch, mock_patch, bad_attempt, good_attempt
):
    monkeypatch.setattr(httpx.AsyncClient, "patch", mock_patch)
    out = await do_upload(
        dummy_mixed_data_generator, "dummy_base_url", "dummy_admin_token", None, 1, 1
    )
    assert out == [bad_attempt, good_attempt] * 5
