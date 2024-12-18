# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import datetime
from pathlib import Path

import httpx
import json
import logging

log = logging.getLogger("upload_notebook")


def get_token(domain, client_id, client_secret, audience, cache_folder=None):
    if cache_folder:
        token_path = timestamp_path(client_id, cache_folder)
        if token_path.exists():
            log.info(f"Token exists at {token_path}, loading from cache")
            return json.loads(token_path.read_text())
    token = request_token(audience, client_id, client_secret, domain)
    if cache_folder:
        token_path.write_text(json.dumps(token))
        log.info(f"Caching token at {token_path}")
    return token


def request_token(audience, client_id, client_secret, domain):
    log.info("Requesting new token")
    response = httpx.post(
        url=f"https://{domain}/oauth/token",
        headers={"Content-Type": "application/json"},
        data=f'{{"client_id":"{client_id}","client_secret":"{client_secret}","audience":"{audience}","grant_type":"client_credentials"}}',
    )
    log.info(response)
    if response.status_code >= 400:
        raise Exception("Bad token request")
    return json.loads(response.content)["access_token"]


def timestamp_path(client_id, cache_folder) -> Path:
    # Renews a token every hour
    now = datetime.datetime.now()
    time_slice = datetime.datetime(
        year=now.year, month=now.month, day=now.day, hour=now.hour
    )
    return Path(cache_folder) / f"{time_slice.timestamp()}-{client_id}-token"
