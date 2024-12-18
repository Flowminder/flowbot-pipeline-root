# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import os
from pathlib import Path

BASE_URL = "https://api.dev.haiti.mobility-dashboard.org/v1"
AUTH0_DOMAIN = "flowminder-dev.eu.auth0.com"
AUTH0_CLIENT_ID_ADMIN = os.getenv("ADMIN_CLIENT")
AUTH0_CLIENT_ID_UPDATER = os.getenv("UPDATER_CLIENT")
AUTH0_CLIENT_SECRET_ADMIN = os.getenv("ADMIN_SECRET")
AUTH0_CLIENT_SECRET_UPDATER = os.getenv("UPDATER_SECRET")
AUDIENCE = "https://flowkit-ui-backend.flowminder.org"
INDICATORS_FOLDER = Path(__file__).parent.parent / "tests" / "sample_indicators"
CONFIG_PATH = Path(__file__).parent / "config.json"
CHUNK_SIZE = 20
DATA_VERSION = os.getenv("DATA_VERSION")
UPLOAD_FOLDER = Path(__file__).parent / ".tmp"
REDACTED_ADMIN_3 = ["HT9999-9"]
