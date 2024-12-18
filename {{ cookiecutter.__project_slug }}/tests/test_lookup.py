# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import httpx

from notebooks.params import CategoryParams, lookup_factory


def dummy_get_srid(x, y):
    return (
        b"{"
        b'"spatial_resolutions":[{"srid":3,"label":"Communal section","index":3,"description":"A communal section is a third-level administrative division in Haiti.","boundaries":null,"label_fr":"Section communale","description_fr":"La section communale est une division administrative de troisi\xc3\xa8me niveau en Ha\xc3\xafti."}]}'
    )


def test_lookup_factory(monkeypatch):
    monkeypatch.setattr(httpx, "get", dummy_get_srid)
    func = lookup_factory(CategoryParams, "dummy_admin_token")
