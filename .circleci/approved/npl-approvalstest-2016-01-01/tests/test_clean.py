# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import json

import pytest
from notebooks.upload.data_cleaning import (
    csv_to_dataset,
    RelocationsDataset,
    ResidentsDataset,
    validate_indicator,
    ConfigMismatchError,
)


@pytest.fixture
def sample_reloc_string():
    yield """date,pcod_from,pcod_to,relocations,relocations_diffwithref,abnormality,relocations_pctchangewithref
2020-03-01,HT0111-01,HT1032-01,3.0,,,
2020-03-01,HT0111-01,HT1031-04,1.0,-3.0,,-75.0
2020-03-01,HT0111-01,HT1031-03,11.0,9.0,,450.0
2020-03-01,HT0111-01,HT1025-03,2.0,0.0,,0.0
2020-03-01,HT0111-01,HT1025-02,6.0,0.0,,0.0
2020-03-01,HT0111-01,HT1024-03,12.0,9.0,,300.0
2020-03-01,HT0111-01,HT1024-01,17.0,9.0,,112.5
2020-03-01,HT0111-01,HT1023-04,1.0,-1.0,,-50.0
2020-03-01,HT0111-01,HT1023-03,13.0,4.0,,44.44444444444444
"""


@pytest.fixture
def sample_residents_string():
    yield """date,pcod,residents,residents_perKm2,arrived,departed,delta_arrived,residents_diffwithref,abnormality,residents_pctchangewithref
2020-03-01,HT0111-01,615312.0,33499.57566101726,10369.0,12507.0,-2138.0,2721.25,0.3760682899478505,0.4442198972152289
2020-03-01,HT0111-02,185724.0,26821.78022073955,1142.0,812.0,330.0,1369.2699999999895,2.1969627895107204,0.7427365709575173
2020-03-01,HT0111-03,342412.0,38213.5338723206,5099.0,3340.0,1759.0,2334.5599999999977,18.127817598799115,0.6864789384441372
2020-03-01,HT0112-01,458469.0,17585.503746654158,13261.0,15142.0,-1881.0,2662.1900000000023,0.34482557314865503,0.5840610411239802
2020-03-01,HT0113-01,983.0,227.36229278363498,,,,-1.5271599999999808,-3.2334510731447725,-0.15511608638607602
2020-03-01,HT0113-02,956.0,75.13103424484376,,,,-2.1931200000000217,-1.805124324376261,-0.22888079179696277
2020-03-01,HT0113-03,682.0,66.22276232654508,,,,-1.1469700000000103,-9.853108973759076,-0.16789505777944244
2020-03-01,HT0113-04,918.0,52.60646853756512,,,,-1.4682000000000244,-3.555698617129565,-0.15967925807548586
2020-03-01,HT0113-05,1186.0,91.81872095322062,,,,-1.5636999999999261,-3.067574880379937,-0.1316729367864584

    """


@pytest.fixture
def sample_reloc_file(sample_reloc_string, tmp_path):
    csv_path = tmp_path / "reloc.csv"
    csv_path.write_text(sample_reloc_string)
    yield csv_path


@pytest.fixture
def sample_residents_file(sample_residents_string, tmp_path):
    csv_path = tmp_path / "res.csv"
    csv_path.write_text(sample_residents_string)
    yield csv_path


def test_clean_flow(sample_reloc_file):
    sample_reloc = RelocationsDataset(sample_reloc_file)
    cleaned_data_generator = csv_to_dataset(
        sample_reloc,
        srid_lookup=lambda _: "dummy_srid",
        trid_lookup=lambda _: "dummy_trid",
        category_type_lookup=lambda _: "dummy_category",
    )
    data_classes = list(cleaned_data_generator)
    assert len(data_classes) == 4
    relocations = data_classes[0]
    assert len(relocations.data_input) == 9

    # Sorted by index, so the last csv row will be the first data_input
    assert relocations.data_input[8].spatial_unit_ids == ("HT0111-01", "HT1032-01")
    assert relocations.data_input[8].data_point == 3

    abnormalities = data_classes[2]
    assert len(abnormalities.data_input) == 0

    reloc_pct = data_classes[3]
    assert len(reloc_pct.data_input) == 8
    assert reloc_pct.data_input[7].data_point == -75.0
    assert ("HT0111-01", "HT1032-01") not in (
        di.spatial_unit_ids for di in reloc_pct.data_input
    )


def test_clean_static(sample_residents_file):
    sample_reloc = ResidentsDataset(sample_residents_file)
    cleaned_data_generator = csv_to_dataset(
        sample_reloc,
        srid_lookup=lambda _: "dummy_srid",
        trid_lookup=lambda _: "dummy_trid",
        category_type_lookup=lambda _: "dummy_category",
    )
    data_classes = list(cleaned_data_generator)
    assert len(data_classes) == 8
    relocations = data_classes[0]
    assert len(relocations.data_input) == 9
    assert len(relocations.data_input[0].spatial_unit_ids) == 1
    assert relocations.data_input[0].data_point == 615312.0
    # TODO: Approvaltest here
    # TODO: Parameterise to test with the other data types


def test_upload_columns(sample_residents_file):
    sample_res = ResidentsDataset(sample_residents_file)
    choice_data_generator = csv_to_dataset(
        sample_res,
        srid_lookup=lambda _: "1",
        trid_lookup=lambda _: "1",
        category_type_lookup=lambda _: "dummy_category",
        redactor=lambda x: x,
        revision="UNIT_TEST",
        indicators=["residents", "residents_perKm2"],
    )
    data_classes = list(choice_data_generator)
    assert sorted([dc.indicator_id for dc in data_classes]) == [
        "residents.residents",
        "residents.residents_perKm2",
    ]


@pytest.fixture
def minimal_config_file(tmp_path):
    with open(tmp_path / "conf.json", "w") as fp:
        json.dump(
            {
                "indicators": [
                    {"category_id": "residents", "indicator_id": "residents.residents"},
                    {
                        "category_id": "residents",
                        "indicator_id": "residents.madeup_shouldntsee",
                    },
                ]
            },
            fp,
        )
    yield tmp_path / "conf.json"


def test_validate_config(sample_residents_file, minimal_config_file):
    bad_indicator = ResidentsDataset(
        sample_residents_file, indicators=["residents", "arrived"]
    )
    with pytest.raises(ConfigMismatchError) as e_info:
        validate_indicator(bad_indicator, minimal_config_file)
        assert "arrived" in e_info
    good_indicator = ResidentsDataset(sample_residents_file, indicators=["residents"])
    validate_indicator(good_indicator, minimal_config_file)
