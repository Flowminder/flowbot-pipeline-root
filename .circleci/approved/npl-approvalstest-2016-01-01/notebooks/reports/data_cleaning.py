# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import dataclasses
import json
import os
from pathlib import Path
from typing import Iterator, Callable, Collection
import logging

import pandas as pd
import datetime

DATA_VERSION = os.getenv("DATA_VERSION")

log = logging.getLogger("upload_notebook")


class ConfigMismatchError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class CsvDataset:
    path: Path
    category_id: str
    indices: tuple
    srid_col: str
    trid_col: str
    indicators: list[str] = dataclasses.field(default_factory=list)
    renames: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class PresenceDataset(CsvDataset):
    @staticmethod
    def renames_default():
        return {"pcod": "spatial_unit"}

    @staticmethod
    def indicators_default():
        return [
            "presence",
            "presence_perKm2",
            "trips_in",
            "trips_out",
            "presence_diffwithref",
            "presence_pctchangewithref",
            "abnormality",
        ]

    category_id: str = "presence"
    indices: tuple = ("date", "spatial_unit")
    srid_col: str = "Communal section"
    trid_col: str = "days"
    indicators: list[str] = dataclasses.field(default_factory=indicators_default)
    renames: dict = dataclasses.field(default_factory=renames_default)


@dataclasses.dataclass(frozen=True)
class ResidentsDataset(CsvDataset):
    @staticmethod
    def renames_default():
        return {"pcod": "spatial_unit"}

    @staticmethod
    def indicators_default():
        return [
            "delta_arrived",
            "residents_pctchangewithref",
            "residents",
            "residents_diffwithref",
            "departed",
            "arrived",
            "residents_perKm2",
            "abnormality",
        ]

    category_id: str = "residents"
    indices: tuple = ("date", "spatial_unit")
    srid_col: str = "Communal section"
    trid_col: str = "months"
    indicators: list[str] = dataclasses.field(default_factory=indicators_default)
    renames: dict = dataclasses.field(default_factory=renames_default)


@dataclasses.dataclass(frozen=True)
class RelocationsDataset(CsvDataset):
    @staticmethod
    def renames_default():
        return {"month": "date", "pcod_from": "origin", "pcod_to": "destination"}

    @staticmethod
    def indicators_default():
        return [
            "relocations",
            "relocations_diffwithref",
            "relocations_pctchangewithref",
            "abnormality",
        ]

    category_id: str = "relocations"
    indices: tuple = ("date", "origin", "destination")
    srid_col: str = "Communal section"
    trid_col: str = "months"
    indicators: list[str] = dataclasses.field(default_factory=indicators_default)
    renames: dict = dataclasses.field(default_factory=renames_default)


@dataclasses.dataclass(frozen=True)
class MovementsDataset(CsvDataset):
    @staticmethod
    def renames_default():
        return {"month": "date", "pcod_from": "origin", "pcod_to": "destination"}

    @staticmethod
    def indicators_default():
        return [
            "travellers",
            "travellers_diffwithref",
            "travellers_pctchangewithref",
            "abnormality",
        ]

    category_id: str = "movements"
    indices: tuple = ("date", "origin", "destination")
    srid_col: str = "Communal section"
    trid_col: str = "days"
    indicators: list[str] = dataclasses.field(default_factory=indicators_default)
    renames: dict = dataclasses.field(default_factory=renames_default)


@dataclasses.dataclass
class DataRecord:
    spatial_unit_ids: list[str]
    data: float


@dataclasses.dataclass(kw_only=True)
class JsonDataset:
    revision: str
    date_added: str
    category_id: str
    indicator_id: str
    srid: int
    trid: int
    dt: str
    data_type: str
    data_input: list[DataRecord]

    def __post_init__(self):
        self.filename = self._get_filename()
        self.request = self._build_request()
        self.hash = hash(json.dumps(self.request))

    def _get_filename(self):
        return (
            "_".join(
                [
                    self.category_id,
                    self.indicator_id,
                    str(self.srid),
                    str(self.trid),
                    self.dt,
                    self.revision,
                ]
            )
            + ".json"
        )

    def _build_request(self):
        dict_ = dataclasses.asdict(self)
        metadata_items = {
            k: v for k, v in dict_.items() if k not in ["data_type", "data_input"]
        }
        data_items = {
            k: v for k, v in dict_.items() if k in ["data_type", "data_input"]
        }
        return {"metadata": metadata_items, **data_items}

    def __hash__(self):
        return self.hash


def csv_to_dataset(
    dataset: CsvDataset,
    srid_lookup: Callable,
    trid_lookup: Callable,
    category_type_lookup: Callable,
    redactor: Callable,
    revision: str,
    indicators: Collection[str],
) -> Iterator[JsonDataset]:
    with pd.option_context("use_inf_as_na", True):
        log.debug(f"Converting {dataset.path} to JsonDataset")
        df = pd.read_csv(dataset.path)
        df = df.rename(columns=dataset.renames)
        df["date"] = pd.to_datetime(df.date)
        df = df.set_index(
            keys=list(dataset.indices)
        )  # !"£$ing Pandas, what's wrong with a tuple
        df = df.sort_index()

        df = redactor(df)
        if df.empty:
            log.warning(
                f"All columns empty for {dataset.path}. No datasets will be generated."
            )

        log.debug(df.columns)

        for column in set(df.columns).intersection(set(indicators)):
            log.debug(f"Writing {column}")
            for this_date, this_df in df[[column]].groupby("date"):
                indicator_df = this_df.dropna()
                if indicator_df.empty:
                    log.warning(f"{column} for {this_date} empty, no dataset generated")
                    continue

                # This assumes that 'date' is always the first index
                spatial_unit_ids = [i[1:] for i in indicator_df.index.tolist()]

                # We need to flatten the output from pandas for some reason....
                values = [v[0] for v in indicator_df.values.tolist()]

                data_records = list(
                    DataRecord(sids, value)
                    for sids, value in zip(spatial_unit_ids, values)
                )

                yield JsonDataset(
                    revision=revision,
                    date_added=str(
                        datetime.datetime.now(datetime.timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S"
                        )
                    ),
                    category_id=dataset.category_id,
                    indicator_id=f"{dataset.category_id}.{column}",
                    srid=int(srid_lookup(dataset.srid_col)),
                    trid=int(trid_lookup(dataset.trid_col)),
                    dt=this_date.strftime("%Y-%m-%dT%H:%M:%S"),
                    data_type=category_type_lookup(dataset.category_id),
                    data_input=data_records,
                )


def validate_indicator(category: CsvDataset, config_path: Path):
    with open(config_path) as fp:
        config = json.load(fp)
    category_indicators = set(
        f"{category.category_id}.{i}" for i in category.indicators
    )
    config_indicators = set(
        ind["indicator_id"]
        for ind in config["indicators"]
        if ind["category_id"] == category.category_id
    )
    if not category_indicators.issubset(config_indicators):
        raise ConfigMismatchError(
            f"{category_indicators.difference(config_indicators)} do not appear in config"
        )
