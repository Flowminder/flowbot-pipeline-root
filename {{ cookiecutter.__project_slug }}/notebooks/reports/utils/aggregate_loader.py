from dataclasses import dataclass
import glob
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeAlias
from functools import singledispatch

import geopandas as gpd
import pandas as pd
from dateutil.relativedelta import relativedelta

from utils.country_config import CountryConfig

AggregateDataFrame: TypeAlias = pd.DataFrame | gpd.GeoDataFrame


@dataclass(kw_only=True)
class AggregateSpec:
    name: str
    config: CountryConfig
    end_date: datetime
    agg_folder: Path
    file_glob: str
    lookback_months: int


@dataclass(kw_only=True)
class StaticDataSpec:
    name: str
    path: Path
    config: CountryConfig


@singledispatch
def load_data(data) -> AggregateDataFrame:
    raise TypeError("load_data only supports AggregateSpec and StaticDataSpec")


@load_data.register
def _(spec: AggregateSpec) -> AggregateDataFrame:
    agg_files = [a for a in spec.agg_folder.glob(spec.file_glob)]
    dates_to_match = [
        (spec.end_date - relativedelta(months=i)).strftime("%Y-%m")
        for i in range(spec.lookback_months)
    ]
    agg_files_before_date = [
        agg for agg in agg_files if any(date in agg.name for date in dates_to_match)
    ]
    print(f"Aggregates files for {spec.name}: {agg_files_before_date}")
    df_list = [_load_file(spec, file) for file in agg_files_before_date]
    out = pd.concat(df_list, ignore_index=True).sort_values(spec.config.date_column)

    # We shouldn't need to do this for any of the prep reports, but just in case.
    return _truncate_to_date_range(
        out, spec.end_date, spec.lookback_months, spec.config.date_column
    )


def _truncate_to_date_range(df, end_date, lookback_months, date_column):
    # Filter the data by date if applicable
    # (i.e. platform indicators that are all in one file, vs crisis which are a file per month)
    start_date = end_date - relativedelta(months=lookback_months)
    df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    return df


@load_data.register
def _(spec: StaticDataSpec) -> AggregateDataFrame:
    return _load_file(spec, spec.path)


def _load_file(spec, file_path: Path) -> AggregateDataFrame:
    """
    Loads a single file into a DataFrame or GeoDataFrame depending on the file extension,
    and filters by date if necessary.

    Parameters:
    file_path : str
        The path to the file to be loaded.

    Returns:
    DataFrame or GeoDataFrame
        The loaded data, filtered by date if applicable.
    """
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        if spec.config.date_column in df.columns:
            df[spec.config.date_column] = pd.to_datetime(df[spec.config.date_column])
    elif file_path.suffix == ".geojson":
        df = gpd.read_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return df


class AggregateCollection:
    config: CountryConfig

    def __init__(
        self, data_specs: list[AggregateSpec | StaticDataSpec], config: CountryConfig
    ):
        self._aggregate_dict = {spec.name: spec for spec in data_specs}
        self.config = config

    def fetch(self, aggregate_name: str) -> AggregateDataFrame:
        return load_data(self._aggregate_dict[aggregate_name])
