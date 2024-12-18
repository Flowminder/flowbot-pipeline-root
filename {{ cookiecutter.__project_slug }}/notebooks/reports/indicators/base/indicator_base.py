from typing import Union

import geopandas as gpd
import pandas as pd

from utils.aggregate_loader import AggregateCollection


class IndicatorBase:
    def __init__(self, aggregates: AggregateCollection):
        self.aggregates = aggregates

    def calculate(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        raise NotImplementedError("")

    def save_result(
        self,
        result: Union[pd.DataFrame, gpd.GeoDataFrame],
        hdf5_file: str,
        indicator_name: str = None,
    ) -> None:
        if indicator_name is None:
            indicator_name = self.__class__.__name__
        result.to_hdf(hdf5_file, key=indicator_name, mode="a")
