import pandas as pd
import geopandas as gpd

from utils.country_config import CountryConfig


class IndicatorLoader:

    def __init__(
        self,
        file_path: str,
        config: CountryConfig,
        date: str,
        shapefile: gpd.GeoDataFrame,
    ):

        self.filepath = file_path
        self.config = config
        self.date = pd.to_datetime(date)
        self.shapefile = shapefile

    def open(self):
        return pd.HDFStore(self.filepath, mode="r")
