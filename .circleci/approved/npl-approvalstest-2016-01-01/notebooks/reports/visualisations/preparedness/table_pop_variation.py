import geopandas as gpd
import pandas as pd
from visualisations.base.table import Table
from visualisations.base.figure import Figure

from slugify import slugify


class TablePopVariation(Table):

    def draw(self, indicators):

        trends = indicators["SnapshotTrendsResidentsAdm3"]
        date = self.indicators.date

        whole_country = []
        for adm1, adm1_df in trends.groupby(
            self.indicators.config.shapefile_regional_spatial_name
        ):
            # TODO: Column names (Department, Section, ect) to be set in CountryConfig
            incr_decr = (
                adm1_df.query(
                    "(abs(Absolute_Change_Trend) > 100) and (has_abnormal_change == False) and (Fluctuating == False) and (Unusual == False) and (has_data)"
                )[
                    [
                        self.indicators.config.shapefile_regional_spatial_name,
                        self.indicators.config.shapefile_spatial_name,
                        "most_recent_pop",
                        "Absolute_Monthly_Change_Trend",
                    ]
                ]
                .assign(
                    Absolute_Monthly_Change_Trend=adm1_df[
                        "Absolute_Monthly_Change_Trend"
                    ]
                    * 12
                )
                .round({"most_recent_pop": -2, "Absolute_Monthly_Change_Trend": -1})
                .rename(
                    columns={
                        self.indicators.config.shapefile_regional_spatial_name: "Department",
                        self.indicators.config.shapefile_spatial_name: "Section",
                        "most_recent_pop": f'Population {pd.to_datetime(date).strftime("%B %Y")}',
                        "Absolute_Monthly_Change_Trend": "Average change",
                    }
                )
            )
            incr_decr["Variation"] = incr_decr.apply(
                lambda z: "Decreasing" if z["Average change"] < 0 else "Increasing",
                axis=1,
            )

            fluct = (
                adm1_df.query(
                    "(has_abnormal_change or Fluctuating or Unusual) and (has_data)"
                )[
                    [
                        self.indicators.config.shapefile_regional_spatial_name,
                        self.indicators.config.shapefile_spatial_name,
                        "most_recent_pop",
                        "largest_abnormal_fluct",
                    ]
                ]
                .round({"most_recent_pop": -2, "largest_abnormal_fluct": -1})
                .rename(
                    columns={
                        self.indicators.config.shapefile_regional_spatial_name: "Department",
                        self.indicators.config.shapefile_spatial_name: "Section",
                        "most_recent_pop": f'Population {pd.to_datetime(date).strftime("%B %Y")}',
                        "largest_abnormal_fluct": "Largest fluctuation",
                    }
                )
            ).assign(Variation="Fluctuating")

            fluct.loc[fluct["Largest fluctuation"] < 100, "Largest fluctuation"] = (
                "< 100"
            )

            whole_country.append(
                pd.concat([incr_decr, fluct])
                .sort_values(
                    f'Population {pd.to_datetime(date).strftime("%B %Y")}',
                    ascending=False,
                )
                .reset_index(drop=True)
            )

        whole_country_df = pd.concat(whole_country)[
            [
                "Department",
                "Section",
                f'Population {pd.to_datetime(date).strftime("%B %Y")}',
                "Variation",
                "Average change",
                "Largest fluctuation",
            ]
        ]
        print("")
        print(whole_country_df)

        # Set data types for columns to avoid mixed types
        whole_country_df = whole_country_df.astype(
            {
                "Department": "str",
                "Section": "str",
                # f'Population {pd.to_datetime(date).strftime("%B %Y")}': 'int64',
                "Variation": "str",
                "Average change": "float64",
                "Largest fluctuation": "str",  # Convert to string
            }
        )

        region = "national"

        yield Figure(
            csv=whole_country_df,
            caption="",
            title="",
            filepath=slugify(region) + "/" + self.__class__.__name__,
        )
