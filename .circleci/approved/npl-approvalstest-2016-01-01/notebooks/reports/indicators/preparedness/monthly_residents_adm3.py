import geopandas as gpd
import pandas as pd

from indicators.base.indicator_base import IndicatorBase


class MonthlyResidentsAdm3(IndicatorBase):

    def _impute_gaps_in_pcods_with_fill(
        self, pcod_df: pd.DataFrame, dates: pd.core.arrays.datetimes.DatetimeArray
    ) -> pd.DataFrame:
        # If a location has data only on some months then we need to fill empty months with 'something' -
        # otherwise population appears to grow / decline but this is soley driven by which pcods have data that month.
        # Most sensible approach is to bring population estimate forwards -
        # (i.e. if data on month X-1, and no data on month X then keep pop static)
        residents_in_pcod_filled = (
            pcod_df.set_index(self.aggregates.config.date_column)
            .reindex(dates)
            .ffill()["residents"]
            .bfill()
        )
        return residents_in_pcod_filled

    def _impute_missing_pcods_with_static(
        self,
        residents: pd.DataFrame,
        admin3_shapefile: gpd.GeoDataFrame,
        projected_population: pd.DataFrame,
    ) -> pd.DataFrame:
        # Some locations have no estimates ever (due to lack of CDR data in this area.
        # But, where we have static estimates we should add these to the platform aggregates as a static value.
        pcods_on_platform = residents[
            self.aggregates.config.monthly_residents_spatial_id
        ].unique()
        pcods_not_on_platform = admin3_shapefile.loc[
            ~admin3_shapefile[self.aggregates.config.shapefile_spatial_id].isin(
                pcods_on_platform
            ),
            self.aggregates.config.shapefile_spatial_id,
        ]

        static_residents_in_missing_areas = (
            projected_population.query(
                f"{self.aggregates.config.projected_population_spatial_id} in @pcods_not_on_platform"
            )
            .filter(
                regex=f"{self.aggregates.config.projected_population_spatial_id}|est_pop_2020_01"
            )
            .rename(
                columns={
                    "est_pop_2020_01": "residents",
                    self.aggregates.config.projected_population_spatial_id: self.aggregates.config.monthly_residents_spatial_id,
                }
            )
        )

        date_list = residents[self.aggregates.config.date_column].unique()

        missing_locs_static_est = static_residents_in_missing_areas.loc[
            static_residents_in_missing_areas.index.repeat(len(date_list))
        ].reset_index(drop=True)

        missing_locs_static_est[self.aggregates.config.date_column] = list(
            date_list
        ) * len(static_residents_in_missing_areas)

        missing_locs_static_est["is_static"] = True
        residents["is_static"] = False

        # append the missing pcods with static estimates to the dataframe from platform, grab relevant columns
        if len(missing_locs_static_est) > 0:
            residents = pd.concat(
                [
                    residents[
                        [
                            self.aggregates.config.monthly_residents_spatial_id,
                            "residents",
                            self.aggregates.config.date_column,
                            "is_static",
                        ]
                    ],
                    missing_locs_static_est,
                ]
            )
        else:
            residents = residents[
                [
                    self.aggregates.config.monthly_residents_spatial_id,
                    "residents",
                    self.aggregates.config.date_column,
                    "is_static",
                ]
            ]

        residents = residents.merge(
            admin3_shapefile,
            left_on=self.aggregates.config.monthly_residents_spatial_id,
            right_on=self.aggregates.config.shapefile_spatial_id,
        )

        return residents

    def calculate(self) -> pd.DataFrame:

        residents_agg = self.aggregates.fetch("monthly_residents")
        admin3_shapefile = self.aggregates.fetch("admin3_shapefile")
        projected_population = self.aggregates.fetch("projected_population")

        # fill gaps in pcods for which we do have data
        residents = (
            residents_agg.sort_values(self.aggregates.config.date_column)
            .groupby(self.aggregates.config.monthly_residents_spatial_id)
            .apply(self._impute_gaps_in_pcods_with_fill, residents_agg.date.unique())
            .stack()
            .reset_index()
            .rename(columns={0: "residents"})
        )

        # add pcods for which we have 0 data using static estimates
        residents = self._impute_missing_pcods_with_static(
            residents,
            admin3_shapefile,
            projected_population,
        )

        residents = residents.reset_index().merge(
            residents_agg[
                [
                    self.aggregates.config.monthly_residents_spatial_id,
                    "abnormality",
                    self.aggregates.config.date_column,
                ]
            ],
            on=[
                self.aggregates.config.monthly_residents_spatial_id,
                self.aggregates.config.date_column,
            ],
        )

        return residents[
            [
                "residents",
                self.aggregates.config.date_column,
                self.aggregates.config.shapefile_spatial_name,
                self.aggregates.config.shapefile_spatial_id,
                self.aggregates.config.shapefile_regional_spatial_name,
                "is_static",
                "abnormality",
            ]
        ]
