import pandas as pd
from indicators.base.indicator_base import IndicatorBase
from indicators.preparedness.monthly_residents_adm3 import MonthlyResidentsAdm3
from indicators.preparedness.snapshot_trends_residents_adm3 import (
    SnapshotTrendsResidentsAdm3,
)


class MonthlyTop6Changes(IndicatorBase):
    def __init__(self, aggregates):
        self.aggregates = aggregates
        self.monthly_residents_adm3 = MonthlyResidentsAdm3(aggregates)
        self.snapshot_trends_residents_adm3 = SnapshotTrendsResidentsAdm3(aggregates)

    def calculate(self) -> pd.DataFrame:
        # Get the interpolated resident counts from the MonthlyResidentsAdm3 class
        monthly_residents_adm3 = self.monthly_residents_adm3.calculate()

        # Get the trends from the SnapshotTrendsResidentsAdm3 class
        trends = self.snapshot_trends_residents_adm3.calculate()

        # Filter trends to show only fluctuating entries
        fluctuating_pcods = trends.query("Fluctuating | has_abnormal_change")[
            self.aggregates.config.shapefile_spatial_id
        ].values

        # only consider adm3 who are fluctuating for the infoboxes
        monthly_residents_adm3 = monthly_residents_adm3[
            monthly_residents_adm3[self.aggregates.config.shapefile_spatial_id].isin(
                fluctuating_pcods
            )
        ]

        def get_top_6_adm1(adm1_df):
            # Filter out entries with abnormality values >= 15 and remove the first row per group
            filtered_values = (
                adm1_df.set_index(
                    [
                        self.aggregates.config.shapefile_spatial_id,
                        self.aggregates.config.date_column,
                    ]
                )
                .query("(abs(abnormality) <= 15) and (abs(abnormality) >= 0)")
                .groupby(level=[self.aggregates.config.shapefile_spatial_id])
                .apply(lambda group: group.iloc[1:])
                .reset_index(level=0, drop=True)
            )

            filtered_indexes = filtered_values.index

            # Calculate the difference in residents month-over-month for each ADM3_PCODE
            resident_diffs = (
                adm1_df.set_index(self.aggregates.config.date_column)
                .groupby(self.aggregates.config.shapefile_spatial_id)
                .residents.apply(lambda group: group.diff().dropna())
            )

            # Extract resident counts for the filtered indexes
            residents = adm1_df.set_index(
                [
                    self.aggregates.config.shapefile_spatial_id,
                    self.aggregates.config.date_column,
                ]
            ).residents

            # Select the top 6 largest changes in absolute numbers, ensuring we pick places with changes
            top_changes = (
                (resident_diffs.loc[filtered_indexes].sort_values(key=abs))
                .to_frame()
                .query("abs(residents) > 0")
            )

            return top_changes.join(
                filtered_values[
                    [self.aggregates.config.shapefile_spatial_name, "abnormality"]
                ]
            ).tail(6)

        return (
            monthly_residents_adm3.groupby("shapefile_regional_spatial_name").apply(
                get_top_6_adm1
            )
        ).reset_index()
