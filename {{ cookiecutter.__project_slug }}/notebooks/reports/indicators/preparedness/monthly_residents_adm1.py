import pandas as pd

from indicators.preparedness.monthly_residents_adm3 import MonthlyResidentsAdm3


class MonthlyResidentsAdm1(MonthlyResidentsAdm3):

    def calculate(self) -> pd.DataFrame:
        monthly_residents_adm3 = super().calculate()

        monthly_residents_adm1 = (
            monthly_residents_adm3.groupby(
                [
                    self.aggregates.config.shapefile_regional_spatial_name,
                    self.aggregates.config.date_column,
                ]
            )
            .residents.sum()
            .sort_index()
        )

        return monthly_residents_adm1
