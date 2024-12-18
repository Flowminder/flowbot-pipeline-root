from indicators.preparedness.monthly_residents_adm1 import MonthlyResidentsAdm1


class SnapshotMedianResidentsAdm1(MonthlyResidentsAdm1):

    def calculate(self):

        monthly_residents_adm1 = super().calculate()
        median_residents_adm1 = (
            monthly_residents_adm1.groupby(
                self.aggregates.config.shapefile_regional_spatial_name
            )
            .median()
            .rename("SnapshotMedianResidentsAdm1")
        )

        return median_residents_adm1
