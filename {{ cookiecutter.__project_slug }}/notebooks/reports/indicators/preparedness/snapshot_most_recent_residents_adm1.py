from indicators.preparedness.monthly_residents_adm1 import MonthlyResidentsAdm1


class SnapshotMostRecentResidentsAdm1(MonthlyResidentsAdm1):

    def calculate(self):

        monthly_residents_adm1 = super().calculate()
        most_recent_residents_adm1 = (
            monthly_residents_adm1.groupby(
                self.aggregates.config.shapefile_regional_spatial_name
            )
            .last()
            .rename("SnapshotMostRecentResidentsAdm1")
        )

        return most_recent_residents_adm1
