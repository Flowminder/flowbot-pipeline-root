from indicators.preparedness.monthly_residents_adm1 import MonthlyResidentsAdm1


class SnapshotDiffResidentsAdm1(MonthlyResidentsAdm1):

    def calculate(self):

        monthly_residents_adm1 = super().calculate()

        first_last = (
            monthly_residents_adm1.reset_index()
            .groupby(self.aggregates.config.shapefile_regional_spatial_name)
            .agg({"residents": ["first", "last"]})
        )

        return (
            first_last[("residents", "last")] - first_last[("residents", "first")]
        ).rename("SnapshotDiffResidentsAdm1")
