from datetime import datetime


from indicators.base.indicator_base import IndicatorBase


class SnapshotUNPopulationProjectionAdm1(IndicatorBase):

    def calculate(self):

        projected_population = self.aggregates.fetch("projected_population")
        admin3_shapefile = self.aggregates.fetch("admin3_shapefile")

        date = self.aggregates.date
        current_year = date.year
        next_year = current_year + 1
        prev_year = current_year - 1

        projected_population_w_geodata = projected_population.merge(
            admin3_shapefile,
            left_on=self.aggregates.config.projected_population_spatial_id,
            right_on=self.aggregates.config.shapefile_spatial_id,
        )

        # now interpolate
        projected_population = projected_population_w_geodata.groupby(
            [self.aggregates.config.shapefile_regional_spatial_name]
        )[
            [
                f"est_pop_{prev_year}_01",
                f"est_pop_{current_year}_01",
                f"est_pop_{next_year}_01",
            ]
        ].sum()

        current_month = int(date.month)
        interpolated_projection_report_date = projected_population[
            f"est_pop_{current_year}_01"
        ] * (
            projected_population[f"est_pop_{next_year}_01"]
            / projected_population[f"est_pop_{current_year}_01"]
        ) ** (
            current_month / 12
        )
        interpolated_projection_report_date_minus_1_year = projected_population[
            f"est_pop_{prev_year}_01"
        ] * (
            projected_population[f"est_pop_{current_year}_01"]
            / projected_population[f"est_pop_{prev_year}_01"]
        ) ** (
            current_month / 12
        )

        proj_pop_column = "est_pop_" + datetime.strftime(date, "%Y_%m")
        proj_pop_minus_1_yr_column = (
            "est_pop_" + str(int(date.year) - 1) + "_" + datetime.strftime(date, "%m")
        )

        projected_population[proj_pop_column] = interpolated_projection_report_date

        projected_population[proj_pop_minus_1_yr_column] = (
            interpolated_projection_report_date_minus_1_year
        )

        projected_population = projected_population.assign(
            pct_change_proj=(
                100
                * (
                    interpolated_projection_report_date
                    - interpolated_projection_report_date_minus_1_year
                )
                / interpolated_projection_report_date
            )
        ).reset_index()

        return projected_population[
            [
                self.aggregates.config.shapefile_regional_spatial_name,
                proj_pop_minus_1_yr_column,
                proj_pop_column,
                "pct_change_proj",
            ]
        ].rename(
            columns={
                proj_pop_minus_1_yr_column: "proj_pop_start",
                proj_pop_column: "proj_pop_end",
            }
        )
