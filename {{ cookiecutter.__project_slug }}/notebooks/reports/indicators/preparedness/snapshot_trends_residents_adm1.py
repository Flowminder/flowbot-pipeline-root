import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression

from indicators.preparedness.monthly_residents_adm1 import MonthlyResidentsAdm1


class RansacError(Exception):
    pass


class SnapshotTrendsResidentsAdm1(MonthlyResidentsAdm1):

    def _compute_trend_and_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []

        for loc, group in df.groupby(
            self.aggregates.config.shapefile_regional_spatial_name
        ):

            # Ensure data is sorted by date
            group = group.sort_values(self.aggregates.config.date_column)

            # Filter data for the last 12 months
            max_date = group[self.aggregates.config.date_column].max()
            min_date = max_date - pd.DateOffset(months=12)
            group = group[group[self.aggregates.config.date_column] >= min_date]
            print(group)

            # Extract date and values
            dates = pd.to_datetime(group[self.aggregates.config.date_column])
            X = (dates - dates.min()).dt.days.values.reshape(
                -1, 1
            )  # Convert dates to ordinal
            y = group["residents"].values
            print("Admin 1 residents dataset")
            print(y)
            try:
                # Apply RANSAC
                # regressor = RANSACRegressor()
                regressor = LinearRegression()
                regressor.fit(X, y)
                y_pred = regressor.predict(X)
            except ValueError:
                results.append(
                    {
                        self.aggregates.config.shapefile_regional_spatial_name: loc,
                        "Trend_Slope": np.nan,
                        "Trend_Intercept": np.nan,
                        "Std_Residuals": np.nan,
                        "Percent_Change_Trend": np.nan,
                        "Percent_Change_Actual": np.nan,
                        "Avg_First_Three_Months": np.nan,
                        "Avg_Last_Three_Months": np.nan,
                        "Unusual": np.nan,
                        "Errored": True,
                    }
                )
                continue
                # raise RansacError

            # Calculate residuals
            residuals = y - y_pred
            std_residuals = np.std(residuals)

            # Calculate percentage change based on the trend
            trend_start = y_pred[0]
            trend_end = y_pred[-1]
            percent_change_trend = ((trend_end - trend_start) / trend_start) * 100

            # Calculate percentage change based on actual values
            actual_start = y[0]
            actual_end = y[-1]
            percent_change_actual = ((actual_end - actual_start) / actual_start) * 100

            # Calculate average residents in the first three months and the last three months
            first_three_months = dates <= (dates.min() + pd.DateOffset(months=3))
            last_three_months = dates >= (dates.max() - pd.DateOffset(months=3))
            avg_first_three_months = y[first_three_months].mean()
            avg_last_three_months = y[last_three_months].mean()

            # Check for unusual circumstances
            unusual = (avg_first_three_months < avg_last_three_months) and (
                percent_change_trend < 0
            )

            # Store the results
            results.append(
                {
                    self.aggregates.config.shapefile_regional_spatial_name: loc,
                    "Trend_Slope": regressor.coef_[0],
                    "Trend_Intercept": regressor.intercept_,
                    "Std_Residuals": std_residuals,
                    "Percent_Change_Trend": percent_change_trend,
                    "Percent_Change_Actual": percent_change_actual,
                    "Avg_First_Three_Months": avg_first_three_months,
                    "Avg_Last_Three_Months": avg_last_three_months,
                    "Unusual": unusual,
                    "Errored": False,
                }
            )

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def calculate(self) -> pd.DataFrame:

        monthly_residents_adm1 = super().calculate().reset_index()

        trends = self._compute_trend_and_residuals(monthly_residents_adm1)

        return trends
