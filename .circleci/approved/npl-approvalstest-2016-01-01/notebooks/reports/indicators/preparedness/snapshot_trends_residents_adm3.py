import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import mean_squared_error

from indicators.preparedness.monthly_residents_adm3 import MonthlyResidentsAdm3


class SnapshotTrendsResidentsAdm3(MonthlyResidentsAdm3):

    def _compute_trend_and_residuals(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        threshold_multiplier = 3

        ABNORMALITY_CUTOFF = 5

        for loc, group in df.groupby(self.aggregates.config.shapefile_spatial_id):
            # Ensure data is sorted by date
            group = group.sort_values(self.aggregates.config.date_column)

            has_abnormal_change = (
                True
                if ((np.abs(group.abnormality) > ABNORMALITY_CUTOFF).sum()) > 0
                else False
            )

            largest_abnormal_fluct = (
                group.residents.diff().loc[np.abs(group.abnormality) > 0].max()
            )

            # Filter data for the last 12 months
            max_date = group[self.aggregates.config.date_column].max()
            min_date = max_date - pd.DateOffset(months=12)
            group = group[group[self.aggregates.config.date_column] >= min_date]

            # Extract date and values
            dates = pd.to_datetime(group[self.aggregates.config.date_column])
            X = (dates - dates.min()).dt.days.values.reshape(
                -1, 1
            )  # Convert dates to ordinal
            y = group["residents"].values
            print("admin3 residents dataset")
            print(y)

            # Apply RANSAC
            # regr = RANSACRegressor()
            regr = LinearRegression()
            regr.fit(X, y)
            y_pred = regr.predict(X)

            # Calculate residuals
            residuals = y - y_pred
            std_residuals = np.std(residuals)

            # Detect outliers
            outliers = np.abs(residuals) > (threshold_multiplier * std_residuals)
            num_outliers = np.sum(outliers)

            # Calculate percentage change based on the trend
            trend_start = y_pred[0]
            trend_end = y_pred[-1]
            percent_change_trend = ((trend_end - trend_start) / trend_start) * 100

            absolute_monthly_change_trend = (trend_end - trend_start) / 12

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
            unusual = (
                (avg_first_three_months < avg_last_three_months)
                and (percent_change_trend < 0)
            ) or (
                (avg_first_three_months > avg_last_three_months)
                and (percent_change_trend > 0)
            )

            has_changed = group["residents"].nunique() > 1

            # Store the results
            results.append(
                {
                    self.aggregates.config.shapefile_spatial_id: loc,
                    self.aggregates.config.shapefile_spatial_name: group[
                        self.aggregates.config.shapefile_spatial_name
                    ].iloc[0],
                    self.aggregates.config.shapefile_regional_spatial_name: group[
                        self.aggregates.config.shapefile_regional_spatial_name
                    ].iloc[0],
                    "Trend_Slope": regr.coef_[0],
                    "Trend_Intercept": regr.intercept_,
                    # "Trend_Slope": regr.estimator_.coef_[0],
                    # "Trend_Intercept": regr.estimator_.intercept_,
                    "Std_Residuals": std_residuals,
                    "Percent_Change_Trend": percent_change_trend,
                    "Percent_Change_Actual": percent_change_actual,
                    "Absolute_Change_Trend": y_pred[-1] - y_pred[0],
                    "Absolute_Change_Actual": y[-1] - y[0],
                    "Unusual": unusual,
                    "Num_Outliers": num_outliers,
                    "Absolute_Monthly_Change_Trend": absolute_monthly_change_trend,
                    "has_data": has_changed,
                    "has_abnormal_change": has_abnormal_change,
                    "largest_abnormal_fluct": largest_abnormal_fluct,
                    "most_recent_pop": actual_end,
                }
            )

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate the threshold for fluctuating regions based on the XXth percentile
        std_residuals_threshold = results_df["Std_Residuals"].quantile(0.95)

        # Determine if a region is fluctuating
        results_df["Fluctuating"] = (
            results_df["Std_Residuals"] > std_residuals_threshold
        )

        return results_df

    def calculate(self):
        monthly_residents_adm3 = super().calculate()
        trends = self._compute_trend_and_residuals(monthly_residents_adm3)
        return trends
