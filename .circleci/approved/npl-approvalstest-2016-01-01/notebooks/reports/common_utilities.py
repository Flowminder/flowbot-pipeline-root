# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
@author: Romain Goldenberg, Robert Eyre
@email: romain.goldenberg@flowminder.org, robert.eyre@flowminder.org
"""

"""
This script contains all the utilities necessary crisis reponse notebooks.  
→ This script has been modified by Robert Eyre from the Ghana standard mobility reports code created by Romain Goldenberg.
"""

## A. Common libraries and settings

import glob
import logging
import os
import time
import warnings
from calendar import monthrange
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import contextily as cx
import flowmindercolors as fmcolors
import folium
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xyzservices.providers as xyz
from IPython.display import Markdown, display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

fmcolors.register_custom_colormaps()

# Plotting style
plt.style.use("seaborn-v0_8-ticks")
mpl.rcParams["legend.frameon"] = True
mpl.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.axisbelow"] = False

# Flowminder overarching colors
fm_colors = ["#095798", "#2977b8", "#cba45a", "#054274"]

# Flowminder colormap (sequential palette)
fm_seq_colors = [
    "#B7EDE6",
    "#9CE5C6",
    "#98D399",
    "#A0BA69",
    "#A99B3B",
    "#AB7A22",
    "#A65432",
    "#992649",
    "#8A005E",
]

# Flowminder colormap (diverging palette)
fm_div_colors = [
    "#034174",
    "#36679C",
    "#6C8FC1",
    "#A5B9DC",
    "#DCDDE9",
    "#CDB1CA",
    "#BD79A4",
    "#9A497B",
    "#701F53",
]

# Flowminder colormap (flows palette)
fm_flow_colors = ["#701F53", "#27B288"]

# Create colormap objects
cmap_fm_seq = LinearSegmentedColormap.from_list("fm_seq", fm_seq_colors, N=100)
cmap_fm_div = LinearSegmentedColormap.from_list("fm_div", fm_div_colors, N=100)
cmap_fm_flows = LinearSegmentedColormap.from_list(
    "fm_flow", colors=fm_flow_colors, N=100
)

# Flowminder additional colors
no_data_c = "#A3A3A3"
fm_other_colors = ["#B6C6E4"]
dark_grey = "#2D2D2D"
light_grey = "#C6C6C6"
lighter_grey = "#E0E0E0"
stable = "#bad1c9"

## B. Useful functions
### 1. Stats or analytical functions


def calculate_baseline(
    baseline_start: str,
    baseline_end: str,
    dataframe: pd.DataFrame,
    value_for_baseline: str,
) -> pd.DataFrame:
    """
    Calculates the percentage change from baseline for a given value over a specified period.

    Parameters:
    -----------
    baseline_start : str
        Start date of the baseline period in YYYY-MM-DD format.
    baseline_end : str
        End date of the baseline period in YYYY-MM-DD format.
    dataframe : pandas.DataFrame
        The data to be used for calculating the baseline.
    value_for_baseline: str
        The name of the column in the dataframe to be used for calculating
        the baseline.

    Returns:
    --------
    merged : pandas.DataFrame
        Table containing the percentage change from baseline for each row
        in the input data.
    """

    # Convert the input strings to datetime
    baseline_start = pd.to_datetime(baseline_start)
    baseline_end = pd.to_datetime(baseline_end)

    # Check if any columns in the dataframe contain "pcod" in their name
    pcod_group = [col for col in dataframe.columns if "pcod" in col]

    # Filter the dataframe for the baseline period
    baseline_df = dataframe[
        (dataframe.date >= baseline_start) & (dataframe.date < baseline_end)
    ]

    if pcod_group:
        # Group the dataframe by the columns containing "pcod"
        grouped = baseline_df.groupby(pcod_group)
        # Calculate the median of the value_for_baseline for each group
        value_baseline = grouped[value_for_baseline].median()
        # Join the values with the corresponding baseline
        merged = dataframe.merge(
            value_baseline,
            left_on=pcod_group,
            right_index=True,
            suffixes=("", "_baseline"),
        )
    else:
        # Calculate the median of the value_for_baseline for the entire dataframe
        value_baseline = baseline_df[value_for_baseline].median()
        # Create a pandas Series with the length of dataframe, containing value baseline
        value_baseline = pd.Series([value_baseline] * len(dataframe))
        # Join the values with the baseline
        merged = pd.concat(
            [dataframe, value_baseline.rename(f"{value_for_baseline}_baseline")], axis=1
        )

    # Calculate the percentage change from baseline
    merged[f"{value_for_baseline}_change_from_baseline"] = (
        merged[value_for_baseline] / merged[f"{value_for_baseline}_baseline"] - 1
    ) * 100

    # Select only the necessary columns
    merged = merged[["date", *pcod_group, f"{value_for_baseline}_change_from_baseline"]]

    # Sort the dataframe by date and pcod_group columns
    merged.sort_values(
        by=["date", *pcod_group],
        ascending=True,
        inplace=True,
        ignore_index=True,
    )
    merged.date = pd.to_datetime(merged.date)

    return merged


def calculate_distance_centroids(
    df_admin: pd.DataFrame, geo_admin: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Calculates the distances between administrative units in a given region, based on their centroids.

    Parameters:
    -----------
    df_admin : pandas.DataFrame
        An OD matrix that contains at least two columns: 'pcod_from' and 'pcod_to',
        the origin and destination of a population flow.
    geo_admin : geopandas.GeoDataFrame
        A GeoDataFrame that contains polygons of the administrative units,
        and also includes a 'centroid' column which stores the centroid of each polygon.

    Returns:
    --------
    unique_pairwise_distances : pandas.DataFrame
        A dataframe that contains the pairwise distances (in kilometers) between
        each administrative unit in df_admin.
    """

    # With .to_crs we reproject the data (to calculate distances)
    geo_admin_proj = geo_admin.to_crs("epsg:25000")
    geo_admin_proj["centroid"] = geo_admin_proj.centroid

    # Create a table of all unique (directed) pairwise combinations.
    unique_pairwise_distances = (
        df_admin.groupby(["pcod_from", "pcod_to"]).count().reset_index()
    )[["pcod_from", "pcod_to"]]

    # Join the centroid informations with both pcod_from and pcod_to
    unique_pairwise_distances = unique_pairwise_distances.merge(
        geo_admin_proj[["pcod", "centroid"]].rename(
            columns={"centroid": "centroid_from"}
        ),
        left_on="pcod_from",
        right_on="pcod",
    ).drop(columns=["pcod"])

    unique_pairwise_distances = unique_pairwise_distances.merge(
        geo_admin_proj[["pcod", "centroid"]].rename(
            columns={"centroid": "centroid_to"}
        ),
        left_on="pcod_to",
        right_on="pcod",
    ).drop(columns=["pcod"])

    # Then we can calculate distances between each pair of centroids
    unique_pairwise_distances["distance_km"] = (
        unique_pairwise_distances.apply(
            lambda x: (x["centroid_to"].distance(x["centroid_from"])), axis=1
        )
        / 1000
    )

    # Merge back on the original data
    unique_pairwise_distances = pd.merge(
        df_admin,
        unique_pairwise_distances[["pcod_from", "pcod_to", "distance_km"]],
        on=["pcod_from", "pcod_to"],
    )
    return unique_pairwise_distances


def calculate_flow_stability(
    dataframe: pd.DataFrame, value_for_flow: str
) -> pd.DataFrame:
    """
    Compute various metrics that characterize the stability of population flow values
    within a time series dataframe with one or more pcod identifiers.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe containing pcod identifiers and flow values (OD matrix).
    value_for_flow : str
        The name of the column in the input dataframe that contains the values.

    Returns
    -------
    result_df : pandas.DataFrame
        A dataframe containing the following columns:
        - pcod_group: pcod identifier(s)
        - flow_value_median: median value of flows per pcod identifier across the time period
        - flow_value_sum: sum of flows per pcod identifier across the time period
        - z_score_extreme: extreme z-score value for the pcod identifier(s)
        - z_score_bool: boolean value indicating if z-score is between 3 and 9 for the pcod identifier(s)
        - skew: skewness of the flow values per pcod identifier across the time period
        - perc_time_period: percentage of time periods with measurements available for the pcod identifier(s)

    Raises
    ------
    Exception
        If the input dataframe doesn't include unique pcod identifiers.
    """

    df = dataframe.copy()
    pcod_group = [col for col in df if "pcod" in col]
    max_periods = df["date"].nunique()

    if not pcod_group:
        raise Exception(f"The dataframe used should include unique pcod identifiers.")

    # Group by pcod, apply z_score_classification to each group, and create a new dataframe with the results.
    result_df = df.groupby(pcod_group).apply(
        z_score_classification, value_for_flow, max_periods
    )

    # Calculate boolean values based on the z_score_extreme column.
    result_df["z_score_bool"] = result_df.apply(
        lambda x: (x["z_score_extreme"] >= 3) & (x["z_score_extreme"] <= 9), axis=1
    )

    # Calculate other metrics for each pcod group.
    result_df["flow_value_sum"] = df.groupby(pcod_group)[value_for_flow].sum()
    result_df["flow_value_median"] = df.groupby(pcod_group)[value_for_flow].median()
    result_df["skew"] = df.groupby(pcod_group)[value_for_flow].skew()
    result_df["perc_time_period"] = (
        (df.groupby(pcod_group)[value_for_flow].count() / max_periods) * 100
    ).astype(int)

    # Select and order the columns in the output dataframe.
    result_df = result_df[
        [
            "flow_value_median",
            "flow_value_sum",
            "z_score_extreme",
            "z_score_bool",
            "skew",
            "perc_time_period",
        ]
    ].reset_index()

    return result_df


def z_score_classification(
    df: pd.DataFrame, value_for_flow: str, max_periods: int
) -> pd.Series:
    """
    Apply z-score classification to the input dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with flow values and a pcod column.
    value_for_flow : str
        The name of the column that contains flow values.
    max_periods : int
        The maximum number of time periods in the input dataframe.

    Returns
    -------
    pandas.Series
        A series with the highest robust z-score value in the input dataframe.
    """

    result = np.nan
    if (len(df.index) > 3) & (len(df.index) > max_periods / 2):
        # Calculate z-scores for the input dataframe.
        z_score_calc = z_score(df, value_for_flow)

        # Find the highest robust z-score value.
        if ~z_score_calc["z_score_extreme_value"].isnull().values.all():
            idx = z_score_calc["robust_z_score"].abs().idxmax()
            highest_z = z_score_calc.loc[idx]["robust_z_score"]
            result = highest_z

    return pd.Series({"z_score_extreme": result})


def calculate_group_stats(
    trends_df: pd.DataFrame, groups: list[str], population_count: str = None
):
    """
    Calculates certain statistics for each group in a Pandas DataFrame.

    Parameters:
    -----------
    trends_df : pandas.DataFrame
        A DataFrame containing data on trends.
    groups : list of str
        A list of column names representing the groups to calculate statistics for.
    population_count : str, optional
        If provided, calculate population counts group statistics on the column provided. Default is None.

    Returns:
    --------
    stats : pandas.DataFrame
        A DataFrame containing the calculated statistics for each group.
    """

    # Group the data by the columns specified in `groups`.
    grouped_data = trends_df.groupby(groups)

    # Apply a lambda function to calculate the number of admin areas
    # and the percentage of total admin areas that are in each group.
    stats = grouped_data["pcod"].apply(
        lambda x: pd.Series(
            {
                "number of admins": x.count(),
                "admins percentage of total": (
                    (x.count() / trends_df["pcod"].nunique()) * 100
                ).round(1),
            }
        )
    )

    # Unstack the resulting Series.
    stats = stats.unstack()

    # Calculate total population counts if required
    if population_count is not None:
        stats["total population"] = grouped_data[population_count].apply(
            lambda x: x.sum()
        )
        stats["population percentage of total"] = grouped_data[population_count].apply(
            lambda x: ((x.sum() / stats["total population"].sum()) * 100).round(1)
        )

        # Convert the "total population" column to an unsigned 32-bit integer.
        stats["total population"] = stats["total population"].astype(np.uint32)

    # Reset the index of the resulting DataFrame.
    stats = stats.reset_index()

    # Convert the "number of admins" column to an unsigned 32-bit integer.
    stats["number of admins"] = stats["number of admins"].astype(np.uint32)

    # Return the resulting DataFrame.
    return stats


def calculate_percentiles(
    df_admin: pd.DataFrame, geo_admin: gpd.GeoDataFrame, value_for_perc: str
) -> pd.DataFrame:
    """
    Calculates percentiles of minimum, maximum and weighted distance for a given set of
    OD (Origin Destination) matrix and administrative units.

    Parameters:
    -----------
    df_admin : pandas.DataFrame
        DataFrame containing an OD matrix.
    geo_admin : geopandas.GeoDataFrame
        DataFrame containing geographic coordinates for the administrative areas.
    value_for_perc : str
        String indicating the column name of the values to calculate percentiles for.

    Returns:
    --------
    perc_df : pandas.DataFrame
        DataFrame containing the calculated percentiles.
    """

    # Define the percentiles to calculate.
    percentiles = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99]

    # Calculate the distances between connected (by the OD matrix) administrative areas.
    distances_df = calculate_distance_centroids(df_admin, geo_admin)

    # Group the distances by date and calculate the minimum distance for each date.
    perc_df = distances_df.groupby(["date"])["distance_km"].min()
    perc_df = perc_df.rename("minimum").to_frame()

    # Calculate the weighted percentiles for each date and add them to the DataFrame.
    for perc in percentiles:
        perc_df[f"{int(perc*100)}%"] = distances_df.groupby(["date"]).apply(
            weighted_perc, "distance_km", value_for_perc, perc
        )

    # Calculate the maximum distance for each date and add it to the DataFrame.
    perc_df["maximum"] = distances_df.groupby(["date"])["distance_km"].max()

    # Reset the index of the DataFrame and return it.
    perc_df = perc_df.reset_index()
    return perc_df


def weighted_perc(
    df: pd.DataFrame, values: str, weights: str, percentile: float
) -> float:
    """
    Calculates the weighted percentile of a given set of values.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the values and weights.
    values : str
        String indicating the column name of the values to calculate the percentile for.
    weights : str
        String indicating the column name of the weights to use.
    percentile : float
        Float indicating the percentile to calculate.

    Returns:
    --------
    median: float
        The weighted percentile.
    """

    # Sort the DataFrame by the values.
    df.sort_values(values, inplace=True)

    # Calculate the cumulative sum of the weights.
    cumsum = df[weights].cumsum()

    # Calculate the cutoff weight for the given percentile.
    cutoff = df[weights].sum() * percentile

    # Find the value that corresponds to the cutoff weight.
    median = df[values][cumsum >= cutoff].iloc[0]

    # Return the value.
    return median


def calculate_proportions(
    dataframe: pd.DataFrame, value_for_proportions: str
) -> pd.DataFrame:
    """
    Calculate the proportion of a certain value relative to the total travels for each date in the dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing the data to be processed.
    value_for_proportions : str
        The name of the column in the dataframe that contains the value for which the proportions will be calculated.

    Returns
    -------
    merged : pandas.DataFrame
        A new dataframe containing the date, pcod_group, and the calculated proportion of `value_for_proportions`
        relative to the total daily travels for each date.
    """

    # Get a list of columns that contain 'pcod'
    pcod_group = [col for col in dataframe if "pcod" in col]

    # Group the dataframe by date and calculate the total daily travels
    total_daily_travels = dataframe.groupby(["date"])[value_for_proportions].sum()

    # Join the values with the corresponding date
    merged = dataframe.merge(
        total_daily_travels,
        left_on="date",
        right_index=True,
        suffixes=("", "_sum"),
    )

    # Calculate the ratio of value_for_proportions to the total daily travels for each date
    merged[f"{value_for_proportions}_ratio_to_all_travels"] = (
        merged[value_for_proportions] / merged[f"{value_for_proportions}_sum"]
    ) * 100

    # Select the relevant columns and sort the resulting dataframe
    merged = merged[
        ["date", *pcod_group, f"{value_for_proportions}_ratio_to_all_travels"]
    ]
    merged.sort_values(
        by=["date", *pcod_group],
        ascending=True,
        inplace=True,
        ignore_index=True,
    )

    # Convert the date column to datetime format and return the resulting dataframe
    merged.date = pd.to_datetime(merged.date)
    return merged


def calculate_trends_flows(
    df_flows: pd.DataFrame, value_for_trends: str, time_resolution: str = "day"
) -> pd.DataFrame:
    """
    Calculate linear trends and variations of population flow values for each unique pcod group in a dataframe.


    Parameters
    ----------
    df_flows : pandas.DataFrame
        The DataFrame to calculate the flow trends for.
    value_for_trends : str
        The name of the column in the DataFrame to calculate the flow trends for.
    time_resolution : {'day', 'month'}, optional
        The time resolution of the input data, by default "day".

    Returns
    -------
    result_df: pandas.DataFrame
        A DataFrame containing the calculated flow trends for each PCOD group.

    Raises
    ------
    Exception
        If the input DataFrame does not contain unique pcod identifiers.
    """

    df = df_flows.copy()
    pcod_group = [col for col in df if "pcod" in col]

    if pcod_group:
        result_df = df.groupby(pcod_group).apply(linear_model, value_for_trends)

        result_df["tendency"] = result_df.apply(lambda x: tendency(x["slope"]), axis=1)
        result_df["weekday_or_weekend"] = df.groupby(pcod_group).apply(
            weekday_or_weekend, value_for_trends
        )
        result_df = result_df.reset_index()

        return result_df

    else:
        raise Exception(f"The dataframe used should include unique pcod identifiers.")


def weekday_or_weekend(df: pd.DataFrame, value_for_trends: str) -> str:
    """
    Calculates the ratio between the median values of the input data on weekdays and weekends
    for each group in the DataFrame. Returns the string 'weekday < weekend' or 'weekday > weekend',
    depending on the ratio.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing data for a single group.
    value_for_trends : str
        The name of the column in the DataFrame to calculate the weekday/weekend ratio for.

    Returns
    -------
    str
        The string 'weekday < weekend' or 'weekday > weekend', depending on the ratio of median values.
    """

    df["is_weekend"] = np.where(
        df["date"].apply(lambda x: x.weekday() >= 5),
        "weekend",
        "weekday",
    )

    ratio = (
        df[df["is_weekend"] == "weekday"][value_for_trends].median()
        / df[df["is_weekend"] == "weekend"][value_for_trends].median()
    )

    if ratio < 0:
        return "weekday < weekend"
    elif ratio >= 0:
        return "weekday > weekend"


def calculate_trends_population(
    dataframe: pd.DataFrame,
    value_for_trends: str,
    time_resolution: str = "day",
    pcod_column: str = "pcod",
) -> pd.DataFrame:
    """
    Calculate linear trends and variations of population values for each unique pcod group in a dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataframe containing population values and pcod column(s).
    value_for_trends : str
        The name of the column containing population values.
    time_resolution : {'day', 'month'}, optional
        The time resolution of the input data, by default "day".
    pcod_column : str, optional
        The time resolution of the input data, by default "day".

    Returns
    -------
    result_df : pandas.DataFrame
        A dataframe with tendency and variation information for each unique pcod group.

    Raises
    ------
    Exception
        If the input dataframe does not include unique pcod identifiers.
    """

    # Create a copy of the input dataframe.
    df = dataframe.copy()

    # Find the columns containing pcod information.
    pcod_group = [col for col in df if "pcod" in col]

    # If the input dataframe includes pcod identifiers, calculate linear trends.
    if pcod_group:
        result_df = (
            df.groupby(pcod_group)
            .apply(linear_model, value_for_trends, time_resolution)
            .reset_index()
        )

        # Calculate tendency and variation values for each pcod group.
        slope_threshold = max(
            np.abs(result_df.slope.quantile(0.15)), result_df.slope.quantile(0.85)
        )
        std_threshold = result_df.std_detrended.quantile(0.85)
        result_df["tendency"] = result_df.apply(
            lambda x: tendency(x["slope"], slope_threshold), axis=1
        )
        result_df["variation"] = result_df.apply(
            lambda x: variation(x["std_detrended"], std_threshold), axis=1
        )

        return result_df

    else:
        raise Exception("The dataframe used should include unique pcod identifiers.")


def variation(std: float, threshold: float) -> str:
    """
    Determine the variation of a flow value standard deviation.

    Parameters
    ----------
    std : float
        The standard deviation of a flow value.

    Returns
    -------
    str
        A string indicating the variation of the standard deviation.
    """
    # print(f'std variation {threshold}')
    if std < threshold:
        return "constant"
    elif std >= threshold:
        return "fluctuating"


def keep_sufficient_flows(
    df: pd.DataFrame,
    baseline_start: str,
    baseline_end: str,
    min_travels_cutoff: int = 15,
    min_baseline_cutoff: float = 0.5,
    min_timeperiod_cutoff: float = 0.5,
) -> pd.DataFrame:
    """
    Remove population flows that have insufficient data coverage and are below a minimum cutoff value.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with flow values and a pcod column.
    baseline_start : str
        A string indicating the start date of the baseline period (in YYYY-MM-DD format).
    baseline_end : str
        A string indicating the end date of the baseline period (in YYYY-MM-DD format).
    min_travels_cutoff : int, optional
        The minimum flow value (i.e. number of travellers),
        above which data will be retained in the filtered DataFrame (default is 15).
    min_baseline_cutoff: float, optional
        The minimum number of data points that should be available during the baseline period,
        above which data will be retained in the filtered DataFrame (default is 0.5, i.e. 50%)
    min_timeperiod_cutoff: float, optional
        The minimum number of data points that should be available during the full time period (including the baseline),
        above which data will be retained in the filtered DataFrame (default is 0.5, i.e. 50%)

    Returns
    -------
    result : pandas.DataFrame
        A filtered DataFrame that retains only flows with sufficient data coverage and values above the minimum cutoff.
    """

    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()

    # Convert baseline start and end dates to datetime objects
    base_start = pd.to_datetime(baseline_start)
    base_end = pd.to_datetime(baseline_end)

    # Find the columns in the DataFrame that contain "pcod" (unique identifier for each location)
    pcod_group = [col for col in result if "pcod" in col]

    # Calculate the number of days in the baseline period and the total number of days in the DataFrame
    nb_baseline_days = (base_end - base_start).days
    nb_days = (result["date"].max() - result["date"].min()).days

    # Remove flows with number of travellers below the minimum cutoff
    result = result[result["value"] >= min_travels_cutoff]

    # Filter for flows that have data for more than the minimum cutoff of the baseline period
    baseline_df = result[(result["date"] >= base_start) & (result["date"] <= base_end)]
    baseline_df = baseline_df[
        baseline_df.groupby(pcod_group)["date"].transform("count")
        >= nb_baseline_days * min_baseline_cutoff
    ]
    baseline_df = baseline_df[pcod_group].drop_duplicates()

    # Keep only flows that match the pcod values in the filtered baseline DataFrame
    baseline_keys = list(baseline_df.columns.values)
    baseline_df = baseline_df.set_index(baseline_keys).index
    to_keep = result.set_index(baseline_keys).index
    result = result[to_keep.isin(baseline_df)]

    # Filter for flows that have data for more than the minimum cutoff of the total time period
    result = result[
        result.groupby(pcod_group)["date"].transform("count")
        >= nb_days * min_timeperiod_cutoff
    ]

    return result


def linear_model(
    df: pd.DataFrame, value_for_trends: str, time_resolution: str
) -> pd.Series:
    """
    Fit a linear regression model to the input dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with a date column and a column with values for trends.
    value_for_trends : str
        The name of the column that contains values for trends.
    time_resolution : str
        The time resolution of the input data. Either 'day' or 'month'.

    Returns
    -------
    pandas.Series
        A series with the slope of the fitted linear regression model,
        the R-squared value, and the standard deviation of the detrended data.
    """

    lr = LinearRegression()

    if time_resolution == "day":
        df["time_from_start"] = (df.date - df.date.min()).dt.days
    elif time_resolution == "month":
        df["time_from_start"] = df.date.dt.to_period("M") - df.date.min().to_period("M")
        df["time_from_start"] = df["time_from_start"].apply(lambda x: x.n)

    # Fit the linear model
    lr.fit(
        df["time_from_start"].values.reshape(-1, 1),
        df[value_for_trends].values.reshape(-1, 1),
    )

    # Make predictions using the fitted model
    trends_pred = lr.predict(df["time_from_start"].values.reshape(-1, 1))

    # Calculate the coefficient of determination R2
    r2 = r2_score(df[value_for_trends].values.reshape(-1, 1), trends_pred)

    # Calculate the detrended standard deviation
    detrended = [
        x1 - x2
        for (x1, x2) in zip(df[value_for_trends].values.reshape(-1, 1), trends_pred)
    ]
    std = np.array(detrended).std()

    return pd.Series(
        {
            "slope": lr.coef_[0][0],
            "intercept": lr.intercept_[0],
            "r2": r2,
            "std_detrended": std,
        }
    )


def tendency(slope: float, threshold: float) -> str:
    """
    Classify the trend as increasing, stable or decreasing based on the input slope.

    Parameters
    ----------
    slope : float
        The individual slope (i.e. for one admin area) from the linear regression model.
    mean_slope : float
        The mean of all slopes (i.e. of all admin areas) from the linear regression model.

    Returns
    -------
    str
        A string representing the trend classification.
    """
    # print(f'slope threshold {threshold}')
    if slope < -threshold:
        return "decreasing"
    elif (slope >= -threshold) & (slope <= threshold):
        return "stable"
    elif slope > threshold:
        return "increasing"


def stats_available_data(
    start_date: str, end_date: str, dates_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the percentage of available data across the time period.

    Parameters
    ----------
    start_date : str
        The start date of the period in the format 'YYYY-MM-DD'.
    end_date : str
        The end date of the period in the format 'YYYY-MM-DD'.
    dates_table : pandas.DataFrame
        A dataframe with a 'date' column and an 'available' column indicating whether
        data is available for each date.

    Returns
    -------
    float
        The percentage of missing data across the time period.
    """

    # Split data between available days and non-available days
    available_ratio = dates_table.groupby(["available_date"]).size()

    # Calculate the number of days
    nb_days = available_ratio.sum()

    # Calculate the number of available days
    nb_days_true = available_ratio[1]

    # Calculate the ratio of available data across the time period
    ratio = (nb_days_true / nb_days) * 100

    # Print info on data available
    print(
        f"Over the period {start_date} to {end_date} ({nb_days} days), {round(ratio,1)}% of the data ({nb_days_true} days) is available."
    )
    missing_data = round(100 - ratio, 1)

    return missing_data


def z_score(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """
    Calculate the robust and extreme values Z-score of a given value in a DataFrame.

    Parameters:
        df : pandas.DataFrame
            A pandas DataFrame containing the data.
        value : str
            A string representing the name of the column containing the data.

    Returns:
        df : pandas.DataFrame
            The input DataFrame with two new columns:
            - robust_z_score: the robust z-score of the data.
            - z_score_extreme_value: the data point if it is an extreme value (i.e., outside of
              the range of -3 and 3 on the robust_z_score).
    """

    # Compute the Median Absolute Deviation (MAD) and the Mean Absolute Deviation (MeanAD)
    data_values = df[value].values
    MAD = np.nanmedian(np.absolute(data_values - np.nanmedian(data_values)))
    MeanAD = np.nanmean(np.absolute(data_values - np.nanmean(data_values)))

    # Compute the robust z-score using MAD or MeanAD depending on the data distribution
    if MAD != 0:
        df["robust_z_score"] = (df[value] - np.nanmedian(data_values)) / (1.486 * MAD)
    else:
        df["robust_z_score"] = (df[value] - np.nanmedian(data_values)) / (
            1.253314 * MeanAD
        )

    # Flag extreme values in a new column
    df["z_score_extreme_value"] = df.apply(
        lambda x: (
            x[value]
            if any([x["robust_z_score"] >= 3, x["robust_z_score"] <= -3])
            else np.nan
        ),
        axis=1,
    )

    return df


### 2. Utilities functions


def create_admins_sjoin_table(
    tuples_gpd_admins: list[tuple[gpd.GeoDataFrame, str]]
) -> pd.DataFrame:
    """
    Create a table combining administrative area of overlapping geometries
    (i.e. find connections between country, regions, districts, etc.).

    Parameters:
    -----------
    tuples_gpd_admins : list of tuples
        List of tuples, where each tuple consists of a GeoDataFrame containing administrative boundaries
        and a string representing the name of the corresponding administrative level (e.g. 'district', 'region').

    Returns:
    --------
    overlay : pandas.DataFrame
        A table containing administrative area of overlapping geometries.
    """
    # We reproject the geodata (to calculate area)
    admins = [
        (x[0][[x[1], "geometry"]].to_crs("epsg:25000"), x[1]) for x in tuples_gpd_admins
    ]

    # Perform the overlay of administrative boundaries, keeping the geometries in a MultiPolygon format.
    overlay = admins[0][0]
    for admin in admins[1:]:
        overlay = gpd.overlay(
            overlay, admin[0], how="intersection", keep_geom_type=False
        )

        # Calculate the area of the overlapping polygons.
        overlay["area"] = overlay.geometry.area

        # Sort by area so the largest area is last, to keep only the polygons intersecting "the most".
        overlay.sort_values(by=["area"], ascending=True, inplace=True)
        overlay.drop_duplicates(subset=[admin[1]], keep="last", inplace=True)
        overlay = overlay.drop(["area"], axis=1)

    # Convert the result to a Pandas DataFrame and sort by administrative level.
    overlay = pd.DataFrame(overlay.drop(columns="geometry"))
    overlay.sort_values(
        by=[x[1] for x in admins], ascending=True, inplace=True, ignore_index=True
    )

    return overlay


def create_dates_table(
    start_date: str, end_date: str, time_available_from_data: dict
) -> pd.DataFrame:
    """
    Create a table of dates for a given period and indicate whether data is available on each date.

    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    time_available_from_data : dict
        Dictionary with information about available data.

    Returns:
    --------
    dates_table : pandas.DataFrame
        Table with date and availability information.
    """

    # All dates for the period considered
    all_dates = pd.date_range(start_date, end_date, inclusive="left")

    # Create a table with dates and a column indicating whether data is available on each date
    dates_table = pd.DataFrame(data={"date": all_dates})
    dates_table["available_date"] = (
        dates_table["date"].isin(time_available_from_data["calls"]).astype("int32")
    )

    return dates_table


def display_metadata(xr_dataset: xr.Dataset) -> None:
    """
    Print the metadata of an xarray dataset.

    Parameters:
    -----------
    xr_dataset : xarray.Dataset
        The dataset to display the metadata for.

    Returns:
    --------
    None
    """
    # Iterate over the attributes of the dataset and print them
    for key, value in xr_dataset.attrs.items():
        if key == "name":
            print(f"\033[1m{key}: {value}\033[0m")
        else:
            print(f"{key}: {value}")


def load_netcdf_data(
    folder: str, list_dataset_names: list[str], use_case: str
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """
    Load multiple NetCDF files and convert each into a Pandas DataFrame.

    Parameters:
    -----------
    folder : str
        path to folder containing the NetCDF files.
    list_dataset_names : List[str]
        names of the datasets to load.
    use_case : str
        prefix of the file names for the specific use case.

    Returns:
    --------
    A tuple of two dictionaries:
        - A dictionary with the dataset name as the key and the corresponding Pandas DataFrame as the value.
        - A dictionary with the dataset name as the key and the corresponding raw metadata as the value.
    """

    dataset: dict[str, pd.DataFrame] = {}
    raw_metadata: dict[str, dict] = {}

    display(Markdown("---"))

    # Open each NetCDF file.
    for name in list_dataset_names:
        dataset[name] = xr.open_dataset(
            glob.glob(f"{folder}/{use_case}*{name}_*.cdf")[0]
        )
        raw_metadata[name] = dataset[name].attrs

        # Display metadata.
        display_metadata(dataset[name])
        display(Markdown("---"))

        # Convert the NetCDF file to Pandas DataFrame.
        dataset[name] = dataset[name].to_dataframe()
        # Convert the 'date' column to datetime format.
        dataset[name].date = pd.to_datetime(dataset[name].date)

    return dataset, raw_metadata


def run_query_in_smaller_blocks(
    query, connection_object, valid_dates, aggregation_unit, **kwargs
):
    mapping_table = kwargs.get("mapping_table", None)
    block_size = kwargs.get("block_size", 7)
    count_interval = kwargs.get("count_interval", "day")

    start_date = valid_dates.min()
    end_date = valid_dates.max()
    name = f"{aggregation_unit}_{query.__name__}"

    # Create the temp folder only if it does not already exist
    temporaryfolder = (
        Path().resolve().parent
        / "Outputs"
        / "Data"
        / "Sensitive"
        / "Aggregates"
        / "temp"
    )
    queryfolder = (
        temporaryfolder
        / f"{name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
    )
    for folder in [temporaryfolder, queryfolder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Check if a checkpoint file containing the latest ran date exists
    checkpoint_file = (
        queryfolder
        / f"chkp_{name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.txt"
    )
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_date = pd.Timestamp(f.read())

    # Print remaining queries to run
    print(
        f"\r{name} from {valid_dates.min().strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} in {block_size} days blocks",
        end="",
    )
    if start_date <= end_date:
        print(
            f"\nRemaining: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

    # Create all queries for the full time period and store them in a dictionary
    all_queries = {}
    for day in valid_dates:
        query_kwargs = {
            "connection": connection_object,
            "start_date": str(day),
            "end_date": str(day + pd.Timedelta("1 day")),
            "aggregation_unit": aggregation_unit,
            "mapping_table": mapping_table,
        }

        if "count_interval" in kwargs:
            query_kwargs["count_interval"] = kwargs.get("count_interval")

        query_instance = query(**query_kwargs)
        all_queries[day] = query_instance

    # Create and run queries in smaller blocks, in the provided range of valid dates
    while start_date <= end_date:
        block_end_date = min(start_date + pd.Timedelta(block_size - 1, "D"), end_date)
        block_dates = valid_dates[
            (valid_dates >= start_date) & (valid_dates <= block_end_date)
        ]
        queries = {day: all_queries[day] for day in block_dates}

        # Run all queries in the block
        for query_instance in queries.values():
            try:
                query_instance.run()
            except:
                raise Exception(
                    "Could not start running a query. Exiting the function."
                )

        # Check the status of all queries in the block
        timeout = block_size * 60  # Time limit in seconds
        start_time = time.time()
        while True:
            query_statuses = [
                query_instance.status for query_instance in queries.values()
            ]
            status_counter = Counter(query_statuses)
            completed_queries = status_counter.get("completed", 0)
            in_progress_queries = status_counter.get(
                "executing", 0
            ) + status_counter.get("queued", 0)

            # Print the status message after clearing the line
            print(f"\r{' ' * 90}", end="", flush=True)

            # Print current status
            print(
                f"\r{start_date.strftime('%Y-%m-%d')} to {block_end_date.strftime('%Y-%m-%d')} - {status_counter}",
                end=" ",
                flush=True,
            )

            # If any query has a status not in the expected set, raise an exception to exit the function
            if not set(status_counter.keys()).issubset(
                {"completed", "executing", "queued"}
            ):
                raise Exception("A query is not running. Exiting the function.")

            # If all queries are completed, exit the loop
            if completed_queries == len(queries):
                break

            # If not wait for block_size*2 seconds before checking the status again
            time.sleep(block_size * 2.5)

            # Check for the timeout to prevent infinite loop
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise Exception(
                    "Timeout: Some queries are not completed after the time limit. Exiting the function."
                )

        try:
            result = pd.concat(
                [
                    query_instance.get_result().assign(date=day)
                    for day, query_instance in queries.items()
                ],
                ignore_index=True,
            )

            # Print the status message
            print(
                f"\r{start_date.strftime('%Y-%m-%d')} to {block_end_date.strftime('%Y-%m-%d')} - {status_counter} || \033[35msaving...\033[0m",
                end="",
                flush=True,
            )

            result.to_csv(
                queryfolder
                / f"{name}_{start_date.strftime('%Y-%m-%d')}_to_{block_end_date.strftime('%Y-%m-%d')}.csv",
                index=False,
            )
        except:
            raise Exception("Could not save the query locally.")
        else:
            start_date = block_end_date + pd.Timedelta(1, "D")
            # Update checkpoint file with the current progress
            with open(checkpoint_file, "w") as f:
                f.write(str(start_date))

    # Assemble all saved data blocks into the final result
    final_result = None

    files = sorted(glob.glob(str(queryfolder / f"{name}_*.csv")))

    final_result = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True,
    )

    # Print end of query
    print(f" || \033[32mQuery finished.\033[0m")

    return final_result


def save_df_to_cdf(
    dataframe: pd.DataFrame,
    path: str,
    **kwargs: Union[str, int],
) -> None:
    """
    Save a pandas dataframe to a netCDF file.

    Parameters:
    -----------
    dataframe: pandas.DataFrame
        A pandas DataFrame.
    path: str
        A string representing the path where the data should be saved.
    **kwargs: Union[str, int]
        Optional arguments that allow to pass additional information about the data:
        - data_type: A string representing the type of the data ('aggregate' or 'indicator').
        - baseline_start: A string representing the start date of the baseline period.
        - baseline_end: A string representing the end date of the baseline period.
        - author: A string representing the name of the person who created the file.
        - data_name: A string representing the name of the data.
        - missing_days: An integer representing the number of missing days.

    Returns:
    --------
        None
    """

    # Convert pandas dataframe to xarray dataset
    xr_data = xr.Dataset.from_dataframe(dataframe)

    # Get period start and end date
    period_start = dataframe.date.min().strftime("%Y-%m-%d")
    period_end = dataframe.date.max().strftime("%Y-%m-%d")

    # Extract optional arguments
    baseline_start = kwargs.pop("baseline_start", "")
    baseline_end = kwargs.pop("baseline_end", "")
    author = kwargs.pop("author", "")
    name = kwargs.pop("data_name", "")
    missing_days = kwargs.pop("missing_data", "")

    # Add attributes to the netCDF file
    if "data_type" in kwargs:
        data_type = kwargs.get("data_type")
        xr_data.attrs["name"] = name
        xr_data.attrs["created_on"] = time.strftime("%Y-%m-%d")
        xr_data.attrs["created_by"] = author

        if data_type == "aggregate":
            xr_data.attrs["data_type"] = "CDR aggregate"

        elif data_type == "indicator":
            xr_data.attrs["data_type"] = "CDR indicator"

            if "" in {baseline_start, baseline_end}:
                xr_data.attrs["baseline_period"] = "no baseline"
            else:
                xr_data.attrs["baseline_period"] = f"{baseline_start} to {baseline_end}"

        else:
            raise Exception(
                "The chosen data type should be one of 'aggregate' or 'indicator'."
            )
    else:
        raise Exception(
            "Please provide a 'data_type' for the dataframe ('aggregate' or 'indicator')."
        )

    xr_data.attrs["time_period"] = f"{period_start} to {period_end}"
    xr_data.attrs["missing_days"] = missing_days

    # Save the netCDF file to the specified path
    xr_data.to_netcdf(path)


### 3. Plotting functions
#### a. Shared function


def flow_legend(fig_ax: mpl.axes.Axes, df_flows, width_multiplier) -> mpl.axes.Axes:
    """
    Create a legend for a flow map visualization.

    Parameters
    ----------
    fig_ax : mpl.axes.Axes
        The axes object to add the legend to.

    Returns
    -------
    fig_ax : mpl.axes.Axes
        A modified version of the axes object with the legend added.
    """

    LegendElement = [
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=None,
            fill=False,
            visible=False,
            label="$\\bf{Flows}$",
        ),
        mpatches.Patch(facecolor="#701F53", edgecolor="none", label="Origin"),
        mpatches.Patch(facecolor="#27B288", edgecolor="none", label="Destination"),
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=None,
            fill=False,
            visible=False,
            label="$\\bf{Values}$",
        ),
        mlines.Line2D(
            [0, 0],
            [1, 1],
            color=dark_grey,
            lw=df_flows["flow_value_median"].min() * width_multiplier,
            label=f'{round(df_flows["flow_value_median"].min(),2)} %',
        ),
        mlines.Line2D(
            [0, 0],
            [1, 1],
            color=dark_grey,
            lw=df_flows["flow_value_median"].max() * width_multiplier,
            label=f'{round(df_flows["flow_value_median"].max(),2)} %',
        ),
    ]

    leg = fig_ax.legend(
        handles=LegendElement,
        bbox_to_anchor=[1, 1, 0.0, 0.0],
        loc="upper right",
        facecolor="white",
        framealpha=0.8,
    )

    return fig_ax


def add_scalebar(fig_ax: mpl.axes.Axes) -> mpl.axes.Axes:
    """
    Adds a scale bar to the given matplotlib axes object.

    Parameters
    ----------
    fig_ax : mpl.axes.Axes
        The axes object to add the scale bar to.

    Returns
    -------
    fig_ax : mpl.axes.Axes
        The modified axes object with the added scale bar.
    """

    fig_ax.add_artist(
        ScaleBar(
            dx=1,
            location="lower right",
            scale_loc="top",
            width_fraction=0.004,
            border_pad=1,
            box_alpha=0,
        )
    )

    return fig_ax


def add_plot_basemap(fig_ax: mpl.axes.Axes) -> mpl.axes.Axes:
    """
    Add a basemap to a plot. Modified to be offline and be Haiti Specific.

    Parameters:
    -----------
    fig_ax : mpl.axes.Axes
        A Matplotlib Axes object where the basemap will be plotted.

    Returns:
    --------
    fig_ax: plt.Axes
        The modified input ax object with a basemap.
    """

    cx.add_basemap(
        ax=fig_ax,
        source=os.environ["MAPBOX_WMTS_URL"],
        attribution=xyz.CartoDB.VoyagerNoLabels.attribution,
        attribution_size=5,
        zorder=0,
    )

    # cx.add_basemap(
    #     ax=fig_ax,
    #     source=xyz.CartoDB.VoyagerOnlyLabels.build_url(scale_factor="@2x"),
    #     attribution=False,
    #     alpha=1,
    #     zorder=5,
    # )

    return fig_ax


def map_boundaries(
    fig_ax: mpl.axes.Axes,
    geodata: gpd.GeoDataFrame,
    **kwargs,
) -> mpl.axes.Axes:
    """
    Set map boundaries based on the total_bounds of the geodata or on provided boundaries.

    Parameters:
    -----------
    fig_ax: mpl.axes.Axes
        A Matplotlib Axes object to set the boundaries.
    geodata: gpd.GeoDataFrame
        A GeoDataFrame to get the total bounds and set the limits of the map.
    boundaries: np.ndarray, optional
        An array with the limits of the map in the following order: [minx, miny, maxx, maxy].
        If provided, the boundaries will be set to these values.

    Returns:
    --------
    fig_ax: mpl.axes.Axes
        The input ax object with the new limits set.
    """

    boundaries = kwargs.pop("boundaries", None)

    if boundaries is None:
        # Select boundaries of the geodata
        minx, miny, maxx, maxy = geodata.total_bounds
    else:
        minx, miny, maxx, maxy = boundaries

    # perc_lim_x = 1 / 100
    # perc_lim_y = 1 / 100

    # # Set the x and y limits of the map with the boundaries values
    # fig_ax.set_xlim(
    #     minx - np.absolute(minx * perc_lim_x), maxx + np.absolute(maxx * perc_lim_x)
    # )
    # fig_ax.set_ylim(
    #     miny - np.absolute(miny * perc_lim_y), maxy + np.absolute(maxy * perc_lim_y)
    # )

    adj_square = np.abs((maxx - minx) - (maxy - miny)) / 2
    adj_small = np.abs((maxx - minx) - (maxy - miny)) / 4

    aspect = kwargs.pop("aspect", 1)

    mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2

    dx, dy = maxx - minx, maxy - miny

    if dx < dy:
        fig_ax.set_xlim(
            mid_x - 0.5 * aspect * (dx + np.abs(dy - dx)) - adj_small,
            mid_x + 0.5 * aspect * (dx + np.abs(dy - dx)) + adj_small,
        )

        fig_ax.set_ylim(mid_y - (0.5 * dy) - adj_small, mid_y + (0.5 * dy) + adj_small)

    elif dx > dy:
        fig_ax.set_xlim(
            mid_x - (0.5 * aspect * dx) - adj_small,
            mid_x + (0.5 * aspect * dx) + aspect * adj_small,
        )

        fig_ax.set_ylim(
            mid_y - 0.5 * (dy + np.abs(dy - dx)) - adj_small,
            mid_y + 0.5 * (dy + np.abs(dy - dx)) + aspect * adj_small,
        )
    else:
        fig_ax.set_xlim(
            mid_x - (0.5 * aspect * dx) - adj_small * aspect,
            mid_x + (0.5 * aspect * dx) + adj_small * aspect,
        )

        fig_ax.set_ylim(mid_y - (0.5 * dy) - adj_small, mid_y + (0.5 * dy) + adj_small)

    return fig_ax


def xaxis_dates(fig_ax: plt.Axes) -> plt.Axes:
    """
    Apply styling for plots with dates on the x-axis.

    Parameters
    ----------
    fig_ax : matplotlib.axes.Axes
        The matplotlib axes object to apply the styling to.

    Returns
    -------
    matplotlib.axes.Axes
        The modified matplotlib axes object.

    """
    # Set up locators and formatters for x-axis ticks
    y_loc = mdates.YearLocator()
    m_loc = mdates.MonthLocator(interval=1)
    y_fmt = mdates.DateFormatter("%b\n%Y")
    m_fmt = mdates.DateFormatter("%b")

    fig_ax.xaxis.set_major_locator(y_loc)
    fig_ax.xaxis.set_minor_locator(m_loc)
    fig_ax.xaxis.set_major_formatter(y_fmt)
    fig_ax.xaxis.set_minor_formatter(m_fmt)

    # Rotate x-axis labels and align to the left
    fig_ax.tick_params(axis="x", which="both", labelrotation=0)
    for label in fig_ax.get_xticklabels(which="both"):
        label.set_horizontalalignment("left")

    return fig_ax


#### b. Plotting functions


def create_interactive_map(
    tuples_gpd_admins: list[tuple[str, str, gpd.GeoDataFrame]]
) -> folium.folium.Map:
    """
    Create an interactive map from a list of Geopandas dataframes

    Parameters
    ----------
    tuples_gpd_admins : list of tuples
        List of tuples, where each tuple contains layer name, column name, and a Geopandas dataframe.

    Returns
    -------
    folium.folium.Map
        An interactive leaflet map created from the Geopandas dataframes.
    """

    # Create a leaflet map using the first Geopandas dataframe
    created_map = tuples_gpd_admins[0][2].explore(
        color="black",
        style_kwds=dict(
            fillColor="#3388ff",
            fill=True,
            opacity=1.0,
            fillOpacity=0.0,
            interactive=True,
        ),
        tiles="OpenStreetMap",
        tooltip=str(tuples_gpd_admins[0][1]),
        tooltip_kwds={},
        popup_kwds={"labels": False},
        smooth_factor=2,
        name=str(tuples_gpd_admins[0][0]),
    )

    # Add subsequent Geopandas dataframes to the map
    for t in tuples_gpd_admins[1:]:
        created_map = t[2].explore(
            m=created_map,
            color="black",
            style_kwds=dict(
                fillColor="#3388ff",
                fill=True,
                opacity=1.0,
                fillOpacity=0.0,
                interactive=True,
            ),
            tooltip=str(t[1]),
            tooltip_kwds={},
            popup_kwds={"labels": False},
            smooth_factor=2,
            name=str(t[0]),
        )

    # Add layer control to the map
    folium.LayerControl().add_to(created_map)

    return created_map


def plot_available_data(dates_table):
    """
    Plots the availability of CDR data from VFGH server.

    Parameters
    ----------
    dates_table: pandas DataFrame
        DataFrame containing the dates and availability of CDR data.

    Returns
    -------
    fig: matplotlib Figure
        The plot of the availability of CDR data.

    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    dates_table.plot(
        x="date",
        y="available_date",
        c="available_date",
        kind="scatter",
        colormap="RdYlGn",
        vmin=0,
        vmax=1,
        colorbar=False,
        linewidths=600 / len(dates_table),
        s=3000,
        marker="|",
        ax=ax,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Is the CDR data available \n from VFGH server?")

    x_min = dates_table["date"].min()
    x_max = dates_table["date"].max()
    ax.set_xlim(left=x_min, right=x_max)
    xaxis_dates(ax)

    ax.yaxis.set_ticks([0, 1])
    ax.set_ylim([-0.7, 1.5])
    ax.set_yticklabels(["No", "Yes"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    return fig


def plot_baseline(
    dataframe_to_plot: pd.DataFrame,
    value_to_plot: str,
    baseline_start: pd.Timestamp = None,
    baseline_end: pd.Timestamp = None,
    y_min: float = None,
    y_max: float = None,
    x_min: float = None,
    x_max: float = None,
) -> plt.Figure:
    """
    Plot a "fancy" Time series plots show the baseline period, missing data (with a dashed line),
    and highlight extreme values compared to their own weekdays, saturdays, and/or sundays groups
    (with the median value of groups across the time period also highlighted)

    Parameters
    ----------
    dataframe_to_plot: pandas.DataFrame
        Pandas DataFrame with a "date" column and a column with the values to plot.
    value_to_plot: str
        The name of the column with the values to plot.
    baseline_start: pandas.Timestamp
        Pandas Timestamp with the start date of the baseline period. Default is None.
    baseline_end: pandas.Timestamp
        Pandas Timestamp with the end date of the baseline period. Default is None.
    y_min: float
        A float with the minimum value to display on the y-axis. Default is None.
    y_max: float
        A float with the maximum value to display on the y-axis. Default is None.
    x_min: float
        A float with the minimum value to display on the x-axis. Default is None.
    x_max: float
        A float with the maximum value to display on the x-axis. Default is None.

    Returns
    -------
    A Matplotlib Figure object.

    """

    # Copy the dataframe to avoid modifying the original
    df = dataframe_to_plot.copy()
    df = iso_calendar(df)

    # Calculate medians (weekdays, saturdays, sundays) over the full period
    medians = period_medians(df, value_to_plot)

    # Calculate Z values for each point
    df = z_score_for_median_groups(df, value_to_plot)

    # Create group of days
    df = days_groups(df, value_to_plot)

    # Color for plots
    color_to_use = fm_div_colors[0]

    # Limits for x-axis
    if x_min is None:
        x_min = df["date"].min()
    else:
        x_min = pd.to_datetime(x_min)

    if x_max is None:
        x_max = df["date"].max()
    else:
        x_max = pd.to_datetime(x_max)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    if (baseline_start) and (baseline_end):
        ax.axvspan(
            pd.Timestamp(baseline_start),
            pd.Timestamp(baseline_end),
            alpha=0.5,
            color="#E0E0E0",
            label="Baseline period",
        )

    for name, med in medians.items():
        ax.hlines(
            med,
            linestyle="dotted",
            linewidth=1,
            color="black",
            xmin=x_min,
            xmax=x_max,
            zorder=5,
        )
        ax.text(x_max + timedelta(days=1), med, name, ha="left", va="center")

    mkr_dict = {
        "saturday": ["o", "white"],
        "sunday": ["d", "white"],
        "week": [".", color_to_use],
    }
    for group in mkr_dict:
        df.plot(
            ax=ax,
            x="date",
            y=group,
            kind="scatter",
            c=mkr_dict[group][1],
            s=25,
            edgecolor=color_to_use,
            marker=mkr_dict[group][0],
            zorder=20,
        )

    if df["z_score_extreme_value"].any():
        df.plot(
            ax=ax,
            x="date",
            y="z_score_extreme_value",
            kind="scatter",
            c=fm_colors[2],
            s=10,
            # edgecolor=color_to_use,
            label="Extreme Z score (+/- 3 MAD)",
            marker="o",
            zorder=25,
        )

    df.dropna(subset=[value_to_plot]).plot(
        ax=ax,
        x="date",
        y=value_to_plot,
        ls="--",
        linewidth=0.75,
        color=color_to_use,
        legend=False,
        zorder=10,
    )

    df.plot(
        ax=ax,
        x="date",
        y=value_to_plot,
        ls="-",
        linewidth=1,
        color=color_to_use,
        legend=False,
        zorder=10,
    )

    h, l = ax.get_legend_handles_labels()
    if (baseline_start) and (baseline_end):
        if df["z_score_extreme_value"].any():
            ax.legend(h[:2], l[:2], framealpha=0.8)
        else:
            ax.legend(h[:1], l[:1], framealpha=0.8)
    else:
        if df["z_score_extreme_value"].any():
            ax.legend(h[:1], l[:1], framealpha=0.8)
        else:
            ax.legend(h[:0], l[:0], framealpha=0.8)

    ax.yaxis.grid()
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("", fontsize=12)

    if all((y_min, y_max)) or any((y_min == 0, y_max == 0)):
        ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_xlim(left=x_min, right=x_max)
    xaxis_dates(ax)

    return fig


def iso_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns with ISO calendar information (year, week, and day) to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to which ISO calendar columns should be added. It is assumed that the input DataFrame
        has a column called "date" containing datetime objects.

    Returns
    -------
    df : pandas.DataFrame
        A new DataFrame with columns for year, week, and day of week added, based on the ISO calendar.
    """

    df = pd.concat(
        [
            df,
            df["date"].dt.isocalendar(),
        ],
        axis=1,
    )
    return df


def days_groups(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """
    Creates new columns in a DataFrame that group a specified value by week, saturday, and sunday.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to which group columns should be added.
    value : str
        The name of the column to be grouped.

    Returns
    -------
    df : pandas.DataFrame
        The input DataFrame with new columns added for each group.
    """
    # df = iso_calendar(df)
    df["week"] = np.where(df["day"].apply(lambda x: x <= 5), df[value], np.nan)
    df["saturday"] = np.where(df["day"].apply(lambda x: x == 6), df[value], np.nan)
    df["sunday"] = np.where(df["day"].apply(lambda x: x == 7), df[value], np.nan)

    return df


def period_medians(df: pd.DataFrame, value: str) -> dict:
    """
    Calculates the median of a specified value for the week, saturday, and sunday, over the entire period of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be used for calculating medians.
    value : str
        The name of the column for which medians should be calculated.

    Returns
    -------
    dict
        A dictionary with keys "Week", "Sat", and "Sun" representing the median values for each group.
    """

    # df = iso_calendar(df)
    ## Calculate medians over the full period
    median_weekday = df[df["day"] <= 5][value].median()
    median_saturday = df[df["day"] == 6][value].median()
    median_sunday = df[df["day"] == 7][value].median()

    return {
        "Week": median_weekday,
        "Sat": median_saturday,
        "Sun": median_sunday,
    }


def z_score_for_median_groups(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """
    Calculates the z-score of a specified daily value for the week, saturday, and sunday groups in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be used for calculating z-scores.
    value : str
        The name of the column for which z-scores should be calculated.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with columns for z-scores of the specified value compared to their group.
    """

    if "day" not in df:
        df = iso_calendar(df)
    weekday_group = df[df["day"] <= 5].copy()
    saturday_groups = df[df["day"] == 6].copy()
    sunday_groups = df[df["day"] == 7].copy()

    for group in [weekday_group, saturday_groups, sunday_groups]:
        group = z_score(group, value)

    df = pd.concat([weekday_group, saturday_groups, sunday_groups])
    df = df.sort_values(by=["date"])

    return df


import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, Point


def plot_arc(A, B, width, ax, one_color=False):
    dist = A.distance(B)

    # midpoint C
    C = Point((A.x + B.x) / 2, (A.y + B.y) / 2)  # midpoint

    # shift midpoint in either direction by some proportion of the length of the line to point D
    arc_peak_modifier = 0.1  # 10% of line width
    theta = np.arctan2((A.x - B.x), (A.y - B.y))
    D = Point(
        C.x + (np.cos(theta) * (dist * arc_peak_modifier)),
        C.y - (np.sin(theta) * (dist * arc_peak_modifier)),
    )

    line = LineString([A, B])
    normal = LineString([C, D])
    curve = LineString([A, D, B])

    # Circle centroid P, found from the three points A, D, B
    x, y, z = A.x + A.y * 1j, B.x + B.y * 1j, D.x + D.y * 1j
    w = z - x
    w /= y - x
    c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x

    P = Point(-c.real, -c.imag)

    # circle radius
    radius = P.distance(A)

    # Angles to draw arc
    alpha = np.arctan2(P.y - A.y, P.x - A.x)
    beta = np.arctan2(P.y - B.y, P.x - B.x)

    if (alpha - beta) < -6 * np.pi / 4:
        alpha += 2 * np.pi

    # Resolution of arc (number of smaller lines the overall arc is composed of)
    t = 40

    # Arc points
    angle = np.linspace(alpha, beta, t)
    arc_x = P.x - radius * np.cos(angle)
    arc_y = P.y - radius * np.sin(angle)

    # make a list of points in arc, then concat with self, but shifted by one to get many straight line segments
    # that form the curve
    points = np.array([arc_x, arc_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap=cmap_fm_flows,
        # color=fm_flow_colors[0],
        linewidth=width,
        linestyles="solid",
        path_effects=[path_effects.Stroke(capstyle="round")],
        alpha=1,
    )
    if one_color:
        lc.set_color(fm_flow_colors[0])
    lc.set_array(np.arange(len(segments)))
    ax.add_collection(lc)

    return segments


def plot_arcs(
    df,
    ax,
    origin="origin",
    destination="destination",
    arc_width="value",
    log_scale=False,
    width_multiplier=1,
    one_color=False,
):
    if log_scale:
        _ = df.apply(
            lambda arc: plot_arc(
                arc[origin],
                arc[destination],
                width_multiplier * np.log(arc[arc_width]),
                ax,
                one_color,
            ),
            axis=1,
        )
    else:
        _ = df.apply(
            lambda arc: plot_arc(
                arc[origin],
                arc[destination],
                width_multiplier * arc[arc_width],
                ax,
                one_color,
            ),
            axis=1,
        )


def plot_map_arcs(
    geodata_boundaries: gpd.GeoDataFrame,
    df_flows: pd.DataFrame,
    value_to_plot: str,
    log_scale: bool = True,
    width_multiplier: float = 0.1,
    **kwargs,
) -> plt.Figure:
    """
    Plot a map with population flows between geographic areas using arcs (origin-destination).

    Parameters
    ----------
    geodata_boundaries : geopandas.GeoDataFrame
        GeoDataFrame with the boundaries of the geographic areas to be displayed.
    df_flows : pandas.GeoDataFrame
        DataFrame with the population flows to be displayed.
    value_to_plot : str
        Name of the column in `df_flows` with the values to use to determine the width of the arcs.
    log_scale : bool, optional
        If True, use a logarithmic scale for the width of the arcs. Default is True.
    width_multiplier : float, optional
        Multiplier for the width of the arcs. Default is 0.1.
    **kwargs : optional
        Additional keyword arguments to pass to `map_boundaries` function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the plot.
    """

    # Change the spatial coordinate system to Web (Spherical) Mercator
    geodata_boundaries = geodata_boundaries.to_crs(3857)

    # Recalculate the centroids
    geodata_boundaries["centroid"] = geodata_boundaries.centroid

    flows_with_geo = create_flows_data(geodata_boundaries, df_flows)

    # Manual boundaries?
    boundaries = kwargs.pop("boundaries", None)
    if isinstance(boundaries, gpd.GeoDataFrame):
        boundaries = boundaries.to_crs(3857).total_bounds

    # Select districts with flows
    districts_to_color = geodata_boundaries[
        geodata_boundaries["pcod"].isin(
            pd.unique(df_flows[["pcod_from", "pcod_to"]].values.ravel("K"))
        )
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    geodata_boundaries.plot(
        linewidth=0.5,
        edgecolor=fm_colors[2],
        facecolor="none",
        ax=ax,
    )

    districts_to_color.plot(
        linewidth=0.5,
        edgecolor=fm_colors[2],
        facecolor=fm_colors[2],
        alpha=0.15,
        ax=ax,
    )

    plot_arcs(
        flows_with_geo,
        ax,
        origin="origin_geometry",
        destination="destination_geometry",
        arc_width=value_to_plot,
        log_scale=log_scale,
        width_multiplier=width_multiplier,
    )

    map_boundaries(ax, geodata_boundaries, boundaries=boundaries)
    flow_legend(ax, df_flows, width_multiplier)
    add_plot_basemap(ax)
    add_scalebar(ax)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def create_flows_data(
    geodata_boundaries: gpd.GeoDataFrame, df_flows: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame with flows data.

    Parameters
    ----------
    geodata_boundaries : gpd.GeoDataFrame
        GeoDataFrame with geometry for each region.
    df_flows : pd.DataFrame
        DataFrame with population flow data.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with flows data.
    """

    # Merge the geodataframe with the flow data
    flows_with_geo = (
        df_flows.merge(geodata_boundaries, left_on="pcod_from", right_on="pcod")
        .drop(columns="pcod")
        .merge(
            geodata_boundaries,
            left_on="pcod_to",
            right_on="pcod",
            suffixes=("_from", "_to"),
        )
        .drop(columns="pcod")
    )

    # Calculate origin and destination geometries
    flows_with_geo["origin_geometry"] = flows_with_geo.apply(
        lambda flow: Point(flow.centroid_from.x, flow.centroid_from.y),
        axis=1,
    )

    flows_with_geo["destination_geometry"] = flows_with_geo.apply(
        lambda flow: Point(flow.centroid_to.x, flow.centroid_to.y),
        axis=1,
    )

    return flows_with_geo


def plot_map_proportions(
    geodata_boundaries: gpd.GeoDataFrame,
    df_proportion: pd.DataFrame,
    value_to_plot: str,
    pcod_to_use: str,
    geodata_origin: gpd.GeoDataFrame = gpd.GeoDataFrame(),
) -> plt.Figure:
    """
    Plots a choropleth map with the proportion data of the input DataFrame for each boundary in the geodata.

    Parameters
    ----------
    geodata_boundaries : geopandas.GeoDataFrame
        GeoDataFrame containing the geographical boundaries.
    df_proportion : pandas.DataFrame
        DataFrame containing the data to plot in the map.
    value_to_plot : str
        Name of the column to plot in the map.
    pcod_to_use : str
        Name of the column to join the proportion data with the boundaries data (i.e. id column).
    geodata_origin : geopandas.GeoDataFrame
        Optional GeoDataFrame with origin spatial data.

    Returns:
    fig : matplotlib.figure.Figure
        Figure object with the plot.
    """

    # Change the spatial coordinate system to Web (Spherical) Mercator
    geodata_boundaries = geodata_boundaries.to_crs(3857)

    # Merge the data to plot with the geodata
    prop_with_geo = geodata_boundaries.merge(
        df_proportion, left_on="pcod", right_on=pcod_to_use
    )

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot the geographical boundaries
    map_boundaries(ax, prop_with_geo)

    # Add a colorbar to the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Plot the geographical boundaries without filling them
    geodata_boundaries.plot(
        linewidth=0.25,
        edgecolor="black",
        facecolor="none",
        ax=ax,
    )

    # Plot the proportional data with a color map
    prop_with_geo.plot(
        value_to_plot,
        legend=True,
        norm=matplotlib.colors.LogNorm(
            vmin=prop_with_geo[value_to_plot].min(),
            vmax=prop_with_geo[value_to_plot].max(),
        ),
        linewidth=0.25,
        edgecolor="black",
        cmap=cmap_fm_seq,
        ax=ax,
        cax=cax,
    )

    # Plot the origin GeoDataFrame
    if not geodata_origin.empty:
        geodata_origin = geodata_origin.to_crs(3857)
        geodata_origin.plot(
            linewidth=0.5,
            edgecolor="white",
            facecolor=fm_colors[2],
            ax=ax,
        )

    # Add a basemap to the plot
    add_plot_basemap(ax)
    # Add a scale bar to the plot
    add_scalebar(ax)

    # Remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def plot_map_proportions_with_arcs(
    geodata_boundaries: gpd.GeoDataFrame,
    dataframe_stability: pd.DataFrame,
    value_to_plot: str,
    pcod_to_use: str,
    geodata_origin: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    **kwargs,
) -> matplotlib.figure.Figure:
    """
    Plot a choropleth map of proportions data and arcs (origin-destination) between administrative areas.

    Parameters:
    -----------
    geodata_boundaries : gpd.GeoDataFrame
        The GeoDataFrame of the boundaries of each area to plot.
    dataframe_stability : pd.DataFrame
        The DataFrame containing the "stability" and population flow values between each areas.
    value_to_plot : str
        Name of the column to plot in the map.
    pcod_to_use : str
        The column name of the unique identifier for each region in `geodata_boundaries`.
    geodata_origin : gpd.GeoDataFrame, optional
        The GeoDataFrame of the origin points to plot arcs from.
    **kwargs : Any
        Additional keyword arguments to pass to `plot_arcs`.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The resulting figure object.
    """

    # Create a combined spatial dataset
    combined_geo = pd.concat([geodata_boundaries, geodata_origin], ignore_index=True)

    # Change the spatial coordinate system to Web (Spherical) Mercator
    geodata_boundaries = geodata_boundaries.to_crs(3857)
    combined_geo = combined_geo.to_crs(3857)

    # Recalculate the centroids
    combined_geo["centroid"] = combined_geo.centroid

    flows_with_geo = create_flows_data(combined_geo, dataframe_stability)

    prop_with_geo = geodata_boundaries.merge(
        dataframe_stability, left_on="pcod", right_on=pcod_to_use
    )

    # Manual boundaries?
    boundaries = kwargs.pop("boundaries", None)
    if isinstance(boundaries, gpd.GeoDataFrame):
        boundaries = boundaries.to_crs(3857).total_bounds

    # Select trend to plot
    if "z_score_bool" in dataframe_stability.columns:
        hatch_geo = prop_with_geo[prop_with_geo["z_score_bool"] == True]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Get an accurate colorbar legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    geodata_boundaries.plot(
        linewidth=0.5,
        edgecolor=fm_colors[2],
        facecolor="none",
        ax=ax,
    )

    # The color normalization will be linear over the log-transformed bounds.
    norm = matplotlib.colors.Normalize(vmin=np.log10(1), vmax=np.log10(50))

    # Add a new column for the log-transformed values
    prop_with_geo["log_" + value_to_plot] = np.log10(prop_with_geo[value_to_plot])

    prop_with_geo.plot(
        "log_" + value_to_plot,
        legend=True,
        norm=norm,
        linewidth=0.5,
        edgecolor=fm_colors[2],
        cmap=cmap_fm_seq,
        ax=ax,
        cax=cax,
        legend_kwds={"label": "Relocation size [%]"},
    )

    # Use ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_fm_seq, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, cax=cax)

    # The ticks will now be the logarithm of the previous values
    log_ticks = np.log10([1, 5, 10, 20, 50])
    cb.set_ticks(log_ticks)
    cb.set_ticklabels([1, 5, 10, 20, 50])  # Original values, not their logarithms

    if not hatch_geo.empty:
        hatch_geo.plot(
            column="z_score_bool",
            categorical=True,
            linewidth=0,
            edgecolor=fm_colors[2],
            facecolor="none",
            hatch="/////",
            alpha=0.9,
            ax=ax,
        )

    if not geodata_origin.empty:
        geodata_origin = geodata_origin.to_crs(3857)
        geodata_origin.plot(
            linewidth=0.5,
            edgecolor=dark_grey,
            facecolor="#EAE5DF",
            ax=ax,
        )

    plot_arcs(
        flows_with_geo,
        ax,
        origin="origin_geometry",
        destination="destination_geometry",
        arc_width=value_to_plot,
        log_scale=False,
        width_multiplier=0.2,
    )

    LegendElement = [
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=None,
            fill=False,
            visible=False,
            label="$\\bf{Variation}$",
        ),
        mpatches.Patch(
            facecolor="w", hatch="/////", edgecolor=fm_colors[2], label="Fluctuating"
        ),
        mpatches.Patch(facecolor="w", edgecolor=fm_colors[2], label="Constant"),
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=None,
            fill=False,
            visible=False,
            label="$\\bf{Relocations}$",
        ),
        mpatches.Patch(facecolor="#701F53", edgecolor="none", label="Origin"),
        mpatches.Patch(facecolor="#27B288", edgecolor="none", label="Destination"),
    ]

    leg = ax.legend(
        handles=LegendElement,
        bbox_to_anchor=[1, 1, 0.0, 0.0],
        loc="upper right",
        facecolor="white",
        framealpha=0.8,
    )

    map_boundaries(ax, prop_with_geo, boundaries=boundaries)
    add_plot_basemap(ax)
    add_scalebar(ax)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def plot_map_proportions_with_nondirectional_arcs(
    geodata_boundaries: gpd.GeoDataFrame,
    dataframe_stability: pd.DataFrame,
    value_to_plot: str,
    pcod_to_use: str,
    geodata_origin: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    **kwargs,
) -> matplotlib.figure.Figure:
    """
    Plot a choropleth map of proportions data and arcs (nondirectional, i.e. both directions) between administrative areas.

    Parameters:
    -----------
    geodata_boundaries : gpd.GeoDataFrame
        The GeoDataFrame of the boundaries of each area to plot.
    dataframe_stability : pd.DataFrame
        The DataFrame containing the "stability" and population flow values between each areas.
    value_to_plot : str
        Name of the column to plot in the map.
    pcod_to_use : str
        The name of the column containing the identifying codes of the geographic boundaries.
    geodata_origin : gpd.GeoDataFrame, default is empty
        The GeoDataFrame of the origin points to plot arcs from.
    **kwargs : dict
        Optional arguments to pass to the function.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """

    # Create a combined spatial dataset
    combined_geo = pd.concat([geodata_boundaries, geodata_origin], ignore_index=True)

    # Change the spatial coordinate system to Web (Spherical) Mercator
    geodata_boundaries = geodata_boundaries.to_crs(3857)
    combined_geo = combined_geo.to_crs(3857)

    # Recalculate the centroids
    combined_geo["centroid"] = combined_geo.centroid

    # Create a "from and "to" columns to draw arcs
    df_stability = dataframe_stability.copy()
    df_stability["pcod_from"] = df_stability[pcod_to_use]
    df_stability["pcod_to"] = df_stability.apply(
        lambda x: geodata_origin["pcod"], axis=1
    )
    df_stability = df_stability.drop(columns=[pcod_to_use])

    flows_with_geo = create_flows_data(combined_geo, df_stability)

    prop_with_geo = geodata_boundaries.merge(
        df_stability, left_on="pcod", right_on="pcod_from"
    )

    # Manual boundaries?
    boundaries = kwargs.pop("boundaries", None)
    if isinstance(boundaries, gpd.GeoDataFrame):
        boundaries = boundaries.to_crs(3857).total_bounds

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Get an accurate legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    geodata_boundaries.plot(
        linewidth=0.5,
        edgecolor=fm_colors[2],
        facecolor="none",
        ax=ax,
    )

    prop_with_geo.plot(
        value_to_plot,
        legend=True,
        # norm=matplotlib.colors.LogNorm(
        #     vmin=prop_with_geo[value_to_plot].min(),
        #     vmax=prop_with_geo[value_to_plot].max(),
        # ),
        linewidth=0.5,
        edgecolor=fm_colors[2],
        cmap=cmap_fm_seq,
        ax=ax,
        cax=cax,
        legend_kwds={"label": "Daily trip size [%]"},
    )

    if not geodata_origin.empty:
        geodata_origin = geodata_origin.to_crs(3857)
        geodata_origin.plot(
            linewidth=0.5,
            edgecolor="white",
            facecolor=fm_colors[2],
            ax=ax,
        )

    plot_arcs(
        flows_with_geo,
        ax,
        origin="origin_geometry",
        destination="destination_geometry",
        arc_width="flow_value_median",
        log_scale=False,
        width_multiplier=0.3,
        one_color=True,
    )

    LegendElement = [
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=None,
            fill=False,
            visible=False,
            label="$\\bf{Flows}$",
        ),
        mpatches.Patch(facecolor="#701F53", edgecolor="none", label="Bi-directional"),
    ]

    leg = ax.legend(
        handles=LegendElement,
        bbox_to_anchor=[1, 1, 0.0, 0.0],
        loc="upper right",
        facecolor="white",
        framealpha=0.8,
    )

    map_boundaries(ax, prop_with_geo, boundaries=boundaries)
    add_plot_basemap(ax)
    add_scalebar(ax)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def plot_map_trends(
    geodata_boundaries: gpd.GeoDataFrame,
    dataframe_trends: pd.DataFrame,
    pcod_spatial: str,
    pcod_data: str,
    geodata_origin: gpd.GeoDataFrame = gpd.GeoDataFrame(),
    **kwargs,
) -> plt.Figure:
    """
    Plots the trends of of each given administrative area.

    Parameters
    ----------
    geodata_boundaries : geopandas.GeoDataFrame
        A GeoDataFrame with the spatial boundaries of the administrative areas.
    dataframe_trends : pandas.DataFrame
        A DataFrame with the trends data to plot.
    pcod_to_use : str
        The name of the column containing the region's pcods in the `geodata_boundaries` DataFrame.
    geodata_origin : geopandas.GeoDataFrame, optional
        A GeoDataFrame with a potential administrative area of origin.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the plotted data.
    """

    # Define the colors for the tendencies
    color_trends = {
        "decreasing": fm_div_colors[1],
        "increasing": fm_div_colors[7],
        "stable": "white",
    }

    # Change the spatial coordinate system to Web (Spherical) Mercator
    geodata_boundaries = geodata_boundaries.to_crs(3857)

    # Merge the trends data with the geospatial data
    trends_with_geo = geodata_boundaries.merge(
        dataframe_trends, left_on=pcod_spatial, right_on=pcod_data
    )

    # Apply color based on tendencies
    trends_with_geo["color"] = trends_with_geo.apply(
        lambda x: color_trends[x["tendency"]], axis=1
    )

    # Select trends to plot (depends on data to plot, presence or movements)
    if "variation" in dataframe_trends.columns:
        hatch_geo = trends_with_geo[trends_with_geo["variation"] == "fluctuating"]
        hatch_to_plot = "variation"
        title_label_A = "$\\bf{Population \, monthly \, variation}$"
        title_label_B = "$\\bf{Population \, trend}$"
        label_A = "Fluctuating"
        label_B = "Constant"

    elif "weekday_or_weekend" in dataframe_trends.columns:
        hatch_geo = trends_with_geo[
            trends_with_geo["weekday_or_weekend"] == "weekday > weekend"
        ]
        hatch_to_plot = "weekday_or_weekend"
        title_label_A = "$\\bf{Week\ trend}$"
        title_label_B = "$\\bf{Movements}$"
        label_A = "Weekdays > Weekend"
        label_B = "Weekdays < Weekend"

    title_label_C = "$\\bf{No \, data}$"

    # Create a new figure
    figsize = kwargs.pop("figsize", (10, 10))
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the geospatial data boundaries
    geodata_boundaries.plot(
        linewidth=0.5,
        edgecolor=fm_colors[2],
        facecolor="fm_missing",
        ax=ax,
    )

    # Plot the trends data
    trends_with_geo.plot(
        categorical=True,
        linewidth=0.5,
        edgecolor=fm_colors[2],
        color=trends_with_geo.color,
        ax=ax,
    )

    # Add hatching to fluctuating trends
    if not hatch_geo.empty:
        hatch_geo.plot(
            column=hatch_to_plot,
            categorical=True,
            linewidth=0,
            edgecolor=fm_colors[2],
            facecolor="none",
            hatch="/////",
            alpha=0.9,
            ax=ax,
        )

    # Plot the origin GeoDataFrame
    if not geodata_origin.empty:
        geodata_origin = geodata_origin.to_crs(3857)
        geodata_origin.plot(
            linewidth=1,
            edgecolor="white",
            facecolor=fm_colors[2],
            ax=ax,
        )

    # Add a legend to the plot
    LegendElement = [
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=None,
            fill=False,
            visible=False,
            label=title_label_A,
        ),
        mpatches.Patch(
            facecolor="w", hatch="/////", edgecolor=fm_colors[2], label=label_A
        ),
        mpatches.Patch(facecolor="w", edgecolor=fm_colors[2], label=label_B),
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=None,
            fill=False,
            visible=False,
            label=title_label_B,
        ),
        mpatches.Patch(
            facecolor=color_trends["increasing"], edgecolor="k", label="Increase"
        ),
        mpatches.Patch(facecolor=color_trends["stable"], edgecolor="k", label="Stable"),
        mpatches.Patch(
            facecolor=color_trends["decreasing"], edgecolor="k", label="Decrease"
        ),
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=None,
            fill=False,
            visible=False,
            label=title_label_C,
        ),
        mpatches.Patch(facecolor="fm_missing", edgecolor="k", label="No data"),
    ]
    leg = ax.legend(
        handles=LegendElement,
        # bbox_to_anchor=[0, 0, 0.0, 0.0],
        loc="upper left",
        facecolor="white",
        alignment="left",
    )

    # Change the map boundaries based on data extent
    map_boundaries(ax, trends_with_geo)
    # Add a basemap to the plot
    add_plot_basemap(ax)
    # Add a scale bar to the plot
    add_scalebar(ax)

    # Remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    return fig


def add_place_labels(
    fig_ax: plt.Axes, place_labels: gpd.GeoDataFrame, fontsize: int
) -> plt.Axes:
    """
    Plot place name labels on an axis. Assumes place names are in a GeoDataFrame with Web Mercator projection.

    Parameters
    ----------
    fig_ax : plt.Axes
        The axis to plot labels on.
    place_labels : gpd.GeoDataFrame
        The GeoDataFrame with labels (index) and location to plot (geometry column)
    fontsize : int
        font size
    """
    for x, y, label in zip(
        place_labels.geometry.x, place_labels.geometry.y, place_labels.index
    ):
        fig_ax.annotate(
            label,
            xy=(x, y),
            va="center",
            ha="center",
            zorder=12,
            annotation_clip=True,
            fontsize=fontsize,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="w"), pe.Normal()],
        )


def with_commas(x, pos):
    return format(x, ",.0f")


def plot_timeseries_trends(
    dataframe_to_plot: pd.DataFrame,
    dataframe_trends: pd.DataFrame,
    value_to_plot: str,
    pcod_to_use: str,
    time_resolution: str = "day",
    y_lim: list[int] = [-100, 100],
) -> plt.Figure:
    """
    Separate and plot the timeseries of a given trends (decrease, stable, increase) for each group defined
    by its variation (constant or fluctuating), and color the groups according to their trends.

    Parameters
    ----------
    dataframe_to_plot : pd.DataFrame
        The main dataframe with timeseries plot.
    dataframe_trends : pd.DataFrame
        The dataframe containing the trends for each group and corresponding timeseries.
    value_to_plot : str
        The column to plot.
    pcod_to_use : str
        The column that contains individual pcods (i.e. identifiers of each administrative area).
    time_resolution : str, optional
        The time resolution of the data (day/month), by default "day".
    y_lim : list[int], optional
        The y-axis limits of the plot, by default [-100, 100].

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the plotted data.
    """

    # Define the colors to use for the different trends.
    color_trends = {
        "decreasing": fm_div_colors[0],
        "increasing": fm_div_colors[8],
        "stable": stable,
    }

    # Determine the type of group that is used (variation or weekday/weekend).
    if "variation" in dataframe_trends.columns:
        plot_group = "variation"
        titles = [
            "Decrease - constant",
            "Stable - constant",
            "Increase - constant",
            "Decrease - fluctuating",
            "Stable - fluctuating",
            "Increase - fluctuating",
        ]

    elif "weekday_or_weekend" in dataframe_trends.columns:
        plot_group = "weekday_or_weekend"
        titles = [
            "Decrease - Weekdays < Weekend",
            "Stable - Weekdays < Weekend",
            "Increase - Weekdays < Weekend",
            "Decrease - Weekdays > Weekend",
            "Stable - Weekdays > Weekend",
            "Increase - Weekdays > Weekend",
        ]

    # Group the data by administrative identifier.
    grouped = dataframe_to_plot.groupby(pcod_to_use)

    fig, axes = plt.subplots(2, 3, figsize=(12, 4), sharey=True, sharex=True)

    for key in grouped.groups.keys():
        group_diff = dataframe_trends[dataframe_trends[pcod_to_use] == key][
            plot_group
        ].values[0]
        group_slope = dataframe_trends[dataframe_trends[pcod_to_use] == key][
            "tendency"
        ].values[0]

        if time_resolution == "day":
            data_to_plot = (
                grouped.get_group(key)[["date", value_to_plot]]
                .set_index("date")
                .asfreq("D")
            )
        elif time_resolution == "month":
            data_to_plot = (
                grouped.get_group(key)[["date", value_to_plot]]
                .set_index("date")
                .asfreq("MS")
            )

        if group_diff in ("constant", "weekday < weekend"):
            if group_slope == "decreasing":
                subplot_trends(
                    data_to_plot,
                    value_to_plot,
                    key,
                    color_trends["decreasing"],
                    axes[0, 0],
                )
            elif group_slope == "stable":
                subplot_trends(
                    data_to_plot, value_to_plot, key, color_trends["stable"], axes[0, 1]
                )
            elif group_slope == "increasing":
                subplot_trends(
                    data_to_plot,
                    value_to_plot,
                    key,
                    color_trends["increasing"],
                    axes[0, 2],
                )

        elif group_diff in ("fluctuating", "weekday > weekend"):
            if group_slope == "decreasing":
                subplot_trends(
                    data_to_plot,
                    value_to_plot,
                    key,
                    color_trends["decreasing"],
                    axes[1, 0],
                )
            elif group_slope == "stable":
                subplot_trends(
                    data_to_plot, value_to_plot, key, color_trends["stable"], axes[1, 1]
                )
            elif group_slope == "increasing":
                subplot_trends(
                    data_to_plot,
                    value_to_plot,
                    key,
                    color_trends["increasing"],
                    axes[1, 2],
                )

    for count, ax in enumerate(axes.flatten()):
        # The commented below does not work correctly, not sure why?
        # xaxis_dates(ax)
        ax.legend(
            [],
            framealpha=0,
        )
        ax.set_title(titles[count])
        ax.yaxis.grid()
        ax.set_xlabel("Date")
        ax.set_ylabel("% change \nfrom baseline")
        ax.set_ylim(y_lim)

    plt.subplots_adjust(wspace=0.1, hspace=0.25)

    return fig


def subplot_trends(data_to_plot, value_to_plot, key_label, plot_color, ax_to_plot):
    data_to_plot.plot(
        y=value_to_plot,
        use_index=True,
        label=key_label,
        ax=ax_to_plot,
        color=plot_color,
        linewidth=1,
        alpha=0.35,
        zorder=10,
    )


def plot_trip_distribution(
    dataframe_to_plot: pd.DataFrame,
    y_min: float = None,
    y_max: float = None,
    x_min: float = None,
    x_max: float = None,
) -> plt.Figure:
    """
    Plots the trip distribution data for different percentile ranges.

    Parameters
    ----------
    dataframe_to_plot: pandas.DataFrame
        A pandas DataFrame containing the trip data to plot.
    y_min: float
        A float with the minimum value to display on the y-axis. Default is None.
    y_max: float
        A float with the maximum value to display on the y-axis. Default is None.
    x_min: float
        A float with the minimum value to display on the x-axis. Default is None.
    x_max: float
        A float with the maximum value to display on the x-axis. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the plotted data.
    """

    # Make a copy of the input DataFrame to avoid modifying the original.
    df = dataframe_to_plot.copy()

    # Define percentiles to plot.
    perc_to_plot = ["2%", "50%", "90%", "95%", "98%"]

    # Compute rolling medians of percentiles.
    for perc in perc_to_plot:
        df[f"{perc}_rolling"] = df[perc].rolling(7, min_periods=5, center=True).median()

    # Limits for x-axis
    if x_min is None:
        x_min = df["date"].min()
    else:
        x_min = pd.to_datetime(x_min)

    if x_max is None:
        x_max = df["date"].max()
    else:
        x_max = pd.to_datetime(x_max)

    # Create figure and axes objects.
    fig, ax = plt.subplots(1, 1, figsize=(13, 4))

    # Create filled polygons between percentiles.
    for count in range(len(perc_to_plot[:-1])):
        ax.fill_between(
            x=df["date"],
            y1=df[f"{perc_to_plot[count]}_rolling"],
            y2=df[f"{perc_to_plot[count+1]}_rolling"],
            facecolor=fm_seq_colors[count],
            linewidth=0.0,
            alpha=0.5,
            label=f"{perc_to_plot[count]} to {perc_to_plot[count+1]}",
            zorder=20,
        )

    # Add legend and gridlines.
    ax.legend(title="$\\bf{Quantiles}$", frameon=True, framealpha=0.8)
    ax.yaxis.grid()

    # Set x-axis tick locators and formatters.
    xaxis_dates(ax)

    # Set axis labels and tick parameters.
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily trip size [km]")

    # Set y-axis limits
    if all((y_min, y_max)) or any((y_min == 0, y_max == 0)):
        ax.set_ylim(bottom=y_min, top=y_max)

    # Set x-axis limits
    ax.set_xlim(left=x_min, right=x_max)

    return fig


## C. Useful variables

# Current date
# timestamp = time.strftime("%Y-%m-%d")
#
### D. Common paths
# outputfolder = Path().resolve().parent / "Outputs"
# imagesfolder = outputfolder / "Images"
# aggregatesimgfolder = imagesfolder / "Aggregates"
# indicatorsimgfolder = imagesfolder / "Indicators"
# datafolder = outputfolder / "Data"
# sensitivefolder = datafolder / "Sensitive"
# aggregatesfolder = sensitivefolder / "Aggregates"
# geodatafolder = datafolder / "GeoData"
# indicatorsfolder = datafolder / "Indicators"
#
## Create the different folders only if they do not already exist
# for folder in [
#    outputfolder,
#    imagesfolder,
#    aggregatesimgfolder,
#    indicatorsimgfolder,
#    datafolder,
#    sensitivefolder,
#    aggregatesfolder,
#    geodatafolder,
#    indicatorsfolder,
# ]:
#    if not os.path.exists(folder):
#        os.makedirs(folder)
