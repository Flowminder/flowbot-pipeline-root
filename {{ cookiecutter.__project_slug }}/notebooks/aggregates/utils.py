# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import json
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from typing import Union as UnionType

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import flowclient as fc
from flowclient import (
    coalesced_location_spec,
    daily_location_spec,
    location_visits_spec,
    majority_location_spec,
    mobility_classification_spec,
    modal_location_spec,
)
from flowmachine.core.connection import Connection
from flowmachine.core.spatial_unit import AnySpatialUnit
from flowmachine.core.union import Union
from flowmachine.features.subscriber.per_subscriber_aggregate import (
    PerSubscriberAggregate,
)
from single_column_join import SingleColumnJoin
from total_locatable_periods import TotalLocatablePeriods


def get_date_in_month(
    dt: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    *,
    day_of_month: int,
    month_offset: int = 0,
) -> "datetime.date":
    """
    Get a date object for a specified day in a month.

    Parameters
    ----------
    dt : date, timestamp or date string
        Datetime from which month will be extracted (will be cast to pd.Timestamp)
    day_of_month : int
        Day within the month (e.g. 20 to get the 20th day of the month)
    month_offset : int, default 0
        Optionally specify a number of months to offset from the month specified by dt.
        Negative values offset to earlier months.

    Returns
    -------
    datetime.date
        Specified date within the specified month

    Examples
    --------
    >>> get_date_in_month("2021-12", day_of_month=20, month_offset=-1)
    datetime.date(2021, 11, 20)
    """
    return (
        pd.Timestamp(dt).replace(day=day_of_month) + pd.DateOffset(months=month_offset)
    ).date()


def rolling_window_over_date_range(
    *,
    start_date: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    end_date: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    window_length: int,
) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get the bounds for a n-day rolling window for each day in a specified date range.
    The 'focal' date for each window is the last included day (i.e. the first
    window is from `start_date - (window_length-1 days) <= dt < start_date + (1 day)`).

    Parameters
    ----------
    start_date : date, timestamp or date string
        The date corresponding to the first window
    end_date : date, timestamp or date string
        The date of the day _after_ the last window (i.e. exclusive upper bound)
    window_length : int
        Length of each window, in days

    Returns
    -------
    dict of str to tuples (pandas.Timestamp, pandas.Timestamp)
        Keys are the iso-formatted reference date for each window
        Values are start and end date/time for each window
        (intention is that lower bound is inclusive; upper bound is exclusive)
    """
    return {
        str(d.date()): (
            d - pd.Timedelta(days=window_length - 1),
            d + pd.Timedelta(days=1),
        )
        for d in pd.date_range(start_date, end_date, inclusive="left")
    }


def rolling_window_over_one_month(
    *,
    start_date: "datetime.date",
    window_length: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get the bounds for a n-day rolling window for each day in a month.
    The 'focal' date for each window is the last included day (i.e. the first
    window is from `start_date - (window_length-1 days) <= dt < start_date + (1 day)`).

    Parameters
    ----------
    start_date : date
        The date corresponding to the first window
        (one window per day will be returned for the month starting on start_date)
    window_length : int
        Length of each window, in days

    Returns
    -------
    dict of str to tuples (pandas.Timestamp, pandas.Timestamp)
        Keys are the iso-formatted reference date for each window
        Values are start and end date/time for each window
        (intention is that lower bound is inclusive; upper bound is exclusive)
    """
    return rolling_window_over_date_range(
        start_date=start_date,
        end_date=start_date + relativedelta(months=1),
        window_length=window_length,
    )


def monthly_subscriber_subset_query(
    dt: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    *,
    month_offset: int,
    month_start_day: int,
    window_length: int,
    min_call_days: int,
    tables: UnionType[str, List[str]],
    spatial_unit: AnySpatialUnit,
    dates_to_exclude: Optional[
        List[UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str]]
    ] = None,
) -> "flowmachine.core.join.Join":
    """
    This is a modified form of 'monthly_subscriber_subset_query', which
    explicitly considers only events at cells with a known location.

    Get a flowmachine query that defines the subset of subscribers classed as
    "active" in a given month.

    "Active" subscribers are those who:
    - Are active on >= 'min_call_days' distinct days per window on average (median),
    - Have at least one locatable event in every window,
    calculated over a 'window_length'-day rolling window, with one window for
    each day in the month.

    This query explicitly considers only events at cells with a known location.

    Parameters
    ----------
    dt : date, timestamp or date string
        Date/datetime from which month will be extracted
    month_offset : int
        Number of months to offset from the month specified by 'dt'.
        Negative values offset to earlier months.
    month_start_day : int
        Start day of the month.
        First window will end on this day of month 'dt + month_offset'
    window_length : int
        Length of windows over which active periods will be calculated
    min_call_days : int
        Minimum number of days per window (average) that a subscriber must be
        active to be included in the subset.
    tables : str or list of str
        Events tables to include
    spatial_unit : flowmachine.core.spatial_unit.*SpatialUnit
        Spatial unit defining the set of "locatable" cell IDs.
    dates_to_exclude : list of date, timestamp or date string
        List of data dates to be skipped

    Returns
    -------
    Query
        Subscriber subset query
    list of Query
        List of queries, one for each day in the month, that return the set of
        subscribers seen in every window up to that day.
        These queries may need to be stored (in order) before storing the main
        subscriber subset query, to avoid issues arising from constructing the
        unstored dependencies graph.
    """
    start_date = get_date_in_month(
        dt, day_of_month=month_start_day, month_offset=month_offset
    )
    rolling_windows = rolling_window_over_one_month(
        start_date=start_date, window_length=window_length
    )

    # Note: not recursively applying a subscriber subset here, because this would
    # mean each TotalLocatablePeriods query would depend on an entirely distinct
    # set of 7 daily sub-queries (whereas without a recursive subset, each daily
    # sub-query can be cached and re-used in 7 different windows).
    active_periods_queries = []
    # Iterate over windows in order here, to ensure eventual query ID is deterministic
    for window in sorted(rolling_windows.keys()):
        window_start, window_end = rolling_windows[window]
        try:
            active_periods_queries.append(
                TotalLocatablePeriods(
                    start=window_start,
                    total_periods=window_length,
                    period_length=1,
                    period_unit="days",
                    spatial_unit=spatial_unit,
                    table=tables,
                    periods_to_exclude=dates_to_exclude,
                )
            )
        except ValueError:
            # If all dates in this window are excluded, skip it
            pass

    # Note: Could do this with a 'reduce', but we want to return a list of all the partially-joined queries
    # so that these can be explicitly stored before the main query if necessary.
    active_in_every_window_subset = active_periods_queries[0]
    intermediate_joins = [active_in_every_window_subset]
    for q in active_periods_queries[1:]:
        active_in_every_window_subset = SingleColumnJoin(
            active_in_every_window_subset,
            q,
            column_name="subscriber",
        )
        intermediate_joins.append(active_in_every_window_subset)

    # Note: this query will calculate each subscriber's median call days per window
    # _only over the windows in which they were active_ - i.e. subscribers are not
    # assigned a count of 0 for windows in which they had no events at all.
    # Fortunately, this doesn't matter, because when combined with the
    # 'active in every window' subset we will only have subscribers who have an
    # "active periods" count in every window.
    average_call_days_above_threshold_subset = PerSubscriberAggregate(
        subscriber_query=Union(*active_periods_queries),
        agg_column="value",
        agg_method="median",
    ).numeric_subset(high=np.inf, low=min_call_days, col="value")

    # Combined subset is the intersection of the two subsets
    combined_subset = average_call_days_above_threshold_subset.join(
        active_in_every_window_subset,
        on_left="subscriber",
        how="inner",
        left_append="_average",
    )

    return combined_subset, intermediate_joins


def daily_home_location_specs(
    *,
    rolling_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    aggregation_unit: str,
    mapping_table: Optional[str] = None,
    geom_table: Optional[str] = None,
    geom_table_join_column: Optional[str] = None,
    subscriber_subset: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    dates_to_exclude: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    # TODO: Docstring
    query_specs = {}
    for ref_date, window in rolling_windows.items():
        dates_in_window = {
            str(d.date())
            for d in pd.date_range(window[0], window[1], freq="D", inclusive="left")
        }.difference(dates_to_exclude)
        print("*********")
        print(dates_in_window)
        print("*********")
        if dates_in_window:
            query_specs[ref_date] = modal_location_spec(
                locations=[
                    daily_location_spec(
                        date=date,
                        method="last",
                        aggregation_unit=aggregation_unit,
                        mapping_table=mapping_table,
                        geom_table=geom_table,
                        geom_table_join_column=geom_table_join_column,
                        subscriber_subset=subscriber_subset,
                        event_types=event_types,
                    )
                    for date in dates_in_window
                ]
            )
        else:
            # If this window is empty, ignore it
            pass
    return query_specs


def monthly_location_visits_spec(
    dt: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    *,
    month_offset: int,
    month_start_day: int,
    window_length: int,
    aggregation_unit: str,
    mapping_table: Optional[str] = None,
    geom_table: Optional[str] = None,
    geom_table_join_column: Optional[str] = None,
    subscriber_subset: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    dates_to_exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get a flowclient query spec for a 'location visits' query based on one month of
    modal-last-locations over a rolling 7-day window.

    This query forms a component of a "monthly home location" query spec.

    Parameters
    ----------
    dt : date, timestamp or date string
        Date/datetime from which month will be extracted
    month_offset : int
        Number of months to offset from the month specified by 'dt'.
        Negative values offset to earlier months.
    month_start_day : int
        Start day of the month.
        First window will end on this day of month 'dt + month_offset'
    window_length : int
        Length of windows over which modal locations will be calculated
    aggregation_unit : str
        Aggregation unit (e.g. "admin3") for subscriber locations
    mapping_table, geom_table, geom_table_join_column : str, optional
        Additional parameters to define mapping from cell IDs to spatial
        elements of aggregation unit
    subscriber_subset : str, optional
        Query ID of a query that defines a subset of subscribers to include
    event_types : list of str, optional
        Optionally specify the list of event types to include
    dates_to_exclude : list of str, optional
        List of iso-format dates (YYYY-mm-dd) to skip when calculating daily locations

    Returns
    -------
    dict
        A "location_visits" query specification

    See Also
    --------
    monthly_home_location_spec
    """
    if dates_to_exclude is None:
        dates_to_exclude = []
    start_date = get_date_in_month(
        dt, day_of_month=month_start_day, month_offset=month_offset
    )

    modal_last_locations = daily_home_location_specs(
        rolling_windows=rolling_window_over_one_month(
            start_date=start_date, window_length=window_length
        ),
        aggregation_unit=aggregation_unit,
        mapping_table=mapping_table,
        geom_table=geom_table,
        geom_table_join_column=geom_table_join_column,
        subscriber_subset=subscriber_subset,
        event_types=event_types,
        dates_to_exclude=dates_to_exclude,
    )

    return location_visits_spec(
        locations=[loc for d, loc in sorted(modal_last_locations.items())]
    )


def monthly_home_location_spec(
    dt: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    *,
    month_offset: int,
    month_start_day: int,
    window_length: int,
    lookback_n_months: Optional[int],
    aggregation_unit: str,
    mapping_table: Optional[str] = None,
    geom_table: Optional[str] = None,
    geom_table_join_column: Optional[str] = None,
    this_month_subscriber_subset: Optional[str] = None,
    last_month_subscriber_subset: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    dates_to_exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get a flowclient query spec for a monthly home location query.

    "Home location" is defined as:

    1. For each subscriber, each day, find the location that was most commonly
       their last location of the day during the 7-day window up to and
       including that day,
    2. For each location that was a subscriber's "modal last location" on any
       day in this month ("this month" is one calendar month starting on the
       specified month start date, e.g. 20 Feb to 19 March inclusive), count
       the number of days on which the location was that subscriber's "modal
       last location",
    3. If a particular location was a subscriber's "modal last location" on
       more than half of the days, assign that as the subscriber's home
       location this month,
    4. If not, but a particular location was a subscriber's "modal last
       location" on more than one third of the days this month _and_ was that
       subscriber's "modal last location" on more than half of the days _last_
       month ("last month" is one calendar month starting on the
       specified month start date, 'lookback_n_months' before "this month",
       e.g. 20 Jan to 19 Feb inclusive), assign that as the subscriber's home
       location this month,
    5. If neither (3) or (4) holds for any location for a particular
       subscriber, assign a "null" location to that subscriber this month.

    Note: if 'lookback_n_months' is None, step (4) will be skipped (i.e. a
    subscriber will only be locatable if they have a majority location "this
    month").

    Parameters
    ----------
    dt : date, timestamp or date string
        Date/datetime from which month will be extracted
    month_offset : int
        Number of months to offset from the month specified by 'dt'.
        Negative values offset to earlier months.
    month_start_day : int
        Start day of the month.
        First window will end on this day of month 'dt + month_offset'
    window_length : int
        Length of windows over which modal locations will be calculated
    lookback_n_months : int or None
        Number of months to look back for a previous majority location
    aggregation_unit : str
        Aggregation unit (e.g. "admin3") for subscriber locations
    mapping_table, geom_table, geom_table_join_column : str, optional
        Additional parameters to define mapping from cell IDs to spatial
        elements of aggregation unit
    this_month_subscriber_subset : str, optional
        Query ID of a query that defines a subset of subscribers for which home
        locations should be calculated this month
    last_month_subscriber_subset : str, optional
        Query ID of a query that defines a subset of subscribers for which
        majority locations should be calculated last month (for input to step
        (4))
    event_types : list of str, optional
        Optionally specify the list of event types to include
    dates_to_exclude : list of str, optional
        List of iso-format dates (YYYY-mm-dd) to skip when calculating daily locations

    Returns
    -------
    dict
        A "coalesced_location" or "majority_location" query specification
    """
    this_month_location_visits = monthly_location_visits_spec(
        dt,
        month_offset=month_offset,
        month_start_day=month_start_day,
        window_length=window_length,
        aggregation_unit=aggregation_unit,
        mapping_table=mapping_table,
        geom_table=geom_table,
        geom_table_join_column=geom_table_join_column,
        subscriber_subset=this_month_subscriber_subset,
        event_types=event_types,
        dates_to_exclude=dates_to_exclude,
    )

    this_month_majority_location = majority_location_spec(
        subscriber_location_weights=this_month_location_visits,
        include_unlocatable=True,
    )

    if lookback_n_months is None:
        return this_month_majority_location
    else:
        last_month_majority_location = majority_location_spec(
            subscriber_location_weights=monthly_location_visits_spec(
                dt,
                month_offset=month_offset - lookback_n_months,
                month_start_day=month_start_day,
                window_length=window_length,
                aggregation_unit=aggregation_unit,
                mapping_table=mapping_table,
                geom_table=geom_table,
                geom_table_join_column=geom_table_join_column,
                subscriber_subset=last_month_subscriber_subset,
                event_types=event_types,
                dates_to_exclude=dates_to_exclude,
            ),
            include_unlocatable=True,
        )

        return coalesced_location_spec(
            preferred_location=this_month_majority_location,
            fallback_location=last_month_majority_location,
            subscriber_location_weights=this_month_location_visits,
            weight_threshold=ceil(len(this_month_location_visits["locations"]) / 3),
        )


def _write_query_result(
    res: pd.DataFrame,
    filename: UnionType[str, Path],
    *,
    file_format: Optional[str] = None,
    overwrite: bool = False,
    attrs: Optional[Dict[str, Any]] = None,
) -> None:
    allowed_formats = {"csv": ".csv", "netcdf": ".nc"}
    formats_lookup = {value: key for key, value in allowed_formats.items()}

    filepath = Path(filename)

    if file_format is None:
        try:
            file_format = formats_lookup[filepath.suffix]
        except KeyError:
            raise ValueError(
                f"Unrecognised file extension. Specify file_format (one of {set(allowed_formats.keys())})."
            )
    elif file_format not in allowed_formats:
        raise ValueError(
            f"file_format must be one of {set(allowed_formats.keys())}, not '{file_format}'"
        )
    if not filepath.suffix:
        filepath = filepath.with_suffix(allowed_formats[file_format])

    if file_format in {"csv"}:
        # csv format doesn't support attributes, so attributes must be written to a separate file
        attrs_in_separate_file = True
        attrs_filepath = filepath.with_suffix(filepath.suffix + ".attrs.json")
    else:
        attrs_in_separate_file = False
    if (not overwrite) and filepath.exists():
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filepath.replace(filepath.with_stem(f"{filepath.stem}_old_{timestamp}"))
        # Move attrs file as well, if it exists
        if attrs_in_separate_file and attrs_filepath.exists():
            attrs_filepath.replace(
                filepath.with_stem(f"{filepath.stem}_old_{timestamp}").with_suffix(
                    filepath.suffix + ".attrs.json"
                )
            )

    if file_format == "netcdf":
        attrs = attrs or dict()
        # NetCDF doesn't support boolean attributes
        res.to_xarray().assign_attrs(
            **{
                key: str(attr) if isinstance(attr, bool) else attr
                for key, attr in attrs.items()
            }
        ).to_netcdf(filepath, mode="w")
    elif file_format == "csv":
        res.to_csv(filepath, index=False)
    if attrs and attrs_in_separate_file:
        attrs_filepath.write_text(json.dumps(attrs, indent=4, sort_keys=True))


def _get_standard_attrs(
    query: UnionType[
        "flowclient.api_query.APIQuery", "flowclient.async_api_query.ASyncAPIQuery"
    ]
) -> Dict[str, str]:
    return dict(
        created_at=datetime.now().isoformat(),
        flowclient_version=fc.__version__,
        parameters=json.dumps(query.parameters),
    )


def run_query_and_write_result(
    query: "flowclient.api_query.APIQuery",
    filepath: UnionType[str, Path],
    *,
    file_format: Optional[str] = None,
    poll_interval: int = 10,
    overwrite: bool = False,
    additional_attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run a FlowAPI query, wait for the result, and write it to a file.

    The following attributes will be written along with the query result:

    - 'created_at': isoformat timestamp of result retrieval
    - 'flowclient_version': flowclient version used to run the query
    - 'parameters': query parameters (as a json string)
    - any other user-provided attributes (supplied through the 'additional_attrs' argument)

    If the file format does not support attributes (e.g. csv), attributes will
    be written to a separate file '<filename>.attrs.json'.

    Parameters
    ----------
    query : APIQuery
        API query to run
    filepath : str or Path
        Path at which output file will be created. If the filename has no suffix, an appropriate suffix will
        be appended according to the specified 'file_format'.
    file_format : {'csv', 'netcdf'}, optional
        Output file format. If not specified, we will attempt to infer the file format from the filename extension.
    poll_interval : int, default 10
        Interval (in seconds) between polls to check query status
    overwrite : bool, default False
        Set overwrite=True to overwrite an existing file at the specified path;
        otherwise any existing file will be renamed.
    additional_attrs : dict, optional
        Extra attributes to write to the output file
    """
    res = query.get_result(poll_interval=poll_interval)
    attrs = dict(
        **_get_standard_attrs(query),
        **(additional_attrs or {}),
    )
    _write_query_result(
        res,
        filepath,
        file_format=file_format,
        overwrite=overwrite,
        attrs=attrs,
    )


async def run_query_and_write_result_async(
    query: "flowclient.async_api_query.ASyncAPIQuery",
    filepath: UnionType[str, Path],
    *,
    file_format: Optional[str] = None,
    poll_interval: int = 10,
    overwrite: bool = False,
    additional_attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Asynchronously run a FlowAPI query, wait for the result, and write it to a file.

    The following attributes will be written along with the query result:

    - 'created_at': isoformat timestamp of result retrieval
    - 'flowclient_version': flowclient version used to run the query
    - 'parameters': query parameters (as a json string)
    - any other user-provided attributes (supplied through the 'additional_attrs' argument)

    If the file format does not support attributes (e.g. csv), attributes will
    be written to a separate file '<filename>.attrs.json'.

    Parameters
    ----------
    query : ASyncAPIQuery
        API query to run
    filepath : str or Path
        Path at which output file will be created. If the filename has no suffix, an appropriate suffix will
        be appended according to the specified 'file_format'.
    file_format : {'csv', 'netcdf'}, optional
        Output file format. If not specified, we will attempt to infer the file format from the filename extension.
    poll_interval : int, default 10
        Interval (in seconds) between polls to check query status
    overwrite : bool, default False
        Set overwrite=True to overwrite an existing file at the specified path;
        otherwise any existing file will be renamed.
    """
    res = await query.get_result(poll_interval=poll_interval)
    attrs = dict(
        **_get_standard_attrs(query),
        **(additional_attrs or {}),
    )
    _write_query_result(
        res,
        filepath,
        file_format=file_format,
        overwrite=overwrite,
        attrs=attrs,
    )


def find_dates_to_exclude(
    *,
    flowdb_connection: "flowmachine.core.connection.Connection",
    start_date: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    end_date: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    event_types: Optional[List[str]] = None,
    latest_truncation_threshold: str = "18:00:00",
    fail_on_missing_latest: bool = True,
) -> Set[str]:
    """
    Find a list of data dates within the specified interval that should be
    excluded when running queries.

    A day of data is excluded if data for any event type on that day are:
    - unavailable, or
    - temporally truncated earlier than `latest_truncation_threshold`

    Parameters
    ----------
    flowdb_connection : flowmachine.core.connection.Connection
        FlowDB connection
    start_date : date, timestamp or date string
        First date in the time interval to check
    end_date : date, timestamp or date string
        Day _after_ the last date in the time interval to check
    event_types : list of str, optional
        Optionally specify the list of event types to include (default is to
        include all event types)
    latest_truncation_threshold : str, default '18:00:00'
        Earliest allowed time of latest available event in a day, in format 'hh:mm:ss'
    fail_on_missing_latest : bool, default True
        If True, raise an error if the latest required day of data is
        after the most recent available date (to prevent computation proceeding
        before data have finished arriving)

    Returns
    -------
    set of str
        Set of data dates (in YYYY-mm-dd format) to exclude when running queries
    """
    latest_truncation_timedelta = pd.Timedelta(latest_truncation_threshold)
    all_dates = set(
        d.date() for d in pd.date_range(start_date, end_date, inclusive="left")
    )
    available_dates = flowdb_connection.available_dates
    if event_types is None:
        event_types = list(available_dates.keys())
    available_dates_all_eventtypes = set(available_dates[event_types[0]]).intersection(
        *(set(available_dates[event_type]) for event_type in event_types[1:])
    )
    if fail_on_missing_latest and (
        max(all_dates) > max(available_dates_all_eventtypes)
    ):
        # Default is to error here - could be that the aggregates are being run too soon, and the latest data are not yet ingested
        raise ValueError("Latest required data are not yet available")
    missing_dates = all_dates.difference(available_dates_all_eventtypes)
    event_types_string = ", ".join(f"'{event_type}'" for event_type in event_types)
    with flowdb_connection.engine.begin():
        qa_latest_timestamps = pd.read_sql_query(
            f"""
            SELECT cdr_date, cdr_type, outcome
            FROM etl.deduped_post_etl_queries
            WHERE cdr_date >= '{min(all_dates):%Y-%m-%d}'
                AND cdr_date <= '{max(all_dates):%Y-%m-%d}'
                AND cdr_type in ({event_types_string})
                AND type_of_query_or_check = 'latest_timestamp'
            """,
            con=flowdb_connection.engine,
        )
    dates_without_qa_results = all_dates.intersection(
        available_dates_all_eventtypes
    ).difference(qa_latest_timestamps["cdr_date"])
    if dates_without_qa_results:
        # TODO: Ideally we'd check this per event type
        raise ValueError(
            f"No QA 'latest_timestamp' result for dates {dates_without_qa_results}"
        )
        # TODO: Instead of failing, calculate latest timestamp from CDR if required
        # Roughly-copied from notebook:
        # timestamp_ranges = []
        # for d in tqdm(pd.date_range("2020-01-01", "2020-06-05")):
        #     try:
        #         timestamp_ranges.append(
        #             pd.read_sql_query(
        #                 f"""
        #                 SELECT '{d.strftime('%Y-%m-%d')}'::date AS cdr_date, min(datetime) AS earliest_timestamp, max(datetime) AS latest_timestamp
        #                 FROM events.calls_clustered_{d.strftime('%Y%m%d')}
        #                 """,
        #                 con=fm.core.context.get_db().engine,
        #             )
        #         )
        #     except ProgrammingError:
        #         timestamp_ranges.append(
        #             pd.read_sql_query(
        #                 f"""
        #                 SELECT '{d.strftime('%Y-%m-%d')}'::date AS cdr_date, min(datetime) AS earliest_timestamp, max(datetime) AS latest_timestamp
        #                 FROM events.calls_{d.strftime('%Y%m%d')}
        #                 """,
        #                 con=fm.core.context.get_db().engine,
        #             )
        #         )
        #
        # filled_latest_timestamps = (
        #     qadf["latest_timestamp"]
        #     .astype("datetime64[ns, UTC]")
        #     .fillna(
        #         pd.concat(timestamp_ranges)
        #         .set_index("cdr_date")["latest_timestamp"]
        #         .astype("datetime64[ns, UTC]")
        #     )
        # )
    if len(qa_latest_timestamps) > 0:
        qa_latest_timestamps.set_index(["cdr_date", "cdr_type"])["outcome"].astype(
            "datetime64[ns, UTC]"
        )
        is_truncated = (
            qa_latest_timestamps.set_index(["cdr_date", "cdr_type"])["outcome"]
            .astype("datetime64[ns, UTC]")
            .apply(lambda dt: dt - pd.to_datetime(dt.date(), utc=True))
            < latest_truncation_timedelta
        )
        any_event_type_truncated = (
            is_truncated.reset_index().groupby("cdr_date")["outcome"].any()
        )
        truncated_dates = set(any_event_type_truncated[any_event_type_truncated].index)
    else:
        # If QA results dataframe is empty, comparison will raise a TypeError,
        # so we handle this case differently
        truncated_dates = set()
    return set(str(d) for d in missing_dates.union(truncated_dates))


def find_dates_to_exclude_monthly(
    dt: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    *,
    flowdb_connection: "flowmachine.core.connection.Connection",
    month_offset: int,
    month_start_day: int,
    window_length: int,
    event_types: Optional[List[str]] = None,
    latest_truncation_threshold: str = "18:00:00",
    fail_on_missing_latest: bool = True,
) -> Set[str]:
    """
    Find a list of data dates that should be excluded when running aggregates
    for the specified month.

    A day of data is excluded if data for any event type on that day are:
    - unavailable, or
    - temporally truncated earlier than `latest_truncation_threshold`

    Parameters
    ----------
    dt : date, timestamp or date string
        Date/datetime from which month will be extracted
    flowdb_connection : flowmachine.core.connection.Connection
        FlowDB connection
    month_offset : int
        Number of months to offset from the month specified by 'dt'.
        Negative values offset to earlier months.
    month_start_day : int
        Start day of the month.
        First window in the most recent month will end on this day of month 'dt + month_offset'
    window_length : int
        Length of rolling windows, in days
    event_types : list of str, optional
        Optionally specify the list of event types to include (default is to
        include all event types)
    latest_truncation_threshold : str, default '18:00:00'
        Earliest allowed time of latest available event in a day, in format 'hh:mm:ss'
    fail_on_missing_latest : bool, default True
        If True, raise an error if the latest required day of data is
        after the most recent available date (to prevent computation proceeding
        before data have finished arriving)

    Returns
    -------
    set of str
        Set of data dates (in YYYY-mm-dd format) to exclude when running queries
    """
    start_date = get_date_in_month(
        dt, day_of_month=month_start_day, month_offset=month_offset
    ) - pd.Timedelta(days=window_length - 1)
    end_date = get_date_in_month(
        dt, day_of_month=month_start_day, month_offset=month_offset + 1
    )
    return find_dates_to_exclude(
        flowdb_connection=flowdb_connection,
        start_date=start_date,
        end_date=end_date,
        event_types=event_types,
        latest_truncation_threshold=latest_truncation_threshold,
        fail_on_missing_latest=fail_on_missing_latest,
    )


def check_month_completeness(
    dt: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    *,
    flowdb_connection: "flowmachine.core.connection.Connection",
    month_offset: int,
    month_start_day: int,
    window_length: int,
    event_types: Optional[List[str]] = None,
    latest_truncation_threshold: str = "18:00:00",
    fail_on_missing_latest: bool = True,
    min_percent_of_dates: Optional[UnionType[int, float]] = None,
    max_allowed_gap: Optional[int] = None,
    max_empty_windows: Optional[int] = None,
    min_median_included_days_per_window: Optional[int] = None,
) -> bool:
    """
    Check whether we have a sufficient amount of data to compute majority locations
    for the specified month.

    Four checks are implemented (each can be skipped by setting the associated
    parameter to 'None'):
    - The percentage of dates present is not below a minimum threshold
    - The maximum length of data gap is not above a maximum threshold
    - The number of windows within the month for which we have no data is not
      above a maximum threshold
    - The median number of days present per non-empty window (i.e. median over
      all windows for which there is at least 1 day of data present) is not
      below a minimum threshold

    Parameters
    ----------
    dt : date, timestamp or date string
        Date/datetime from which month will be extracted
    flowdb_connection : flowmachine.core.connection.Connection
        FlowDB connection
    month_offset : int
        Number of months to offset from the month specified by 'dt'.
        Negative values offset to earlier months.
    month_start_day : int
        Start day of the month.
        First window in the most recent month will end on this day of month 'dt + month_offset'
    window_length : int
        Length of rolling windows, in days
    event_types : list of str, optional
        Optionally specify the list of event types to include (default is to
        include all event types)
    latest_truncation_threshold : str, default '18:00:00'
        Earliest allowed time of latest available event in a day, in format 'hh:mm:ss'
    fail_on_missing_latest : bool, default True
        If True, raise an error if the latest required day of data is
        after the most recent available date (to prevent computation proceeding
        before data have finished arriving)
    min_percent_of_dates : numeric, optional
        Minimum percentage of dates that must be included
    max_allowed_gap : int, optional
        Maximum allowed length of data gap (in days)
    max_empty_windows : int, optional
        Maximum number of windows for which all data can be missing
    min_median_included_days_per_window : int, optional
        Minimum threshold for median number of days of data present per non-empty window.
        This should be at least as large as the per-subscriber "median active days per window"
        threshold used when calculating the subscriber subset, otherwise the subscriber subset
        may be empty.

    Returns
    -------
    bool
        True if data are sufficiently complete for computation to proceed for this month
    """
    month_start = get_date_in_month(
        dt, day_of_month=month_start_day, month_offset=month_offset
    )
    print(f"Checking data completeness for month {month_start}...")
    dates_to_exclude = find_dates_to_exclude_monthly(
        dt,
        flowdb_connection=flowdb_connection,
        month_offset=month_offset,
        month_start_day=month_start_day,
        window_length=window_length,
        event_types=event_types,
        latest_truncation_threshold=latest_truncation_threshold,
        fail_on_missing_latest=fail_on_missing_latest,
    )
    data_checks_passed = True
    windows = [
        {
            str(d.date())
            for d in pd.date_range(window_start, window_end, freq="D", inclusive="left")
        }
        for window_start, window_end in rolling_window_over_one_month(
            start_date=month_start, window_length=window_length
        ).values()
    ]
    all_dates = set().union(*windows)
    included_dates = all_dates.difference(dates_to_exclude)
    # Skip month if proportion of dates included is too small
    percent_of_dates_included = 100 * len(included_dates) / len(all_dates)
    if (min_percent_of_dates is not None) and (
        percent_of_dates_included < min_percent_of_dates
    ):
        print(
            f"Too many missing dates: only {len(included_dates)} of {len(all_dates)} dates included ({percent_of_dates_included}%)"
        )
        data_checks_passed = False
    # Skip month if maximum data gap is too large
    longest_gap = max(
        [
            td.days - 1
            for td in np.diff(
                sorted(
                    [pd.Timestamp(d) for d in included_dates]
                    + [
                        pd.Timestamp(min(all_dates)) - pd.Timedelta(days=1),
                        pd.Timestamp(max(all_dates)) + pd.Timedelta(days=1),
                    ]
                )
            )
        ]
    )
    if (max_allowed_gap is not None) and (longest_gap > max_allowed_gap):
        print(f"Gap of {longest_gap} days is too large")
        data_checks_passed = False
    # Skip month if number of entirely-empty windows is too large
    days_per_window = pd.Series(
        [len(window.difference(dates_to_exclude)) for window in windows]
    )
    empty_windows = len(days_per_window[days_per_window == 0])
    if (max_empty_windows is not None) and (empty_windows > max_empty_windows):
        print(f"Too many empty windows: {empty_windows} of {len(windows)} have no data")
        data_checks_passed = False
    # Skip month if median number of non-missing days per non-empty window is too small
    # (because the subscriber subset for this month will be empty)
    median_days_per_window = days_per_window[days_per_window != 0].median()
    if (min_median_included_days_per_window is not None) and (
        median_days_per_window < min_median_included_days_per_window
    ):
        print(
            f"Median number of included days per non-empty window ({median_days_per_window}) is too small"
        )
        data_checks_passed = False

    print(
        f"""completeness metrics:
        * {len(included_dates)} of {len(all_dates)} dates included ({percent_of_dates_included}%),
        * Maximum data gap {longest_gap} days,
        * {empty_windows} of {len(windows)} {window_length}-day windows with no data,
        * {median_days_per_window} days of data (median) per non-empty window."""
    )
    return data_checks_passed


def check_data_availability_for_home_locations(
    dt: UnionType["datetime.date", "datetime.datetime", pd.Timestamp, str],
    *,
    flowdb_connection: "flowmachine.core.connection.Connection",
    month_offset: int,
    month_start_day: int,
    window_length: int,
    event_types: Optional[List[str]] = None,
    latest_truncation_threshold: str = "18:00:00",
    fail_on_missing_latest: bool = True,
    min_percent_of_dates: Optional[UnionType[int, float]] = None,
    max_allowed_gap: Optional[int] = None,
    max_empty_windows: Optional[int] = None,
    min_median_included_days_per_window: Optional[int] = None,
):
    """
    Check whether we have a sufficient amount of data to compute home locations
    for the specified month.

    The possible outcomes are:
    - Insufficient data to compute home locations this month
      (data_available=False)
    - Sufficient data to compute home locations this month, and sufficient data
      to compute majority locations last month (data_available=True,
      lookback_data_available=True, lookback_n_months=1)
    - Sufficient data to compute home locations this month; insufficient data
      to compute majority locations last month, but sufficient data to compute
      majority locations the month before last (data_available=True,
      lookback_data_available=True, lookback_n_months=2)
    - Sufficient data to compute home locations this month; insufficient data
      to compute majority locations for either of the two preceding months
      (data_available=True, lookback_data_available=False,
      lookback_n_months=None). In this situation it is recommended to define
      home locations for this moth using only this month's majority location
      (i.e. no lookback to previous months)

    Parameters
    ----------
    dt : date, timestamp or date string
        Date/datetime from which month will be extracted
    flowdb_connection : flowmachine.core.connection.Connection
        FlowDB connection
    month_offset : int
        Number of months to offset from the month specified by 'dt'.
        Negative values offset to earlier months.
    month_start_day : int
        Start day of the month.
        First window in the most recent month will end on this day of month 'dt + month_offset'
    window_length : int
        Length of rolling windows, in days
    event_types : list of str, optional
        Optionally specify the list of event types to include (default is to
        include all event types)
    latest_truncation_threshold : str, default '18:00:00'
        Earliest allowed time of latest available event in a day, in format 'hh:mm:ss'
    fail_on_missing_latest : bool, default True
        If True, raise an error if the latest required day of data is
        after the most recent available date (to prevent computation proceeding
        before data have finished arriving)
    min_percent_of_dates : numeric, optional
        Minimum percentage of dates that must be included
    max_allowed_gap : int, optional
        Maximum allowed length of data gap (in days)
    max_empty_windows : int, optional
        Maximum number of windows for which all data can be missing
    min_median_included_days_per_window : int, optional
        Minimum threshold for median number of days of data present per non-empty window.
        This should be at least as large as the per-subscriber "median active days per window"
        threshold used when calculating the subscriber subset, otherwise the subscriber subset
        may be empty.

    Returns
    -------
    data_available : bool
        True if data are sufficiently complete for computation to proceed for this month
    lookback_data_available : bool
        True if data in a previous month are sufficiently complete for the home location
        algorithm to refer back to majority locations in that month
    lookback_n_months : int or None
        Number of months the home location algorithm should look back for previous
        majority locations
    """
    # Check data availability for focal month
    data_available = check_month_completeness(
        dt,
        flowdb_connection=flowdb_connection,
        month_offset=month_offset,
        month_start_day=month_start_day,
        window_length=window_length,
        event_types=event_types,
        latest_truncation_threshold=latest_truncation_threshold,
        fail_on_missing_latest=fail_on_missing_latest,
        min_percent_of_dates=min_percent_of_dates,
        max_allowed_gap=max_allowed_gap,
        max_empty_windows=max_empty_windows,
        min_median_included_days_per_window=min_median_included_days_per_window,
    )
    # Check data availability for lookback month
    lookback_n_months = 1
    lookback_data_available = check_month_completeness(
        dt,
        flowdb_connection=flowdb_connection,
        month_offset=month_offset - lookback_n_months,
        month_start_day=month_start_day,
        window_length=window_length,
        event_types=event_types,
        latest_truncation_threshold=latest_truncation_threshold,
        fail_on_missing_latest=fail_on_missing_latest,
        min_percent_of_dates=min_percent_of_dates,
        max_allowed_gap=max_allowed_gap,
        max_empty_windows=max_empty_windows,
        min_median_included_days_per_window=min_median_included_days_per_window,
    )
    if not lookback_data_available:
        # If previous month is unavailable, try looking back one month further
        lookback_n_months = 2
        lookback_data_available = check_month_completeness(
            dt,
            flowdb_connection=flowdb_connection,
            month_offset=month_offset - lookback_n_months,
            month_start_day=month_start_day,
            window_length=window_length,
            event_types=event_types,
            latest_truncation_threshold=latest_truncation_threshold,
            fail_on_missing_latest=fail_on_missing_latest,
            min_percent_of_dates=min_percent_of_dates,
            max_allowed_gap=max_allowed_gap,
            max_empty_windows=max_empty_windows,
            min_median_included_days_per_window=min_median_included_days_per_window,
        )
        if not lookback_data_available:
            # 2 months in a row missing - don't look back to a previous month
            lookback_n_months = None

    return data_available, lookback_data_available, lookback_n_months
