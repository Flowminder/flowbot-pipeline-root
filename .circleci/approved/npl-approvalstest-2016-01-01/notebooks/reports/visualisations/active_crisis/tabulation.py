from pathlib import Path
from pandas import DataFrame
import pandas as pd
from slugify import slugify


def excess_national_arrivals_table(
    excess_arrivals_past_week: DataFrame,
    most_recent_excess_residents_per_host: DataFrame,
    out_folder: Path,
) -> Path:
    table_output = (
        excess_arrivals_past_week.groupby(["admin3name", "admin3pcod"])[["value"]]
        .sum()
        .merge(
            most_recent_excess_residents_per_host.set_index(
                ["admin3name", "admin3pcod"]
            )[["value"]],
            suffixes=("_ex_arrived", "_ex_residents"),
            how="outer",
            left_index=True,
            right_index=True,
        )
        .fillna(0)
    ).reset_index()

    table_output = table_output.sort_values(
        "value_ex_residents", ascending=False
    ).reset_index(drop=True)

    names = table_output["admin3name"] + " (" + table_output["admin3pcod"] + ")"
    names.name = "Communal section (pcod)"

    arrived = table_output["value_ex_arrived"].round(-1).astype(int)
    arrived.name = f"New arrivals"

    residents = table_output["value_ex_residents"].round(-1).astype(int)
    residents.name = "Identified as displaced"

    table_output = pd.concat([names, arrived, residents], axis=1)

    def fix_displaced(row):
        if row["New arrivals"] > row["Identified as displaced"]:
            return row["New arrivals"]
        else:
            return row["Identified as displaced"]

    table_output["Identified as displaced"] = table_output.apply(
        lambda z: fix_displaced(z), axis=1
    )

    table_output.columns = [
        # Rob question; which of these goes into the final report?
        "Communal section (pcod)",
        "New arrivals",
        "Total displaced in section",
    ]

    table_output = table_output.sort_values(
        "Total displaced in section", ascending=False
    ).reset_index(drop=True)

    out_filepath = (
        out_folder / "national" / "excess_arrivals_excess_residents_table.csv"
    )
    table_output.to_csv(out_filepath, index=False)
    return out_filepath


def excess_regional_arrivals_table(
    regional_dataframe: DataFrame,
    most_recent_excess_residents_per_host: DataFrame,
    out_folder: Path,
    aa_name: str,
) -> Path:

    excesses_per_AA = (
        regional_dataframe.groupby(["admin3name", "admin3pcod"])
        .value.sum()
        .reset_index()
    )
    table_output = excesses_per_AA.merge(
        most_recent_excess_residents_per_host[["admin3pcod", "value"]],
        on="admin3pcod",
        suffixes=("_ex_arrived", "_ex_residents"),
    )

    # make it tidy
    table_output = table_output.sort_values(
        "value_ex_residents", ascending=False
    ).reset_index(drop=True)

    names = table_output["admin3name"] + " (" + table_output["admin3pcod"] + ")"
    names.name = "Communal section (pcod)"

    arrived = table_output["value_ex_arrived"].round(-1).astype(int)
    arrived.name = f"Newly displaced"

    residents = table_output["value_ex_residents"].round(-1).astype(int)
    residents.name = "Identified as displaced"

    table_output = pd.concat([names, arrived, residents], axis=1)

    def fix_displaced(row):
        if row["Newly displaced"] > row["Identified as displaced"]:
            return row["Newly displaced"]
        else:
            return row["Identified as displaced"]

    table_output["Identified as displaced"] = table_output.apply(
        lambda z: fix_displaced(z), axis=1
    )

    table_output.columns = [
        "Communal section (pcod)",
        "Newly displaced",
        "Total displaced in section",
    ]

    table_output = table_output.sort_values(
        "Total displaced in section", ascending=False
    ).reset_index(drop=True)

    out_filepath = (
        out_folder
        / "affected_areas"
        / slugify(aa_name)
        / "excess_arrivals_excess_residents_table.csv"
    )
    table_output.to_csv(out_filepath)

    print(table_output)
    return out_filepath
