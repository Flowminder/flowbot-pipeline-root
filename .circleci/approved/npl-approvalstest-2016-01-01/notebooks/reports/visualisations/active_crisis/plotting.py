from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
from slugify import slugify

px = 1 / plt.rcParams["figure.dpi"]


def excess_arrivals_excess_residents_barchart(
    excess_arrivals_past_week: DataFrame,
    most_recent_excess_residents_per_host: DataFrame,
    out_folder: Path,
) -> Path:

    excess_arrivals_per_adm2_sum_past_week = (
        excess_arrivals_past_week.sort_values("value", ascending=False)
        .groupby("admin2name")
        .value.sum()
    )
    excess_arrivals_per_adm2_sum_past_week.name = "Newly displaced"

    excess_arrivals_per_adm2_sum_since_event = (
        most_recent_excess_residents_per_host.sort_values("value", ascending=False)
        .groupby("admin2name")
        .value.sum()
    )
    excess_arrivals_per_adm2_sum_since_event.name = "Total displaced"

    # excess_arrived_residents_bar = excess_arrivals_per_adm2_sum_past_week.to_frame().merge(excess_arrivals_per_adm2_sum_since_event.to_frame(), left_index=True, right_index=True, suffixes = ('_ex_res', '_ex_arr'), how = 'right')

    excess_arrived_residents_bar = (
        excess_arrivals_per_adm2_sum_past_week.to_frame()
        .merge(
            excess_arrivals_per_adm2_sum_since_event.to_frame(),
            left_index=True,
            right_index=True,
            suffixes=("_ex_res", "_ex_arr"),
            how="outer",
        )
        .fillna(0)
    )
    excess_arrived_residents_bar = excess_arrived_residents_bar[
        (excess_arrived_residents_bar > 0).any(axis=1)
    ]

    def fix_displaced(row):
        if row["Newly displaced"] > row["Total displaced"]:
            return row["Newly displaced"]
        else:
            return row["Total displaced"]

    excess_arrived_residents_bar["Total displaced"] = np.round(
        excess_arrived_residents_bar.apply(lambda z: fix_displaced(z), axis=1), -1
    )

    # excess_arrived_residents_bar['Total displaced'] = excess_arrived_residents_bar['Newly displaced']

    ax = excess_arrived_residents_bar.plot.bar(figsize=[336 * px * 3.5, 146 * px * 3.5])

    for i, container in enumerate(ax.containers):
        ax.bar_label(
            container,
            fontsize=7,
            color="k",
            labels=[f"{x:,.0f}" for x in container.datavalues],
            padding=3,
        )

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)

    plt.grid("on", axis="y", zorder=1)
    plt.xlabel("")
    plt.tight_layout()

    out_filepath = (
        out_folder / "national" / "excess_arrivals_excess_residents_barchart.png"
    )
    plt.savefig(out_filepath)
    return out_filepath


def MAD(df, col):
    return abs(df[col] - df[col].median()).median()


def get_current_color(cycler_iterator):
    try:
        current_color = next(cycler_iterator)
        return current_color
    except StopIteration:
        # Reset the iterator if it reaches the end
        cycler_iterator = iter(cycler)
        current_color = next(cycler_iterator)
        return current_color


def top_3_adm_hosts(
    excess_arrivals_df: DataFrame,
    arrivals_from_aa: DataFrame,
    key_dates: dict,  # TODO: This should be a dataclass
    date_string: str,
    aa_name: str,
    output_folder: Path,
):
    def _plot_arrivals(df, ax):
        df = df.set_index("date")

        threshold = df[
            (df.index >= key_dates["event_date_minus_3_months"])
            & (df.index < key_dates["event_date"])
        ].value.median() + (3 * 1.486) * MAD(
            df[
                (df.index >= key_dates["event_date_minus_3_months"])
                & (df.index < key_dates["event_date"])
            ],
            "value",
        )

        # I wonndered about just using itertools.cycler here, but I think this
        # interacts with matplotlib.cycler somehow so I'm not touching it.
        c = get_current_color(color_cycle_iter)["color"]

        df.value.plot(
            ax=ax,
            marker="o",
            label=f"{df.admin3name.iloc[0]} ({df.admin3pcod.iloc[0]})",
            mfc="white",
        )

        excess_arrival_points = df.query("value > @threshold").value
        excess_arrival_points.name = ""
        excess_arrival_points.plot(
            ax=ax, marker="o", legend=False, lw=0, color=c, label=None
        )

    fig, ax = plt.subplots(figsize=[336 * px * 4.5, 146 * px * 4.5])

    topX = (
        excess_arrivals_df.groupby("admin3name")
        .value.sum(numeric_only=True)
        .sort_values(ascending=False)
        .head(3)
        .index
    )

    # TODO: checkme @ix should be @aa_name
    arrivals_from_aa.query("index_AA == @aa_name").query("admin3name in @topX").groupby(
        "admin3name"
    ).apply(_plot_arrivals, ax)

    plt.axvspan(
        key_dates["event_date_minus_3_months"],
        key_dates["event_date"] - pd.DateOffset(days=1),
        color="silver",
        alpha=0.3,
        label=f"Baseline period ({date_string})",
    )

    leg1 = plt.legend(fontsize="small", loc="upper left")

    leg2 = [
        Line2D([0], [0], color="k", marker="o", mfc="white", label="Normal arrivals"),
        Line2D([0], [0], color="k", marker="o", label="Arrivals above usual"),
    ]
    ax.add_artist(leg1)
    ax.add_artist(
        ax.legend(
            handles=leg2,
            loc="upper left",
            fontsize="small",
            bbox_to_anchor=(0.25, 0.95),
            alignment="center",
        )
    )

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    # ax.set_position([0, 0, 1, 1])
    plt.grid("on", axis="y", zorder=1)
    plt.xlim(
        [
            key_dates["event_date_minus_2_months"] - pd.DateOffset(days=1),
            key_dates["report_date"] + pd.DateOffset(days=1),
        ]
    )
    plt.xlabel("")
    plt.ylabel(f"Arrivals from {aa_name}")
    plt.tight_layout()

    out_filepath = (
        output_folder / "affected_areas" / slugify(aa_name) / "top3_adm_hosts.png"
    )
    plt.savefig(out_filepath)
    return out_filepath


def top_3_adm_hosts_multi_aa(
    excess_arrivals_df: DataFrame,
    arrivals_from_aa: DataFrame,
    key_dates: dict,  # TODO: This should be a dataclass
    date_string: str,
    aa_name: list[str],
    output_folder: Path,
):
    def _plot_arrivals(df, ax):
        df = df.groupby("date").sum()

        threshold = df[
            (df.index >= key_dates["event_date_minus_3_months"])
            & (df.index < key_dates["event_date"])
        ].value.median() + (3 * 1.486) * MAD(
            df[
                (df.index >= key_dates["event_date_minus_3_months"])
                & (df.index < key_dates["event_date"])
            ],
            "value",
        )

        # I wonndered about just using itertools.cycler here, but I think this
        # interacts with matplotlib.cycler somehow so I'm not touching it.
        c = get_current_color(color_cycle_iter)["color"]

        df.value.plot(
            ax=ax,
            marker="o",
            label=f"{df.admin3name.iloc[0]} ({df.admin3pcod.iloc[0]})",
            mfc="white",
        )

        excess_arrival_points = df.query("value > @threshold").value
        excess_arrival_points.name = ""
        excess_arrival_points.plot(
            ax=ax, marker="o", legend=False, lw=0, color=c, label=None
        )

    fig, ax = plt.subplots(figsize=[336 * px * 4.5, 146 * px * 4.5])

    topX = (
        excess_arrivals_df.groupby("admin3name")
        .value.sum(numeric_only=True)
        .sort_values(ascending=False)
        .head(3)
        .index
    )

    # TODO: checkme @ix should be @aa_name
    arrivals_from_aa.query("index_AA in @aa_name").query("admin3name in @topX").groupby(
        "admin3name"
    ).apply(_plot_arrivals, ax)

    plt.axvspan(
        key_dates["event_date_minus_3_months"],
        key_dates["event_date"] - pd.DateOffset(days=1),
        color="silver",
        alpha=0.3,
        label=f"Baseline period ({date_string})",
    )

    leg1 = plt.legend(fontsize="small", loc="upper left")

    leg2 = [
        Line2D([0], [0], color="k", marker="o", mfc="white", label="Normal arrivals"),
        Line2D([0], [0], color="k", marker="o", label="Arrivals above usual"),
    ]
    ax.add_artist(leg1)
    ax.add_artist(
        ax.legend(
            handles=leg2,
            loc="upper left",
            fontsize="small",
            bbox_to_anchor=(0.25, 0.95),
            alignment="center",
        )
    )

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    # ax.set_position([0, 0, 1, 1])
    plt.grid("on", axis="y", zorder=1)
    plt.xlim(
        [
            key_dates["event_date_minus_2_months"] - pd.DateOffset(days=1),
            key_dates["report_date"] + pd.DateOffset(days=1),
        ]
    )
    plt.xlabel("")
    plt.ylabel(f"Arrivals from {aa_name}")
    plt.tight_layout()

    out_filepath = output_folder / "national" / "top3_adm_hosts_multi_aa.png"
    plt.savefig(out_filepath)
    return out_filepath


def residents_hosts_vs_aa(
    residents_hosts_df: DataFrame,  # TODO: What should we name this?
    hosts: DataFrame,
    residents_hosts: DataFrame,
    key_dates: dict,
    date_string: str,
    aa_name: str,
    output_folder: Path,
):

    fig, ax = plt.subplots(figsize=[336 * px * 4.5, 146 * px * 4.5])

    hosts_for_aa = hosts.query("name == @aa_name").admin3pcod.unique()

    for counter, (j, jdf) in enumerate(
        residents_hosts.query("admin3pcod in @hosts_for_aa").groupby("admin3pcod")
    ):
        legend = True if counter == 0 else False
        jdf.set_index("date").value.plot(
            color="#27B288", ax=ax, label="Host locations", legend=legend
        )

    residents_hosts_df.set_index("date").value.plot(
        color="#701F53", ax=ax, label="Affected area", legend=True
    )

    plt.axvspan(
        key_dates["event_date_minus_3_months"],
        key_dates["event_date"] - pd.DateOffset(days=1),
        color="silver",
        alpha=0.3,
        label=f"Baseline period ({date_string})",
    )

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_axisbelow(True)
    # ax.set_position([0, 0, 1, 1])
    plt.grid("on", axis="y", zorder=1)
    plt.xlabel("")
    plt.ylabel(f"Change in resident subscribers in host locations / affected area")
    plt.xlim(
        [
            key_dates["event_date_minus_2_months"] - pd.DateOffset(days=1),
            key_dates["report_date"] + pd.DateOffset(days=1),
        ]
    )
    plt.tight_layout()

    out_filepath = (
        output_folder
        / "affected_areas"
        / slugify(aa_name)
        / "residents_hosts_vs_aa.png"
    )
    plt.savefig(out_filepath)
    return out_filepath
