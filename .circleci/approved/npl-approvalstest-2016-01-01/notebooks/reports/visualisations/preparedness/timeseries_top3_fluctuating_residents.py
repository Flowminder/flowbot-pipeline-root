import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from slugify import slugify

import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.colors as mcolor

from visualisations.base.figure import Figure
from visualisations.base.timeseries import TimeSeries

import textwrap


class TimeSeriesTop3FluctuatingResidents(TimeSeries):

    def draw(self, indicators: pd.HDFStore) -> Figure:
        trends = indicators["SnapshotTrendsResidentsAdm3"]
        residents = indicators["MonthlyResidentsAdm3"]

        # Helper function to wrap text
        def wrap_text(text, width=15):
            return "\n".join(textwrap.wrap(text, width))

        markers = ["s", "^"]
        c = "gray"

        for adm1, adm1_df in trends.groupby(
            self.indicators.config.shapefile_regional_spatial_name
        ):

            fig, ax = plt.subplots(
                1,
                2,
                figsize=[2.9523 * 1.65 * 1.25 * 1, 1 * 1.65 * 1.25],
                width_ratios=[4, 2],
            )

            top_fluct = (
                adm1_df.query(
                    "((has_abnormal_change == True) or (Fluctuating == True) or (Unusual == True)) and (has_data)"
                )
                .sort_values("largest_abnormal_fluct")
                .dropna()
                .tail(2)[self.indicators.config.shapefile_spatial_id]
            )

            if len(top_fluct) > 0:

                for e, (adm3, adm3_df) in enumerate(
                    residents[residents.admin3pcod.isin(top_fluct)].groupby(
                        self.indicators.config.shapefile_spatial_id
                    )
                ):

                    adm3_name = adm3_df[
                        self.indicators.config.shapefile_spatial_name
                    ].iloc[0]
                    adm3_df = adm3_df.set_index("date").residents.sort_index()
                    (adm3_df - adm3_df.iloc[0]).plot(
                        ax=ax[0], marker=markers[e], fillstyle="none", color=c
                    )
                    ax[1].scatter(
                        0.08,
                        e * 1.5 - 0.5,
                        c="white",
                        marker=markers[e],
                        s=100,
                        edgecolors=[c],
                    )
                    ax[1].text(
                        0.15,
                        e * 1.5 - 0.5,
                        wrap_text(adm3_name),
                        fontsize=7.5,
                        verticalalignment="center",
                        horizontalalignment="left",
                    )

            else:

                adm3_df = pd.Series(np.nan, index=sorted(residents.date.unique()))
                adm3_df.plot(ax=ax[0])
                ax[0].text(
                    0.5,
                    0.5,
                    "No fluctuating communal sections",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax[0].transAxes,
                    fontsize=8,
                    color="grey",
                )

            ax[0].set_xlabel("")
            ax[0].set_ylabel("Change in residents")

            ax[0].spines["right"].set_visible(False)
            ax[0].spines["top"].set_visible(False)
            ax[0].spines["left"].set_visible(False)

            ax[0].yaxis.grid(
                True, linestyle="-", linewidth=0.5, color="silver", zorder=0
            )
            ax[0].set_axisbelow(True)

            dates = residents.date.unique()
            x_padding = pd.Timedelta(
                days=30
            )  # Add padding of around one month to the x-axis
            ax[0].set_xlim(dates.min() - x_padding, dates.max() + 1.25 * x_padding)

            ax[1].set_xlim(-0.25, 0.7)
            ax[1].set_ylim(-2.5, 2.5)
            ax[1].axis("off")

            plt.tight_layout()

            source_text = "Source: Flowminder"
            ax[0].annotate(
                source_text,
                xy=(0.005, 0.005),
                xycoords="axes fraction",
                fontsize=8,
                ha="left",
                va="bottom",
                font="Roboto",
                color="silver",
                zorder=1,
            )

            title = ""

            yield Figure(
                figure=fig,
                ax=ax,
                caption="",
                title=title,
                filepath=slugify(adm1) + "/" + self.__class__.__name__,
            )
