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


class TimeSeriesTop3Residents(TimeSeries):

    def draw(self, indicators: pd.HDFStore) -> Figure:

        for region in indicators["MonthlyResidentsAdm3"][
            self.indicators.config.shapefile_regional_spatial_name
        ].unique():
            print(region)
            # metadata
            title = f'Largest increases and decreases (by percent change) in the {region} Department between {self.indicators.date.replace(year = self.indicators.date.year - 1).strftime("%B %Y")} and {self.indicators.date.strftime("%B %Y")}'

            X = 2

            # get the relevant time series
            topX_decreasing = (
                indicators["SnapshotTrendsResidentsAdm3"]
                .query(
                    f"({self.indicators.config.shapefile_regional_spatial_name} == @region) and (Unusual == False) and (Fluctuating == False) & (has_abnormal_change == False)"
                )
                .sort_values("Absolute_Change_Trend")
                .head(X)
                .query("Absolute_Change_Trend < -100")
            )
            topX_decreasing_pcods = topX_decreasing[
                self.indicators.config.shapefile_spatial_id
            ]

            topX_increasing = (
                indicators["SnapshotTrendsResidentsAdm3"]
                .query(
                    f"({self.indicators.config.shapefile_regional_spatial_name} == @region) and (Unusual == False) and (Fluctuating == False) & (has_abnormal_change == False)"
                )
                .sort_values("Absolute_Change_Trend")
                .tail(X)
                .query("Absolute_Change_Trend > 100")
            )
            topX_increasing_pcods = topX_increasing[
                self.indicators.config.shapefile_spatial_id
            ]

            # create the figure
            fig, ax = plt.subplots(
                1,
                2,
                figsize=[2.9523 * 1.65 * 1.25 * 1, 1 * 1.65 * 1.25],
                width_ratios=[4, 2],
            )

            markers = ["s", "^"]

            bounds = [-float("inf"), -1000, -500, -100, 100, 500, 1000, float("inf")]
            colormap = plt.get_cmap("fm_div")
            norm = mcolor.BoundaryNorm(bounds, ncolors=colormap.N, clip=True)

            # increasing
            for alpha, (communal_section, cs_df) in enumerate(
                indicators["MonthlyResidentsAdm3"]
                .set_index(self.indicators.config.shapefile_spatial_id)
                .loc[topX_increasing_pcods]
                .groupby(self.indicators.config.shapefile_spatial_id, sort=False)
            ):

                color_value = (
                    indicators["SnapshotTrendsResidentsAdm3"]
                    .query(
                        f"{self.indicators.config.shapefile_spatial_id} == @communal_section"
                    )
                    .Absolute_Change_Trend.values[0]
                )
                c = colormap(norm(color_value))

                pct_change = (
                    cs_df.set_index(self.indicators.config.date_column).residents
                    - cs_df.set_index(
                        self.indicators.config.date_column
                    ).residents.iloc[0]
                )
                pct_change.plot(
                    ax=ax[0],
                    c=c,
                    markevery=(0, 1),
                    marker=markers[alpha],
                    markersize=4,
                    fillstyle="none",
                )

                # trend lines
                dates = (pct_change.index - pct_change.index.min()).days.values
                s = (
                    indicators["SnapshotTrendsResidentsAdm3"]
                    .query(
                        f"{self.indicators.config.shapefile_spatial_id} == @communal_section"
                    )
                    .Trend_Slope.values[0]
                )
                i = (
                    indicators["SnapshotTrendsResidentsAdm3"]
                    .query(
                        f"{self.indicators.config.shapefile_spatial_id} == @communal_section"
                    )
                    .Trend_Intercept.values[0]
                )
                adjustment = cs_df.set_index(
                    self.indicators.config.date_column
                ).residents.iloc[0]
                pd.Series(i + (dates * s) - adjustment, index=pct_change.index).plot(
                    ax=ax[0], c=c, ls=":", lw=1.3
                )

            # decreasing
            for alpha, (communal_section, cs_df) in enumerate(
                indicators["MonthlyResidentsAdm3"]
                .set_index(self.indicators.config.shapefile_spatial_id)
                .loc[topX_decreasing_pcods]
                .groupby(self.indicators.config.shapefile_spatial_id, sort=False)
            ):

                color_value = (
                    indicators["SnapshotTrendsResidentsAdm3"]
                    .query(
                        f"{self.indicators.config.shapefile_spatial_id} == @communal_section"
                    )
                    .Absolute_Change_Trend.values[0]
                )
                c = colormap(norm(color_value))

                pct_change = (
                    cs_df.set_index(self.indicators.config.date_column).residents
                    - cs_df.set_index(
                        self.indicators.config.date_column
                    ).residents.iloc[0]
                )
                pct_change.plot(
                    ax=ax[0],
                    c=c,
                    markevery=(0, 1),
                    marker=markers[alpha],
                    markersize=4,
                    fillstyle="none",
                )

                # trend lines
                dates = (pct_change.index - pct_change.index.min()).days.values
                s = (
                    indicators["SnapshotTrendsResidentsAdm3"]
                    .query(
                        f"{self.indicators.config.shapefile_spatial_id} == @communal_section"
                    )
                    .Trend_Slope.values[0]
                )
                i = (
                    indicators["SnapshotTrendsResidentsAdm3"]
                    .query(
                        f"{self.indicators.config.shapefile_spatial_id} == @communal_section"
                    )
                    .Trend_Intercept.values[0]
                )
                adjustment = cs_df.set_index(
                    self.indicators.config.date_column
                ).residents.iloc[0]
                pd.Series(i + (dates * s) - adjustment, index=pct_change.index).plot(
                    ax=ax[0], c=c, ls=":", lw=1.3
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

            dates = indicators["MonthlyResidentsAdm3"].date.unique()
            x_padding = pd.Timedelta(
                days=30
            )  # Add padding of around one month to the x-axis
            ax[0].set_xlim(dates.min() - x_padding, dates.max() + 1.25 * x_padding)

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

            ax[1].set_xlim(-0.25, 0.7)
            ax[1].set_ylim(-2.5, 1.5)
            ax[1].axis("off")

            # Helper function to wrap text
            def wrap_text(text, width=15):
                return "\n".join(textwrap.wrap(text, width))

            # Plot increasing locations
            for alpha, (communal_section, cs_df) in enumerate(
                indicators["MonthlyResidentsAdm3"]
                .set_index(self.indicators.config.shapefile_spatial_id)
                .loc[topX_increasing_pcods]
                .groupby(self.indicators.config.shapefile_spatial_id, sort=False)
            ):
                color_value = (
                    indicators["SnapshotTrendsResidentsAdm3"]
                    .query(
                        f"{self.indicators.config.shapefile_spatial_id} == @communal_section"
                    )
                    .Absolute_Change_Trend.values[0]
                )
                c = colormap(norm(color_value))
                ax[1].scatter(
                    0.08, alpha, c="white", marker=markers[alpha], s=100, edgecolors=[c]
                )
                ax[1].text(
                    0.15,
                    alpha,
                    wrap_text(cs_df.index[0]),
                    fontsize=7.5,
                    verticalalignment="center",
                    horizontalalignment="left",
                )

            # Plot decreasing locations
            for alpha, (communal_section, cs_df) in enumerate(
                indicators["MonthlyResidentsAdm3"]
                .set_index(self.indicators.config.shapefile_spatial_id)
                .loc[topX_decreasing_pcods]
                .groupby(self.indicators.config.shapefile_spatial_id, sort=False)
            ):
                color_value = (
                    indicators["SnapshotTrendsResidentsAdm3"]
                    .query(
                        f"{self.indicators.config.shapefile_spatial_id} == @communal_section"
                    )
                    .Absolute_Change_Trend.values[0]
                )
                c = colormap(norm(color_value))
                ax[1].scatter(
                    0.08,
                    -alpha - 1,
                    c="white",
                    marker=markers[alpha],
                    s=100,
                    edgecolors=[c],
                )
                ax[1].text(
                    0.15,
                    -alpha - 1,
                    wrap_text(cs_df.index[0]),
                    fontsize=7.5,
                    verticalalignment="center",
                    horizontalalignment="left",
                )

            plt.tight_layout()

            yield Figure(
                figure=fig,
                ax=ax,
                caption="",
                title=title,
                filepath=slugify(region) + "/" + self.__class__.__name__,
            )
