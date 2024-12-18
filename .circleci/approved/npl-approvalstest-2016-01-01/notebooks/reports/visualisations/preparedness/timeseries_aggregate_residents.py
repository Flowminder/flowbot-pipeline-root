import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np
import pandas as pd
from slugify import slugify

from utils.country_config import Language
from visualisations.base.figure import Figure
from visualisations.base.timeseries import TimeSeries


class TimeSeriesAggregateResidents(TimeSeries):

    def draw(self, indicators: pd.HDFStore):

        for region, region_df in indicators["MonthlyResidentsAdm1"].groupby(
            level=self.indicators.config.shapefile_regional_spatial_name
        ):
            if self.indicators.config.language == Language.English:
                title = f'Monthly Population Estimates in the {region} Department, between {self.indicators.date.replace(year = self.indicators.date.year - 1).strftime("%B %Y")} and {self.indicators.date.strftime("%B %Y")}'
            elif self.indicators.config.language == Language.French:
                title = f'Monthly Population Estimates in the {region} Department, between {self.indicators.date.replace(year = self.indicators.date.year - 1).strftime("%B %Y")} and {self.indicators.date.strftime("%B %Y")}'
            else:
                title = ""

            region_df = region_df.reset_index().set_index("date")[["residents"]]
            dates = region_df.index.unique()
            region_df.residents = np.round(region_df.residents, -2)

            # show time series
            ax = region_df.plot(
                figsize=[2.9523 * 1.65 * 1.25, 1 * 1.65 * 1.25],
                legend=False,
                markevery=(0, 1),
                marker=".",
                color="fm_dark_blue",
            )

            # show numbers on time series
            for i in range(len(region_df) - 1, -1, -1):
                value = region_df.iloc[i]
                date = value.name  # Extract the date from the index
                ax.annotate(
                    f"{int(value.residents):,}",
                    (date, value.residents),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=5,
                )

            ax.set_xlabel("")
            ax.set_ylabel("Residents")

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # ax.xaxis.set_major_locator(MonthLocator())
            # ax.xaxis.set_major_formatter(DateFormatter('%b'))

            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: "{:,}".format(int(x)))
            )

            ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="silver", zorder=0)
            ax.set_axisbelow(True)

            x_padding = pd.Timedelta(
                days=30
            )  # Add padding of around one month to the x-axis
            y_padding = (region_df.max() - region_df.min()) * 0.08
            ax.set_xlim(dates.min() - x_padding, dates.max() + 1.25 * x_padding)
            ax.set_ylim(
                (region_df.min() - y_padding).values[0],
                (region_df.max() + y_padding).values[0],
            )

            source_text = "Source: Flowminder"
            ax.annotate(
                source_text,
                xy=(0.99, 0.01),
                xycoords="axes fraction",
                fontsize=8,
                ha="right",
                va="bottom",
                font="Roboto",
                color="silver",
            )

            plt.tight_layout()

            yield Figure(
                figure=plt.gcf(),
                ax=ax,
                caption="",
                title=title,
                filepath=slugify(region) + "/" + self.__class__.__name__,
            )
