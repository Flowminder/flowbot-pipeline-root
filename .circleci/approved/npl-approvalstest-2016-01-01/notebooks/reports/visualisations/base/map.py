import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
import numpy as np
import xyzservices.providers as xyz
from matplotlib_scalebar.scalebar import ScaleBar
from utils.indicator_loader import IndicatorLoader
import os


class Map:
    def __init__(self, indicators: IndicatorLoader):
        self.indicators = indicators

    def draw(self, indicators):
        raise NotImplementedError("")

    def make(self):
        with self.indicators.open() as indicators:
            for fig in self.draw(indicators):
                yield fig

    def map_boundaries(
        self, fig_ax: plt.Axes, geodata: gpd.GeoDataFrame, **kwargs
    ) -> plt.Axes:
        boundaries = kwargs.pop("boundaries", None)
        if boundaries is None:
            minx, miny, maxx, maxy = geodata.total_bounds
        else:
            minx, miny, maxx, maxy = boundaries

        aspect = kwargs.pop("aspect", 1)
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        dx, dy = maxx - minx, maxy - miny
        adj_small = np.abs((dx - dy) / 4)

        if dx < dy:
            fig_ax.set_xlim(
                mid_x - 0.5 * aspect * (dx + np.abs(dy - dx)) - adj_small,
                mid_x + 0.5 * aspect * (dx + np.abs(dy - dx)) + adj_small,
            )
            fig_ax.set_ylim(mid_y - 0.5 * dy - adj_small, mid_y + 0.5 * dy + adj_small)
        elif dx > dy:
            fig_ax.set_xlim(
                mid_x - 0.5 * aspect * dx - adj_small,
                mid_x + 0.5 * aspect * dx + aspect * adj_small,
            )
            fig_ax.set_ylim(
                mid_y - 0.5 * (dy + np.abs(dy - dx)) - adj_small,
                mid_y + 0.5 * (dy + np.abs(dy - dx)) + aspect * adj_small,
            )
        else:
            fig_ax.set_xlim(
                mid_x - 0.5 * aspect * dx - adj_small * aspect,
                mid_x + 0.5 * aspect * dx + adj_small * aspect,
            )
            fig_ax.set_ylim(mid_y - 0.5 * dy - adj_small, mid_y + 0.5 * dy + adj_small)

        return fig_ax

    def add_place_names(self, fig_ax: plt.Axes) -> plt.Axes:
        cx.add_basemap(
            ax=fig_ax,
            source=xyz.CartoDB.VoyagerOnlyLabels.build_url(scale_factor="@2x"),
            attribution=False,
            alpha=0.6,
            zorder=4,
        )

    def add_plot_basemap(self, fig_ax: plt.Axes) -> plt.Axes:
        cx.add_basemap(
            ax=fig_ax,
            source=os.environ["MAPBOX_WMTS_URL"],
            attribution=xyz.CartoDB.VoyagerNoLabels.attribution,
            zorder=0,
        )
        return fig_ax

    def add_scalebar(self, fig_ax: plt.Axes) -> plt.Axes:
        fig_ax.add_artist(
            ScaleBar(
                dx=1,
                location="lower right",
                scale_loc="top",
                width_fraction=0.004,
                border_pad=1,
                box_alpha=0,
                font_properties={"size": 12},
            )
        )
        return fig_ax
