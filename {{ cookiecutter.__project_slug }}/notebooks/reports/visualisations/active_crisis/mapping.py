from typing import Optional
from matplotlib.patches import Patch
from pathlib import Path
from pandas import DataFrame
from slugify import slugify
from geopandas import GeoDataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextily as cx
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from cycler import cycler
import xyzservices.providers as xyz
import geopandas as gpd
from flowmindercolors import register_custom_colormaps
import os

px = 1 / plt.rcParams["figure.dpi"]


# TODO: Rewrite to use styler class at some point that we hand to the functions explicitly
def set_default_style_to_flowminder():
    color_cycle = cycler(
        color=[
            "#034174",
            "#CBA45A",
            "#701F53",
            "#006E8C",
            "#BF6799",
            "#00989A",
            "#9E6257",
        ]
    )
    try:
        register_custom_colormaps()
    except ValueError:
        print("Custom colormaps already registered")
    plt.style.use("seaborn-v0_8-ticks")
    mpl.rcParams["legend.frameon"] = True
    mpl.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.axisbelow"] = False
    mpl.rcParams["axes.prop_cycle"] = color_cycle


# Replicated in the Map class
def map_boundaries(
    fig_ax: mpl.axes.Axes,
    geodata: gpd.GeoDataFrame,
    boundaries: Optional[np.ndarray] = None,
    aspect: Optional[int] = 1,
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


def plot_national_map(
    out_folder: Path,
    spatial_geometry: GeoDataFrame,
    total_excess_arrivals_past_week: DataFrame,
    host_areas: GeoDataFrame,
    affected_areas: GeoDataFrame,
    date_string: str,
) -> Path:
    ax = spatial_geometry.to_crs(epsg=3857).plot(
        color=(0, 0, 0, 0), figsize=(900 * px, 900 * px)
    )

    # !
    total_excess_arrivals_past_week.to_crs(epsg=3857).plot(
        ax=ax,
        column="value",
        cmap="fm_seq",
        legend=True,
        scheme="fisherjenks",
        k=min(5, total_excess_arrivals_past_week["value"].nunique()),
        legend_kwds={
            "loc": "upper right",
            "fontsize": 10,
            "title": f"Total excess arrivals,\nbetween {date_string}.",
            "alignment": "left",
        },
    )

    spatial_geometry.to_crs(epsg=3857).boundary.plot(
        lw=0.1, color="fm_land_edge", ax=ax
    )
    spatial_geometry.to_crs(epsg=3857).groupby("admin2name").geometry.apply(
        lambda x: x.unary_union
    ).boundary.plot(ax=ax, lw=0.3, color="fm_land_edge")
    spatial_geometry.to_crs(epsg=3857).groupby("admin1name").geometry.apply(
        lambda x: x.unary_union
    ).boundary.plot(ax=ax, lw=1, color="fm_land_edge")

    # !
    host_areas.to_crs(epsg=3857).boundary.plot(
        ax=ax, lw=1.2, zorder=7, color="fm_purple", ls=":"
    )
    affected_areas.boundary.to_crs(epsg=3857).plot(
        ax=ax, lw=0.8, zorder=5, color="k", hatch="///", alpha=1
    )

    geopandas_legend = ax.get_legend()

    # Custom legend items
    legend_elements = [
        Patch(
            facecolor="fm_land",
            edgecolor="fm_land_edge",
            lw=1,
            label="Communal section boundary",
        ),
        Patch(
            facecolor="fm_land",
            edgecolor="fm_land_edge",
            lw=0.5,
            label="Neighbourhood boundary",
        ),
        Patch(
            facecolor=(0, 0, 0, 0),
            edgecolor="#2D2D2D",
            lw=2,
            label="Affected area",
            hatch="///",
        ),
        Patch(
            facecolor=(0, 0, 0, 0),
            edgecolor="fm_purple",
            lw=1,
            label="Neighbourhoods with displaced subscribers since event start",
            ls=":",
        ),
    ]

    ax.legend(handles=legend_elements, loc="upper left")

    ax.add_artist(geopandas_legend)

    add_scalebar(ax)
    map_boundaries(ax, spatial_geometry.to_crs(epsg=3857))
    add_plot_basemap(ax)
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])

    out_filepath = out_folder / "national" / "all_excess_arrivals.png"
    plt.savefig(out_filepath)
    return out_filepath


def plot_regional_map(
    regional_dataframe: GeoDataFrame,
    affected_areas: GeoDataFrame,
    spatial_geometry: GeoDataFrame,
    host_areas: GeoDataFrame,
    output_folder: Path,
    aa: str,
    date_str: str,
) -> Path:

    ax = spatial_geometry.to_crs(epsg=3857).plot(
        color=(0, 0, 0, 0), figsize=(900 * px, 900 * px)
    )
    regional_dataframe.to_crs(epsg=3857).plot(
        ax=ax,
        column="value",
        cmap="fm_seq",
        legend=True,
        scheme="fisherjenks",
        k=min(5, regional_dataframe["value"].nunique()),
        legend_kwds={
            "loc": "upper right",
            "fontsize": 10,
            "title": f"Total excess arrivals from {aa},\nbetween {date_str}.",
            "alignment": "left",
        },
    )

    spatial_geometry.to_crs(epsg=3857).boundary.plot(
        lw=0.1, color="fm_land_edge", ax=ax
    )
    spatial_geometry.to_crs(epsg=3857).groupby("admin2name").geometry.apply(
        lambda x: x.unary_union
    ).boundary.plot(ax=ax, lw=0.3, color="fm_land_edge")
    spatial_geometry.to_crs(epsg=3857).groupby("admin1name").geometry.apply(
        lambda x: x.unary_union
    ).boundary.plot(ax=ax, lw=1, color="fm_land_edge")

    # !
    host_areas.to_crs(epsg=3857).boundary.plot(
        ax=ax, lw=1.2, zorder=7, color="fm_purple", ls=":"
    )
    affected_areas.query("name == @aa").boundary.to_crs(epsg=3857).plot(
        ax=ax, lw=0.8, zorder=5, color="k", hatch="///", alpha=1
    )

    geopandas_legend = ax.get_legend()

    # Custom legend items
    legend_elements = [
        Patch(
            facecolor="fm_land",
            edgecolor="fm_land_edge",
            lw=1,
            label="Communal section boundary",
        ),
        Patch(
            facecolor="fm_land",
            edgecolor="fm_land_edge",
            lw=0.5,
            label="Neighbourhood boundary",
        ),
        Patch(
            facecolor=(0, 0, 0, 0),
            edgecolor="#2D2D2D",
            lw=2,
            label="Affected area",
            hatch="///",
        ),
        Patch(
            facecolor=(0, 0, 0, 0),
            edgecolor="fm_purple",
            lw=1,
            label="Neighbourhoods with displaced subscribers since event start",
            ls=":",
        ),
    ]

    ax.legend(handles=legend_elements, loc="upper left")

    ax.add_artist(geopandas_legend)

    add_scalebar(ax)
    map_boundaries(ax, spatial_geometry.to_crs(epsg=3857))
    add_plot_basemap(ax)
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])

    out_filepath = (
        output_folder / "affected_areas" / slugify(aa) / "all_excess_arrivals.png"
    )
    plt.savefig(out_filepath)
    return out_filepath


def plot_zoomed_regional_map(
    regional_dataframe: GeoDataFrame,
    affected_areas: GeoDataFrame,
    spatial_geometry: GeoDataFrame,
    host_areas: GeoDataFrame,
    date_string: str,
    aa_name: str,
    output_folder: Path,
):
    ax = spatial_geometry.to_crs(epsg=3857).plot(
        color=(0, 0, 0, 0), figsize=(900 * px, 900 * px)
    )
    # !
    regional_dataframe.to_crs(epsg=3857).plot(
        ax=ax,
        column="value",
        cmap="fm_seq",
        legend=True,
        scheme="fisherjenks",
        k=min(5, len(regional_dataframe)),
        legend_kwds={
            "loc": "upper right",
            "fontsize": 10,
            "title": f"Total excess arrivals from {aa_name},\nbetween {date_string}.",
            "alignment": "left",
        },
    )

    spatial_geometry.to_crs(epsg=3857).boundary.plot(
        lw=0.1, color="fm_land_edge", ax=ax
    )
    spatial_geometry.to_crs(epsg=3857).groupby("admin2name").geometry.apply(
        lambda x: x.unary_union
    ).boundary.plot(ax=ax, lw=0.3, color="fm_land_edge")
    spatial_geometry.to_crs(epsg=3857).groupby("admin1name").geometry.apply(
        lambda x: x.unary_union
    ).boundary.plot(ax=ax, lw=1, color="fm_land_edge")

    # !
    host_areas.to_crs(epsg=3857).boundary.plot(
        ax=ax, lw=1.2, zorder=7, color="fm_purple", ls=":"
    )
    affected_areas.query("name == @aa_name").boundary.to_crs(epsg=3857).plot(
        ax=ax, lw=0.8, zorder=5, color="k", hatch="//", alpha=1
    )

    geopandas_legend = ax.get_legend()

    # Custom legend items
    legend_elements = [
        Patch(
            facecolor="fm_land",
            edgecolor="fm_land_edge",
            lw=1,
            label="Communal section boundary",
        ),
        Patch(
            facecolor="fm_land",
            edgecolor="fm_land_edge",
            lw=0.5,
            label="Neighbourhood boundary",
        ),
        Patch(
            facecolor=(0, 0, 0, 0),
            edgecolor="#2D2D2D",
            lw=2,
            label="Affected area",
            hatch="//",
        ),
        Patch(
            facecolor=(0, 0, 0, 0),
            edgecolor="fm_purple",
            lw=1,
            label="Neighbourhoods with displaced subscribers since event start",
            ls=":",
        ),
    ]

    ax.legend(handles=legend_elements, loc="upper left")

    ax.add_artist(geopandas_legend)

    add_scalebar(ax)
    map_boundaries(
        ax,
        affected_areas.query("name == @aa_name")
        .to_crs(epsg=3857)
        .centroid.buffer(40000),
    )
    add_plot_basemap(ax)
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])

    out_filepath = (
        output_folder
        / "affected_areas"
        / slugify(aa_name)
        / "all_excess_arrivals_zoomed.png"
    )
    plt.savefig(out_filepath)
    return out_filepath
