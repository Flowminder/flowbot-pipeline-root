import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolor
import geopandas as gpd
import numpy as np
from slugify import slugify
from visualisations.base.map import Map
from visualisations.base.figure import Figure
from utils.fmcolors import register_custom_colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

# Register custom colormaps
register_custom_colormaps()


class MapResidentsTrendsNational(Map):

    def plot_infobox(
        self,
        geodataframe,
        main_rect_pct,
        main_rect_height_pct,
        small_rect_width_pct,
        small_rect_height_pct,
        ax,
    ):
        # Extract axis limits and calculate figure dimensions
        min_x, max_x = ax.get_xlim()
        min_y, max_y = ax.get_ylim()
        fig_width = max_x - min_x
        fig_height = max_y - min_y

        # Calculate main and small rectangle dimensions
        main_rect_width = fig_width * main_rect_pct
        main_rect_height = fig_height * main_rect_height_pct
        small_rect_width = main_rect_width * small_rect_width_pct
        small_rect_height = main_rect_height * small_rect_height_pct

        for _, row in geodataframe.iterrows():
            # Calculate main rectangle position
            centroid = row.geometry
            main_rect_x = centroid.x - main_rect_width / 2
            main_rect_y = centroid.y - main_rect_height / 2

            # Draw main rectangle
            main_rect = patches.Rectangle(
                (main_rect_x, main_rect_y),
                main_rect_width,
                main_rect_height,
                facecolor=(1, 1, 1, 0.65),
                edgecolor="none",
            )
            main_rect.set_zorder(6)
            ax.add_patch(main_rect)

            # Format text values for infobox
            residents_diff = f"{round(row['SnapshotDiffResidentsAdm1'], -2):,.0f}"
            residents_diff_sub = "over the last year"
            total_residents = (
                f"{round(row['SnapshotMostRecentResidentsAdm1'], -2):,.0f}"
            )
            total_residents_sub = "total residents"
            add_sign = "+" if np.sign(row["SnapshotDiffResidentsAdm1"]) == 1 else ""
            text_color = "#2d2d2d"

            # Add text to infobox
            ax.text(
                main_rect_x + main_rect_width - main_rect_width / 30,
                main_rect_y + main_rect_height - main_rect_height / 4,
                add_sign + residents_diff,
                fontsize=16,
                color=text_color,
                fontname="Roboto",
                ha="right",
                va="center",
                zorder=7,
            )
            ax.text(
                main_rect_x + main_rect_width - main_rect_width / 30,
                main_rect_y + main_rect_height - main_rect_height / 2.25,
                residents_diff_sub,
                fontsize=10,
                color=text_color,
                fontname="Roboto",
                ha="right",
                va="center",
                zorder=7,
            )
            ax.text(
                main_rect_x + main_rect_width - main_rect_width / 30,
                main_rect_y
                + main_rect_height
                - 3 * main_rect_height / 4
                + main_rect_height / 30,
                total_residents,
                fontsize=16,
                color=text_color,
                fontname="Roboto",
                ha="right",
                va="center",
                zorder=7,
            )
            ax.text(
                main_rect_x + main_rect_width - main_rect_width / 30,
                centroid.y - main_rect_height / 2.5,
                total_residents_sub,
                fontsize=10,
                color=text_color,
                fontname="Roboto",
                ha="right",
                va="center",
                zorder=7,
            )

            # Draw small rectangle and add text
            small_rect = patches.Rectangle(
                (main_rect_x, main_rect_y + main_rect_height),
                small_rect_width,
                small_rect_height,
                edgecolor="none",
                facecolor="fm_gold",
            )
            small_rect.set_zorder(6)
            ax.add_patch(small_rect)
            ax.text(
                main_rect_x + small_rect_width / 20,
                main_rect_y + main_rect_height + small_rect_height / 2,
                f"{row[self.indicators.config.shapefile_regional_spatial_name]}",
                ha="left",
                va="center",
                fontsize=12,
                color="white",
                fontname="Frank Ruhl Libre",
                zorder=7,
            )

    def draw(self, indicators):
        variable_to_plot = "Absolute_Change_Trend"

        # Prepare shapefile and indicators data
        shapefile = self.indicators.shapefile
        trends_geo_df = (
            gpd.GeoDataFrame(
                indicators["SnapshotTrendsResidentsAdm3"].merge(
                    shapefile, on=self.indicators.config.shapefile_spatial_id
                )
            )
            .set_crs(epsg=4326)
            .to_crs(epsg=3857)
        )
        adm1_geoseries = gpd.GeoSeries(
            shapefile.groupby(
                self.indicators.config.shapefile_regional_spatial_name
            ).apply(lambda x: x.geometry.unary_union)
        ).set_crs(epsg=4326)
        adm1_centroids = (
            shapefile.groupby(self.indicators.config.shapefile_regional_spatial_name)
            .apply(lambda x: x.geometry.unary_union.centroid)
            .to_frame()
            .rename(columns={0: "geometry"})
        )
        adm1_centroids = (
            gpd.GeoDataFrame(
                indicators["SnapshotMostRecentResidentsAdm1"]
                .to_frame()
                .join(indicators["SnapshotDiffResidentsAdm1"].to_frame())
                .join(adm1_centroids)
                .merge(
                    indicators["SnapshotTrendsResidentsAdm1"],
                    left_index=True,
                    right_on=self.indicators.config.shapefile_regional_spatial_name,
                )
            )
            .set_crs(epsg=4326)
            .to_crs(epsg=3857)
        )

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=[14, 14])
        title = "A map"

        # Add basemap, boundaries, and scalebar
        self.map_boundaries(ax, shapefile.to_crs(epsg=3857))
        self.add_plot_basemap(ax)
        self.add_scalebar(ax)

        # Configure axis appearance
        plt.tight_layout()
        ax.set_axis_off()

        # Define color bounds and colormap for the plot
        bounds = [-float("inf"), -1000, -500, -100, 100, 500, 1000, float("inf")]
        colormap = plt.get_cmap("fm_div")
        norm = mcolor.BoundaryNorm(bounds, ncolors=colormap.N, clip=True)

        # Define custom legend labels and colors
        legend_labels = [
            ">1,000 decrease",
            "500 - 1,000 decrease",
            "100 - 500 decrease",
            "Stable",
            "100 - 500 increase",
            "500 - 1,000 increase",
            ">1,000 increase",
        ][::-1]
        legend_colors = [colormap(norm(b)) for b in bounds[:-1]][
            ::-1
        ]  # Get the colors from the colormap

        # Plot trends data on the map
        trends_geo_df.query(
            "(Unusual == False) and (Fluctuating == False) and (has_abnormal_change == False) and (has_data == True)"
        ).plot(
            ax=ax,
            zorder=2,
            column=variable_to_plot,
            cmap=colormap,
            norm=norm,
            legend=False,
        )
        # Hash out fluctuating
        trends_geo_df[
            (trends_geo_df["Unusual"] == True)
            | (trends_geo_df["has_abnormal_change"] == True)
            | (trends_geo_df["Fluctuating"] == True)
        ].boundary.plot(ax=ax, zorder=3, color="k", alpha=0.4, hatch="////", lw=0.5)
        # Block out any areas we don't have data for
        trends_geo_df[trends_geo_df["has_data"] == False].plot(
            ax=ax, zorder=3, color="fm_land"
        )

        # Plot shapefile boundaries
        shapefile.boundary.to_crs(epsg=3857).plot(
            ax=ax, color="fm_land_edge", zorder=3, lw=0.6
        )
        adm1_geoseries.to_crs(epsg=3857).boundary.plot(
            ax=ax, edgecolor="fm_land_edge", lw=2.8, zorder=3
        )

        # Create and add a custom legend for the main data
        legend_patches = [
            patches.Patch(color=color, label=label)
            for color, label in zip(legend_colors, legend_labels)
        ]
        legend = ax.legend(
            handles=legend_patches,
            loc="upper left",
            fontsize=12,
            title="Population trend",
            title_fontsize=12,
        )
        ax.add_artist(legend)

        # Add custom legend for fluctuating areas below the main legend
        fluctuating_legend_patches = [
            patches.Patch(
                facecolor="none", edgecolor="black", hatch="////", label="Fluctuating"
            )
        ]
        fluctuating_legend = ax.legend(
            handles=fluctuating_legend_patches,
            loc="upper left",
            fontsize=12,
            bbox_to_anchor=(0.0, 0.85),
        )
        ax.add_artist(fluctuating_legend)

        # # Add place names to the map
        # self.add_place_names(ax)

        # Plot infoboxes on the map
        self.plot_infobox(adm1_centroids, 0.11, 0.064, 0.72, 0.32, ax)

        # Yield the figure
        yield Figure(
            figure=fig,
            ax=ax,
            caption="",
            title=title,
            filepath=slugify("national") + "/" + self.__class__.__name__,
        )
