import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolor
import geopandas as gpd
from slugify import slugify
from visualisations.base.map import Map
from visualisations.base.figure import Figure
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects

# from adjustText import adjust_text


class MapResidentsTrendsByRegion(Map):

    def draw(self, indicators: gpd.GeoDataFrame):
        # Load shapefile and set title
        shp = self.indicators.shapefile

        # Merge and transform shapefile data
        trends_admin3_geo = (
            gpd.GeoDataFrame(
                indicators["SnapshotTrendsResidentsAdm3"].merge(
                    shp,
                    on=self.indicators.config.shapefile_spatial_id,
                    suffixes=("", "_"),
                )
            )
            .set_crs(epsg=4326)
            .to_crs(epsg=3857)
        )

        # Transform shapefile data once
        shp_3857 = shp.to_crs(epsg=3857)
        trends_admin3_geo_3857 = trends_admin3_geo.to_crs(epsg=3857)

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

        for adm1_name, adm1_df in trends_admin3_geo_3857.groupby(
            self.indicators.config.shapefile_regional_spatial_name
        ):

            X = 2

            # Get top 3 increasing and decreasing trends
            topX_decreasing = (
                indicators["SnapshotTrendsResidentsAdm3"]
                .query(
                    f"({self.indicators.config.shapefile_regional_spatial_name} == @adm1_name) and (Unusual == False) and (Fluctuating == False) and (has_abnormal_change == False)"
                )
                .sort_values("Absolute_Change_Trend")
                .head(X)
                .query("Absolute_Change_Trend < -100")
            )

            topX_increasing = (
                indicators["SnapshotTrendsResidentsAdm3"]
                .query(
                    f"({self.indicators.config.shapefile_regional_spatial_name} == @adm1_name) and (Unusual == False) and (Fluctuating == False) and (has_abnormal_change == False)"
                )
                .sort_values("Absolute_Change_Trend")
                .tail(X)
                .query("Absolute_Change_Trend > 100")
            )

            focal_adm3_topX = set(
                topX_increasing[self.indicators.config.shapefile_spatial_id]
            ).union(set(topX_decreasing[self.indicators.config.shapefile_spatial_id]))
            focal_adm3_top6 = set({})

            # fluctuating placenames
            top_fluct = set(
                adm1_df.query(
                    "((has_abnormal_change == True) or (Fluctuating == True) or (Unusual == True)) and (has_data)"
                )
                .sort_values("largest_abnormal_fluct")
                .query("~largest_abnormal_fluct.isnull()")
                .tail(2)[self.indicators.config.shapefile_spatial_id]
            )

            fig, ax = plt.subplots(figsize=[14, 14])

            # Group and transform shapefile data for plotting
            adm1 = gpd.GeoSeries(
                shp_3857.groupby(
                    self.indicators.config.shapefile_regional_spatial_name
                ).apply(lambda x: x.geometry.unary_union)
            ).set_crs(epsg=3857)
            print(adm1)

            # Plot boundaries and background map
            shp_3857.boundary.plot(ax=ax, color="fm_land_edge", zorder=4, lw=0.6)
            adm1.boundary.plot(ax=ax, edgecolor="fm_land_edge", lw=2.5, zorder=4)

            # Plot the main data
            adm1_df.query(
                "(Unusual == False) and (Fluctuating == False) and (has_data == True) and (has_abnormal_change == False)"
            ).plot(
                ax=ax,
                zorder=2,
                column="Absolute_Change_Trend",
                cmap=colormap,
                norm=norm,
                legend=False,
            )
            adm1_df[
                (adm1_df["Unusual"] == True)
                | (adm1_df["Fluctuating"] == True)
                | (adm1_df["has_abnormal_change"] == True)
            ].boundary.plot(ax=ax, zorder=3, color="k", alpha=0.3, hatch="////", lw=0.5)
            adm1_df[adm1_df["has_data"] == False].plot(ax=ax, zorder=2, color="fm_land")

            self.map_boundaries(ax, adm1_df)
            self.add_plot_basemap(ax)
            self.add_scalebar(ax)

            # Plot region names at representative points with enhanced visibility
            texts = []
            for idx, row in shp_3857.query(
                f"{self.indicators.config.shapefile_spatial_id} in @focal_adm3_topX or {self.indicators.config.shapefile_spatial_id} in @top_fluct"
            ).iterrows():
                centroid = row.geometry.centroid
                rep_point = (
                    centroid
                    if row.geometry.contains(centroid)
                    else row.geometry.representative_point()
                )
                bbox_props = dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="none",
                    facecolor="white",
                    alpha=0.7,
                )
                text = ax.annotate(
                    xy=(rep_point.x, rep_point.y),
                    text=row[self.indicators.config.shapefile_spatial_name],
                    fontsize=12,
                    ha="center",
                    va="center",
                    color="black",
                    bbox=bbox_props,
                    zorder=5,
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=0.1, foreground="black"),
                        path_effects.Normal(),
                    ]
                )
                texts.append(text)

            # Adjust text positions to avoid overlap, draw arrows for moved texts
            # adjust_text(texts, ax=ax, arrowprops=dict(
            #             arrowstyle='-|>',  # Line and arrow style
            #             mutation_scale=20,  # Overall scaling factor for the arrow
            #             lw=1,               # Line width
            #     color='fm_dark_blue', zorder=4),min_arrow_len = 0, force_static = (0.30, 4*0.35), force_explode = (0.15, 0.22), force_text=(0.35, 0.45), force_pull=(0.03, 0.04))

            ax.set_axis_off()

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
                    facecolor="none",
                    edgecolor="black",
                    hatch="////",
                    label="Fluctuating",
                )
            ]
            fluctuating_legend = ax.legend(
                handles=fluctuating_legend_patches,
                loc="upper left",
                fontsize=12,
                bbox_to_anchor=(0.0, 0.85),
            )
            ax.add_artist(fluctuating_legend)

            self.add_place_names(ax)

            plt.tight_layout()

            total_yoy_diff = (
                indicators["SnapshotDiffResidentsAdm1"]
                .to_frame()
                .query(
                    f"{self.indicators.config.shapefile_regional_spatial_name} == @adm1_name"
                )
                .values[0][0]
            )

            yield Figure(
                figure=fig,
                ax=ax,
                caption="",
                title=f"{total_yoy_diff:,} residents",
                filepath=slugify(adm1_name) + "/" + self.__class__.__name__,
            )
