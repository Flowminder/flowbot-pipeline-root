import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dateutil.relativedelta import relativedelta
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import colorsys
import textwrap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from visualisations.base.infobox import Infobox
from visualisations.base.figure import Figure
from slugify import slugify


class InfoboxTop6MonthlyChanges(Infobox):

    def is_color_dark(self, color):
        rgb = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        return l < 0.5

    def add_wrapped_text(self, ax, x, y, text, width, **kwargs):
        wrapped_text = textwrap.fill(text, width=width)
        for i, line in enumerate(wrapped_text.split("\n")):
            ax.text(x, y - i * 0.15, line, **kwargs)

    def draw(self, indicators):
        # Normalize values for colormap
        norm = mcolors.Normalize(vmin=0, vmax=15)
        cmap = plt.get_cmap("fm_seq")

        data = indicators["MonthlyTop6Changes"]

        for adm1, adm1_df in data.groupby("ADM1_EN", sort=False):

            w = 143 / 60
            h = 67 / 60

            fig, ax = plt.subplots(2, 3, figsize=(w * 3, h * 2))

            for ix, (index, entry) in enumerate(adm1_df.iterrows()):

                row, col = divmod(ix, 3)
                row = 1 - row
                col = 2 - col
                color = cmap(norm(abs(entry["abnormality"])))

                img_color = "white" if self.is_color_dark(color) else "black"

                aspect_ratio = ax[row, col].get_data_ratio()

                rect = patches.Rectangle(
                    (0, 1), 0.96, 0.96, linewidth=1, facecolor=color
                )
                ax[row, col].add_patch(rect)

                self.add_wrapped_text(
                    ax[row, col],
                    0.27,
                    0.92,
                    entry["ADM3_EN"],
                    width=25,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="k",
                    font="Frank Ruhl Libre",
                )
                add_text = "+" if str(entry["residents"])[0] != "-" else ""
                ax[row, col].text(
                    0.27,
                    0.45,
                    add_text
                    + "{:,.0f}".format(np.round(entry["residents"], -2))
                    + " residents",
                    ha="left",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="k",
                    font="Frank Ruhl Libre",
                )

                date = pd.to_datetime(entry["date"])
                previous_month_date = date - relativedelta(months=1)
                ax[row, col].text(
                    0.27,
                    0.225,
                    previous_month_date.strftime("%b %Y")
                    + " - "
                    + date.strftime("%b %Y"),
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="k",
                    fontweight="light",
                )

                ax[row, col].set_xlim(0, 1)
                ax[row, col].set_ylim(0, 1)
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])

                light_grey = "#d3d3d3"
                for spine in ax[row, col].spines.values():
                    spine.set_edgecolor(light_grey)

                circ = patches.Ellipse(
                    (0.14, 0.5), 0.21, 0.21 * (w / h), facecolor=color
                )
                ax[row, col].add_patch(circ)

                # Load an image
                if (img_color == "black") and (add_text == "+"):
                    image_path = "./static_imgs/population(4).png"
                elif (img_color == "black") and (add_text == ""):
                    image_path = "./static_imgs/population(3).png"
                elif (img_color == "white") and (add_text == ""):
                    image_path = "./static_imgs/population(2).png"
                elif (img_color == "white") and (add_text == "+"):
                    image_path = "./static_imgs/population(1).png"

                image = mpimg.imread(image_path)

                # Create an OffsetImage
                imagebox = OffsetImage(image, zoom=0.035, interpolation="sinc")

                # Create an AnnotationBbox
                ab = AnnotationBbox(
                    imagebox, (0.14, 0.5), frameon=False, box_alignment=(0.5, 0.5)
                )

                # Add the AnnotationBbox to the plot
                ax[row, col].add_artist(ab)

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.04, hspace=0.09)

            yield Figure(
                figure=fig,
                ax=ax,
                caption="",
                title="",
                filepath=slugify(adm1) + "/" + self.__class__.__name__,
            )
