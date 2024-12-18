"""
Adds new colormaps:

'fm_seq' : sequential colormap for continuous data
'fm_div' : divering colormap for data either side of 0

'fm_qual' : qualitative colour map

New named colors: 'fm_land_edge', 'fm_land', 'fm_water_edge', 'fm_water', + all colors in fm_qual (fm_dark_blue, fm_gold, ... see below for full list)

'fm_flow' : cmap used for drawing flows
"""

import matplotlib as mpl
import numpy as np


def _hex_to_rgb(value):
    """Converts hex to rgb colours.

    Args:
        value (string): string of 6 characters representing a hex colour.

    Returns:
        tuple: list length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def _rgb_to_dec(value):
    """Converts rgb to decimal colours (i.e. divides each value by 256)

    Args:
        value (list): list (length 3) of RGB values

    Returns:
        list: (length 3) of decimal values
    """
    return [v / 256 for v in value]


def _get_continuous_cmap(hex_list, float_list=None):
    """creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Args:
        hex_list (list): list of hex code strings
        float_list (list, optional):  list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1. Defaults to None.

    Returns:
        colour map: colormap that can be used in cmap=''
    """
    rgb_list = [_rgb_to_dec(_hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def register_custom_colormaps():
    QUALITATIVE_LIST = [
        "#034174",
        "#CBA45A",
        "#701F53",
        "#006E8C",
        "#BF6799",
        "#00989A",
        "#9E6257",
    ]
    QUALITATIVE = mpl.colors.ListedColormap(QUALITATIVE_LIST)
    DIVERGING = _get_continuous_cmap(
        ["#034174", "#5F86BF", "#B6C6E4", "#F3F3F3", "#E6B8D0", "#BF6799", "#701F53"]
    )
    SEQUENTIAL = _get_continuous_cmap(
        ["#8A005E", "#9E3842", "#AB6E24", "#A99B3B", "#9DC379", "#98E0B8", "#B7EDE6"][
            ::-1
        ]
    )
    FLOWS = mpl.colors.LinearSegmentedColormap.from_list(
        "fm_flow", colors=["#701F53", "#27B288"], N=256
    )

    # Usually, sequential is for count data.
    # It's helpful to set the value of a spatial unit to be negative if we do not have any data in that region.
    # By using >> plt.scatter(*data, c = np.linspace(-3, 2, 5), cmap = 'fm_seq', vmin = 0), and points below 0 will be coloured grey.
    # It's important to set the minimum value (vmin) or the plotting library will not know when to classify counts as 'missing'.
    # Useful as well for redaction purposes, so you can easily hide anything below say, 100, by setting vmin = 100 in the plot that is being produced.
    SEQUENTIAL.set_under("#C6C6C6")

    # Registers colormaps so they can be used in plots.
    try:
        mpl.colormaps.register(cmap=QUALITATIVE, name="fm_qual")
        mpl.colormaps.register(cmap=DIVERGING, name="fm_div")
        mpl.colormaps.register(cmap=SEQUENTIAL, name="fm_seq")
        mpl.colormaps.register(cmap=FLOWS, name="fm_flow")
    except ValueError:
        pass

    # Add new 'named' colors when convinient (hack).
    fm_colors = {
        "fm_dark_blue": "#034174",
        "fm_gold": "#CBA45A",
        "fm_purple": "#701F53",
        "fm_dark_turquoise": "#006E8C",
        "fm_pink": "#BF6799",
        "fm_teal_blue": "#27B288",
        "fm_brown": "#9E6257",
        "fm_land_edge": "#CBA45A",
        "fm_land": "#EAE5DF",
        "fm_water_edge": "#9CBAC9",
        "fm_water": "#E5EDF1",
        "fm_missing": "#C6C6C6",
    }
    mpl.colors._colors_full_map.update(fm_colors)


register_custom_colormaps()
