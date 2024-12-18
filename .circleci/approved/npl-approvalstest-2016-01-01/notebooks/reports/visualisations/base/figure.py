import matplotlib
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class Figure:
    figure: Optional[matplotlib.figure.Figure] = None
    csv: Optional[Union[pd.DataFrame, gpd.GeoDataFrame]] = None
    ax: matplotlib.axes._axes.Axes = field(default=None)
    caption: str = ""
    title: str = ""
    filepath: str = ""
