from utils.indicator_loader import IndicatorLoader
from visualisations.base.figure import Figure


class Infobox:

    def __init__(self, indicators: IndicatorLoader):
        self.indicators = indicators

    def draw(self, indicators):
        raise NotImplementedError("")

    def make(self) -> Figure:

        # plt.rcParams["font.family"] = "Roboto"
        # plt.rcParams["font.sans-serif"] = ["Roboto"]
        # plt.rcParams["font.size"] =  8

        with self.indicators.open() as loaded_indicators:
            for fig in self.draw(loaded_indicators):
                yield fig
