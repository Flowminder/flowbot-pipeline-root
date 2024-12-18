from utils.indicator_loader import IndicatorLoader
from visualisations.base.figure import Figure


class TimeSeries:

    def __init__(self, indicators: IndicatorLoader):
        self.indicators = indicators

    def draw(self, indicators):
        raise NotImplementedError("")

    def make(self) -> Figure:

        with self.indicators.open() as indicators:
            for fig in self.draw(indicators):
                yield fig
