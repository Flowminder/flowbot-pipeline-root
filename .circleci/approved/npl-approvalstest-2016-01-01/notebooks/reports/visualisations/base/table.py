from utils.indicator_loader import IndicatorLoader
from visualisations.base.figure import Figure


class Table:

    def __init__(self, indicators: IndicatorLoader):
        self.indicators = indicators

    def draw(self, indicators):
        raise NotImplementedError("")

    def make(self) -> Figure:
        with self.indicators.open() as loaded_indicators:
            for table in self.draw(loaded_indicators):
                yield table
