from dataclasses import dataclass
from enum import StrEnum, auto


class Country(StrEnum):
    Haiti = auto()
    Ghana = auto()
    Nepal = auto()
    DRC = auto()


class Language(StrEnum):
    English = auto()
    French = auto()


@dataclass
class CountryConfig:
    """
    Configuration settings for a specific country.

    Attributes:
        projected_population_spatial_id (str): Name of column in projected population dataframe that (uniquely) identifies spatial units.
        shapefile_spatial_id (str): Name of column in shapefile that (uniquely) identifies spatial units.
        shapefile_spatial_name (str): Name of column in shapefile for actual names of spatial units for analysis.
        shapefile_regional_spatial_name (str): Name of column in shapefule for names of regions for subreport enum purposes.
        monthly_residents_spatial_id (str): Name of column in shapefule that (uniquely) identifies spatial units.
    """

    projected_population_spatial_id: str
    shapefile_spatial_id: str
    shapefile_spatial_name: str
    shapefile_regional_spatial_name: str
    monthly_residents_spatial_id: str
    date_column: str
    language: Language  # I think we break this out - it's not tied to the country


# NOTE: A lot of this is replicated in either the cookiecutter config or the dags themselves
haiti_config = CountryConfig(
    projected_population_spatial_id="pcod",
    shapefile_spatial_id=f"admin3pcod",
    shapefile_spatial_name=f"admin3name",
    shapefile_regional_spatial_name=f"admin1name",
    monthly_residents_spatial_id="pcod",
    date_column="date",
    language=Language.English,
)

nepal_config = CountryConfig(
    projected_population_spatial_id="pcod",
    shapefile_spatial_id=f"admin3pcod",
    shapefile_spatial_name=f"admin3name",
    shapefile_regional_spatial_name=f"admin1name",
    monthly_residents_spatial_id="pcod",
    date_column="date",
    language=Language.English,
)


def get_country_config(
    country: Country | str, language: Language = Language.English
) -> CountryConfig:

    # TODO: modify this to use pycountry
    if country in [Country.Haiti, "hti"]:
        out = haiti_config
    elif country in [Country.Nepal, "npl"]:
        out = nepal_config
    else:
        raise ValueError(f"Configuration for country '{country}' is not defined.")
    out.language = language
    return out
