# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from enum import StrEnum, auto
from hashlib import md5
import json
from typing import Callable, List

from bs4 import BeautifulSoup, Comment, Tag
import pandas as pd
import jinja2
from itertools import chain, islice
import csv
from dataclasses import dataclass, fields
import datetime as dt
from pathlib import Path

from translation_utils import TranslatorConfig, translations


def csv_to_html(csv_path, rows: slice) -> str:

    with open(csv_path) as fp:
        reader = csv.reader(fp)
        header = "\n".join(_html_row_gen(next(reader), item_tag="th"))
        row_html = "\n".join(
            "".join(_html_row_gen(row))
            for row in islice(reader, rows.start, rows.stop, rows.step)
        )
        return f"<table style='width:100%'>{header}{row_html}</table>"


def _html_row_gen(csv_row, item_tag="td"):
    yield "<tr>"
    yield from (f"<{item_tag}>{value}</{item_tag}>" for value in csv_row)
    yield "</tr>"


@dataclass
class ReportTable:
    data: Path | pd.DataFrame
    rows: int | slice = 6

    def __post_init__(self):
        if isinstance(self.rows, int):
            self.rows = slice(0, self.rows)
        if isinstance(self.data, str):
            self.data = Path(self.data)
        if isinstance(self.data, Path):
            self._html = csv_to_html(self.data, self.rows)
        elif isinstance(self.data, pd.DataFrame):
            self._html = self.data.iloc[self.rows].to_html(
                index=False, index_names=False, na_rep=""
            )
        else:
            raise TypeError(
                f"ReportTable only supports paths to CSVs and dataframes, not {type(self.data)}"
            )

    def __str__(self):
        return self._html


@dataclass
class SplitReportTable:
    data_path: Path
    max_rows_per_side: int

    def __post_init__(self):
        self.left = ReportTable(self.data_path, slice(0, self.max_rows_per_side))
        self.right = ReportTable(
            self.data_path, slice(self.max_rows_per_side, self.max_rows_per_side * 2)
        )


@dataclass(kw_only=True, eq=True, frozen=True)
class StaticReportImage:
    asset_folder: Path = Path("/opt/airflow/static_dir/templates/images")
    figure_path: Path

    def __str__(self):
        return f"<img src={self.asset_folder/ self.figure_path} style='width:100%; height:100%'/>"


def make_static_factory(asset_folder: Path) -> Callable:
    def inner(figure_path: Path) -> StaticReportImage:
        return StaticReportImage(asset_folder=asset_folder, figure_path=figure_path)

    return inner


@dataclass
class ReportImage:
    figure_path: Path

    def __str__(self):
        return f"<img src={self.figure_path} style='width:100%; height:100%'/>"


class ReportDate(dt.date):
    def __str__(self):
        return self.strftime("%d %B %Y")


@dataclass(frozen=True)
class ReportPeriod:
    start_date: dt.date
    end_date: dt.date

    def __str__(self):
        if self.start_date.month == self.end_date.month:
            return f"{self.start_date.day}-{self.end_date.day} {self.end_date.strftime('%B')} {self.end_date.strftime('%Y')}"
        elif self.start_date.year == self.end_date.year:
            return f"{self.start_date.strftime('%d %b')} - {self.end_date.strftime('%d %b')} {self.end_date.strftime('%Y')}"
        else:
            return f"{self.start_date.strftime('%d %b %Y')} - {self.end_date.strftime('%d %b %Y')}"

    def short(self):
        if self.start_date.month == self.end_date.month:
            return f"{self.start_date.day}-{self.end_date.day} {self.end_date.strftime('%b')} {self.end_date.strftime('%Y')}"
        elif self.start_date.year == self.end_date.year:
            return f"{self.start_date.strftime('%d %b')} - {self.end_date.strftime('%d %b')} {self.end_date.strftime('%Y')}"
        else:
            return f"{self.start_date.strftime('%d %b %y')} - {self.end_date.strftime('%d %b %y')}"


@dataclass(kw_only=True)
class Template:
    pub_date: ReportDate
    partner_logo_1: StaticReportImage = None
    partner_logo_2: StaticReportImage = None
    partner_logo_3: StaticReportImage = None
    partner_logo_4: StaticReportImage = None
    partner_logo_5: StaticReportImage = None
    fm_logo_white: StaticReportImage = StaticReportImage(
        figure_path=Path("fm_white.png")
    )
    fm_logo_blue: StaticReportImage = StaticReportImage(figure_path=Path("fm_blue.png"))
    page_num: int
    total_pages: int
    country: str
    period: ReportPeriod
    period_short: ReportPeriod

    def html_dict(self) -> dict:
        return {field.name: str(getattr(self, field.name)) for field in fields(self)}


@dataclass(kw_only=True)
class PreparednessTemplate(Template):
    pub_date: ReportDate
    country: str


@dataclass(kw_only=True)
class PreparednessTemplateCover(PreparednessTemplate):
    key_obs: str
    top_average_change_table: ReportImage
    period_start: ReportDate
    period_end: ReportDate
    country_pop_map: ReportImage


@dataclass(kw_only=True)
class PreparednessTemplateDepartment(PreparednessTemplate):
    department_name: str
    dept_pop_ts: ReportImage
    outlier_areas_ts: ReportImage
    fluctuating_areas_ts: ReportImage
    pop_trend_areas_map: ReportImage
    period_start: ReportDate
    period_end: ReportDate


@dataclass(kw_only=True)
class PreparednessTemplateSummary(PreparednessTemplate):
    department_name: str
    areas_of_interest_table_left: ReportTable
    areas_of_interest_table_right: ReportTable


@dataclass(kw_only=True)
class PreparednessTemplateBackMatter(PreparednessTemplate):
    month: str
    year: int
    prep_month: str
    prep_year: int
    prep_num: int
    strec_month: str
    strec_year: int
    strec_num: int


@dataclass(kw_only=True)
class CrisisTemplate(Template):
    name_of_crisis: str
    crisis_date: ReportDate
    update_freq_days: int
    update_num: int
    period: ReportPeriod


@dataclass(kw_only=True)
class CrisisTemplateCover(CrisisTemplate):
    manual_key_observations: str
    displaced_subs_table: ReportTable
    location_displacement_table: ReportTable
    area_map: ReportImage
    displaced_plot: ReportImage
    remaining_displaced: int
    displaced_neighbourhoods: int
    newly_displaced: int
    newly_displaced_ts: ReportImage
    nomad_staying: StaticReportImage = StaticReportImage(
        figure_path=Path("nomad_staying.png")
    )
    nomad_leaving: StaticReportImage = StaticReportImage(
        figure_path=Path("nomad_going.png")
    )


@dataclass(kw_only=True)
class CrisisTemplateRegion(CrisisTemplate):
    affected_area: str
    area_map: str
    total_displaced: int
    displaced_since_last: int
    newly_displaced_ts: ReportImage
    remaining_displaced_ts: ReportImage


@dataclass(kw_only=True)
class CrisisTemplateRegionContinued(CrisisTemplate):
    affected_area: str
    left_table: ReportTable
    right_table: ReportTable


# TODO: this doesn't need to
@dataclass(kw_only=True)
class CrisisTemplateBackMatter(CrisisTemplate):
    month: str
    year: int
    prep_month: str
    prep_year: int
    prep_num: int
    strec_month: str
    strec_year: int
    strec_num: int


def render_report(
    page: Template,
    template_name: str,
    style: str,
    env: jinja2.Environment,
    out_folder: Path,
    out_name: str,
    translator_config: TranslatorConfig,
):
    page_template = env.get_template(template_name)
    base_template = env.get_template("page_preamble.html")
    stylesheet = env.get_template(style).render()
    for country, translation in translations(template_name, translator_config):
        localised_html = page_template.render(translation)
        print(localised_html)
        localised_template = jinja2.Environment(loader=jinja2.BaseLoader).from_string(
            localised_html
        )
        content = localised_template.render(page.html_dict())
        out_path = out_folder / country / out_name
        (out_folder / country).mkdir(exist_ok=True)
        out_path.write_text(
            base_template.render(content=content, stylesheet=stylesheet)
        )
