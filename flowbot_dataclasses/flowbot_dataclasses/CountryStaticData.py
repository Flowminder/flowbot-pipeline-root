from dataclasses import dataclass
from pathlib import Path


@dataclass
class CountryStaticData:
    region_query: (
        str  # Query that returns pcod3 IDs along with associacted pcod 3,2 and 1 names
    )
    pop_estimates_file: Path
    flow_weights_file: Path
    full_name: str
    notebook_uid: str = "1001"
    notebook_gid: str = "1001"
    internal_network: str = "flowbot_flowapi_flowmachine"


npl = CountryStaticData(
    region_query="""
        SELECT admin3pcod, admin3name, admin2name, admin1name
        FROM geography.admin3
        LEFT JOIN geography.admin2 ON admin2pcod = substring(admin3pcod FOR 7) || '_1'
        LEFT JOIN geography.admin1 ON admin1pcod = substring(admin2pcod FOR 5) || '_1';
    """,
    pop_estimates_file=Path("data/nepal_pop_estimates.csv"),
    flow_weights_file=Path("data/nepal_flow_weights.csv"),
    full_name="Nepal",
)
hti = CountryStaticData(
    region_query="""
        SELECT admin3pcod, admin3name, admin2name, admin1name
        FROM geography.admin3
        LEFT JOIN geography.admin2 ON admin2pcod = substring(admin3pcod FOR 6)
        LEFT JOIN geography.admin1 ON admin1pcod = substring(admin2pcod FOR 4)
    """,
    pop_estimates_file=Path("data/haiti_pop_estimates.csv"),
    flow_weights_file=Path("data/hti_flow_weights.csv"),
    full_name="Haiti",
    notebook_uid="1189",
    notebook_gid="502",
)
gha = CountryStaticData(
    region_query="UNSET",
    pop_estimates_file=Path("data/ghana_pop_estimates.csv"),
    flow_weights_file=Path("data/ghana_flow_weights.csv"),
    full_name="Ghana",
)
drc = CountryStaticData(
    region_query="UNSET",
    pop_estimates_file=Path("data/drc_pop_estimates.csv"),
    flow_weights_file=Path("data/drc_flow_weights.csv"),
    full_name="The Democratic Republic of the Congo",
)

country_dict = dict(npl=npl, hti=hti, gha=gha, drc=drc)
