from cookiecutter.utils import simple_filter
import pycountry

from flowbot_dataclasses.CountryStaticData import country_dict


@simple_filter
def to_iso(country_name):
    return pycountry.countries.get(name=country_name).alpha_3


@simple_filter
def region_query(country_name):
    return country_dict[country_name].region_query


@simple_filter
def pop_estimates_file(country_name):
    return str(country_dict[country_name].pop_estimates_file)


@simple_filter
def flow_weights_file(country_name):
    return str(country_dict[country_name].flow_weights_file)


@simple_filter
def full_name(country_name):
    return str(country_dict[country_name].full_name)


@simple_filter
def notebook_uid(country_name):
    return str(country_dict[country_name].notebook_uid)


@simple_filter
def notebook_gid(country_name):
    return str(country_dict[country_name].notebook_gid)


@simple_filter
def internal_network(country_name):
    return str(country_dict[country_name].internal_network)
