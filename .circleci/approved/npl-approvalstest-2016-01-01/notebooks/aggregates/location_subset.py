# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import List

from flowmachine.core import Query
from flowmachine.features.utilities.subscriber_locations import BaseLocation


class LocationSubset(BaseLocation, Query):
    """
    Subset a subscriber location query to a specified set of locations.
    """

    # TODO: Full docstring and type hints
    def __init__(
        self,
        *,
        parent,
        locations,
    ):
        self.parent = parent
        self.locations = locations
        self.spatial_unit = parent.spatial_unit
        super().__init__()

    @property
    def column_names(self) -> List[str]:
        return self.parent.column_names

    def _make_query(self):
        return f"""
        SELECT {self.parent.column_names_as_string_list}
        FROM ({self.parent.get_query()}) AS parent
        {self.parent.spatial_unit.location_subset_clause(self.locations)}
        """
