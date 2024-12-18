# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import List

from flowmachine.core import Query


class SingleColumnJoin(Query):
    def __init__(self, left: Query, right: Query, *, column_name: str):
        """
        Inner-join two queries using a single column,
        and keep only the column used for the join.

        Parameters
        ----------
        left, right : Query
            Queries to join
        column_name : str
            Name of the column to join on (column name must be the same in both queries)
        """
        if (column_name not in left.column_names) or (
            column_name not in right.column_names
        ):
            raise ValueError(f"Column '{column_name}' must be present in both queries.")
        self.left = left
        self.right = right
        self.column_name = column_name
        super().__init__()

    @property
    def column_names(self) -> List[str]:
        return [self.column_name]

    def _make_query(self):
        sql = f"""
        SELECT {self.column_name}
        FROM ({self.left.get_query()}) l
        INNER JOIN ({self.right.get_query()}) r
        USING ({self.column_name})
        """
        return sql
