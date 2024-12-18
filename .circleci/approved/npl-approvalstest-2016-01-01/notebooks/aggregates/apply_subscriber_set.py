# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import List

from flowmachine.core import Query


class ApplySubscriberSet(Query):
    """
    Given a parent query that returns a per-subscriber result, and a query that
    specifies a set of subscribers, this query will return a result for each
    subscriber in the specified set (taking values from the parent query where present,
    or as specified in `fill_values` otherwise).
    """

    # TODO: Full docstring and type hints
    def __init__(
        self,
        *,
        parent,
        subscriber_set,
        fill_values,
    ):
        self.parent = parent
        self.subscriber_set = subscriber_set
        self.fill_values = fill_values
        if not "subscriber" in self.parent.column_names:
            raise ValueError("Parent query must have a 'subscriber' column")
        if not "subscriber" in self.subscriber_set.column_names:
            raise ValueError("Subscriber set query must have a 'subscriber' column")
        if not all(key in self.parent.column_names for key in fill_values):
            # TODO: More helpful error message
            raise ValueError("Received fill value(s) for unrecognised column(s)")
        super().__init__()

    @property
    def column_names(self) -> List[str]:
        return self.parent.column_names

    def _make_query(self):
        value_columns = [col for col in self.parent.column_names if col != "subscriber"]
        # TODO: This coalesce will work for numeric fill values, but not strings.
        # Should really use psycopg2 mogrify to make this more robust
        filled_values_string = ", ".join(
            (
                f"COALESCE(q.{col}, {self.fill_values[col]}) AS {col}"
                if col in self.fill_values
                else f"q.{col}"
            )
            for col in value_columns
        )

        sql = f"""
        SELECT subscriber, {filled_values_string}
        FROM ({self.parent.get_query()}) AS q
        RIGHT JOIN ({self.subscriber_set.get_query()}) AS s
        USING (subscriber)
        """

        return sql
