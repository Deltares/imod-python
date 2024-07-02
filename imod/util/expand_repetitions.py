import itertools
from datetime import datetime
from typing import Dict, List


def expand_repetitions(
    repeat_stress: List[datetime], time_min: datetime, time_max: datetime
) -> Dict[datetime, datetime]:
    expanded = {}
    for year, date in itertools.product(
        range(time_min.year, time_max.year + 1),
        repeat_stress,
    ):
        newdate = date.replace(year=year)
        if newdate < time_max:
            expanded[newdate] = date
    return expanded
