import itertools
from datetime import datetime
from typing import Dict, List

import numpy as np


def expand_repetitions(
    repeat_stress: List[datetime], time_min: datetime, time_max: datetime
) -> Dict[np.datetime64, np.datetime64]:
    expanded = {}
    for year, date in itertools.product(
        range(time_min.year, time_max.year + 1),
        repeat_stress,
    ):
        newdate = np.datetime64(date.replace(year=year))
        if newdate < time_max:
            expanded[newdate] = np.datetime64(date)
    return expanded
