import pandas as pd
import numpy as np


def to_metaswap_timeformat(times):
    """
    Convert times to MetaSWAP's own time format, which consists of a year as
    integer and the number of days since the start of the year as float.

    Returns
    -------
    tuple
        Consists of the year as integer and the number of days since the
        start of the year as float.

    """

    # TODO: Also support cftime
    times = pd.DatetimeIndex(times)

    year = times.year

    # MetaSWAP requires a days since start year
    days_since_start_year = times.day_of_year.astype(np.float64) - 1.0
    # Behind the decimal is the time since start day
    time_since_start_day = times.hour / 24 + times.minute / 1440 + times.second / 86400

    time_since_start_year = days_since_start_year + time_since_start_day

    return year, time_since_start_year
