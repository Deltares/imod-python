import datetime
import warnings

import cftime
import dateutil
import numpy as np

DATETIME_FORMATS = {
    14: "%Y%m%d%H%M%S",
    12: "%Y%m%d%H%M",
    10: "%Y%m%d%H",
    8: "%Y%m%d",
    4: "%Y",
}


def to_datetime(s):
    try:
        time = datetime.datetime.strptime(s, DATETIME_FORMATS[len(s)])
    except (ValueError, KeyError):  # Try fullblown dateutil date parser
        time = dateutil.parser.parse(s)
    return time

def _convert_datetimes(times, use_cftime):
    """
    Return times as np.datetime64[ns] or cftime.DatetimeProlepticGregorian
    depending on whether the dates fall within the inclusive bounds of
    np.datetime64[ns]: [1678-01-01 AD, 2261-12-31 AD].

    Alternatively, always returns as cftime.DatetimeProlepticGregorian if
    ``use_cf_time`` is True.
    """
    if all(time == "steady-state" for time in times):
        return times, False

    out_of_bounds = False
    if use_cftime:
        converted = [
            cftime.DatetimeProlepticGregorian(*time.timetuple()[:6]) for time in times
        ]
    else:
        for time in times:
            year = time.year
            if year < 1678 or year > 2261:
                out_of_bounds = True
                break

        if out_of_bounds:
            use_cftime = True
            msg = "Dates are outside of np.datetime64[ns] timespan. Converting to cftime.DatetimeProlepticGregorian."
            warnings.warn(msg)
            converted = [
                cftime.DatetimeProlepticGregorian(*time.timetuple()[:6])
                for time in times
            ]
        else:
            converted = [np.datetime64(time, "ns") for time in times]

    return converted, use_cftime


def _compose_timestring(time, time_format="%Y%m%d%H%M%S") -> str:
    """
    Compose timestring from time. Function takes care of different
    types of available time objects.
    """
    if time == "steady-state":
        return time
    else:
        if isinstance(time, np.datetime64):
            # The following line is because numpy.datetime64[ns] does not
            # support converting to datetime, but returns an integer instead.
            # This solution is 20 times faster than using pd.to_datetime()
            return time.astype("datetime64[us]").item().strftime(time_format)
        else:
            return time.strftime(time_format)