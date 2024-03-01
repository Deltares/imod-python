import datetime
import warnings

import cftime
import dateutil
import numpy as np
import pandas as pd

DATETIME_FORMATS = {
    14: "%Y%m%d%H%M%S",
    12: "%Y%m%d%H%M",
    10: "%Y%m%d%H",
    8: "%Y%m%d",
    4: "%Y",
}


def to_datetime(s: str) -> datetime.datetime:
    """
    Convert string to datetime. Part of the public API for backwards
    compatibility reasons. 
    
    Fast performance is important, as this function is used to parse IDF names,
    so it being called 100,000 times is a common usecase. Function stored
    previously under imod.util.to_datetime. 
    """
    try:
        time = datetime.datetime.strptime(s, DATETIME_FORMATS[len(s)])
    except (ValueError, KeyError):  # Try fullblown dateutil date parser
        time = dateutil.parser.parse(s)
    return time


def _check_year(year: int) -> None:
    """Check whether year is out of bounds for np.datetime64[ns]"""
    if year < 1678 or year > 2261:
        raise ValueError(
            "A datetime is out of bounds for np.datetime64[ns]: "
            "before year 1678 or after 2261. You will have to use "
            "cftime.datetime and xarray.CFTimeIndex in your model "
            "input instead of the default np.datetime64[ns] datetime "
            "type."
        )


def to_datetime_internal(
        time: cftime.datetime | np.datetime64 | str, use_cftime: bool
    ) -> np.datetime64 | cftime.datetime:
    """
    Check whether time is cftime object, else convert to datetime64 series.

    cftime currently has no pd.to_datetime equivalent: a method that accepts a
    lot of different input types. Function stored previously under
    imod.wq.timeutil.to_datetime. 

    Parameters
    ----------
    time : cftime object or datetime-like scalar
    """
    if isinstance(time, cftime.datetime):
        return time
    elif isinstance(time, np.datetime64):
        # Extract year from np.datetime64.
        # First force a yearly datetime64 type,
        # convert to int, and add the reference year.
        # This appears to be the safest method
        # see https://stackoverflow.com/a/26895491
        # time.astype(object).year, produces inconsistent
        # results when 'time' is datetime64[d] or when it is datetime64[ns]
        # at least for numpy version 1.20.1
        year = time.astype("datetime64[Y]").astype(int) + 1970
        _check_year(year)
        # Force to nanoseconds, concurrent with xarray and pandas.
        return time.astype(dtype="datetime64[ns]")
    elif isinstance(time, str):
        time = to_datetime(time)
        if not use_cftime:
            _check_year(time.year)

    if use_cftime:
        return cftime.DatetimeProlepticGregorian(*time.timetuple()[:6])
    else:
        return np.datetime64(time, "ns")


def timestep_duration(times: np.array, use_cftime: bool):
    """
    Generates dictionary containing stress period time discretization data.

    Parameters
    ----------
    times : np.array
        Array containing containing time in a datetime-like format

    Returns
    -------
    duration : 1D numpy array of floats
        stress period duration in decimal days
    """
    if not use_cftime:
        times = pd.to_datetime(times)

    timestep_duration = []
    for start, end in zip(times[:-1], times[1:]):
        timedelta = end - start
        duration = timedelta.days + timedelta.seconds / 86400.0
        timestep_duration.append(duration)
    return np.array(timestep_duration)


def forcing_starts_ends(package_times: np.array, globaltimes: np.array):
    """
    Determines the stress period numbers for start and end for a forcing defined
    at a starting time, until the next starting time.
    Numbering is inclusive, in accordance with the iMODwq runfile.

    Parameters
    ----------
    package_times : np.array, listlike
        Treated as starting time of forcing
    globaltimes : np.array, listlike
        Global times of the simulation. Defines starting time of the stress
        periods.

    Returns
    -------
    starts_ends : list of tuples
        For every entry in the package, return index of start and end.
        Numbering is inclusive.
    """
    # From searchsorted docstring:
    # Find the indices into a sorted array a such that, if the corresponding
    # elements in v were inserted before the indices, the order of a would be
    # preserved.
    # Add one because of difference in 0 vs 1 based indexing.
    starts = np.searchsorted(globaltimes, package_times) + 1
    ends = np.append(starts[1:] - 1, len(globaltimes))
    starts_ends = [
        f"{start}:{end}" if (end > start) else str(start)
        for (start, end) in zip(starts, ends)
    ]
    return starts_ends


def _convert_datetimes(times: np.array, use_cftime: bool):
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


def _compose_timestring(
        time: np.datetime64 | cftime.datetime, time_format: str="%Y%m%d%H%M%S"
    ) -> str:
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