import datetime

import cftime
import dateutil  # is a dependency of pandas
import numpy as np
import pandas as pd


def to_datetime(time, use_cftime):
    """
    Check whether time is cftime object, else convert to datetime64 series.

    cftime currently has no pd.to_datetime equivalent:
    a method that accepts a lot of different input types.

    Parameters
    ----------
    time : cftime object or datetime-like scalar
    """
    if isinstance(time, cftime.datetime):
        return time
    elif isinstance(time, np.datetime64):
        return time
    elif isinstance(time, str):
        time = dateutil.parser.parse(time)
        year = time.year
        if year < 1678 or year > 2261:
            if not use_cftime:
                raise ValueError(
                    "A datetime is out of bounds for np.datetime64[ns]: "
                    "before year 1678 or after 2261. You will have to use "
                    "cftime.datetime and xarray.CFTimeIndex in your model "
                    "input instead of the default np.datetime64[ns] datetime "
                    "type."
                )

    if use_cftime:
        return cftime.DatetimeProlepticGregorian(*time.timetuple()[:6])
    else:
        return np.datetime64(time, "ns")


array_to_datetime = np.vectorize(to_datetime)


def timestep_duration(times, use_cftime):
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


def forcing_starts_ends(package_times, globaltimes):
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
    assert len(package_times) == len(starts_ends)
    return starts_ends
