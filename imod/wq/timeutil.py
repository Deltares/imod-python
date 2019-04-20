import numpy as np
import pandas as pd
import cftime


def to_datetime(time):
    """
    Check whether time is cftime object, else convert to datetime64 series.
    
    cftime currently has no pd.to_datetime equivalent:
    a method that accepts a lot of different input types.
    
    Parameters
    ----------
    time : cftime object or datetime-like scalar
    """
    # TODO: deal with CFTimeIndex as well
    if isinstance(time, cftime.datetime):
        return time
    else:
        return pd.to_datetime(time)


def timestep_duration(times):
    """
    Generates dictionary containing stress period time discretization data.

    Parameters
    ----------
    times : np.array
        Array containing containing time in a datetime-like format
    
    Returns
    -------
    collections.OrderedDict
        Dictionary with dates as strings for keys,
        stress period duration (in days) as values.
    """
    times = sorted([to_datetime(t) for t in times])

    timestep_duration = []
    for start, end in zip(times[:-1], times[1:]):
        timedelta = end - start
        duration = timedelta.days + timedelta.seconds / 86400.0
        timestep_duration.append(duration)
    return times, timestep_duration


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
