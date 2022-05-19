"""
This module contains time related functions and can be potentially merged with
imod.wq's timutil.py (also used a lot in flow), to a generic set of utilities.

This utility was made because a similar logic that was contained in
ImodflowModel.create_time_discretization was required in PkgGroup, hence
insert_unique_package_times.  Containing this function in model.py, however,
results in a circular import with PkgGroups; contain the function in PkgGroups
felt out of place.
"""

import numpy as np
import pandas as pd


def _to_list(t):
    """Catch packages that have only one time step"""
    if not isinstance(t, (np.ndarray, list, tuple, pd.DatetimeIndex)):
        return [t]
    else:
        return list(t)


def insert_unique_package_times(package_mapping, manual_insert=[]):
    """
    Insert unique package times in a list of times.

    Parameters
    ----------
    package_mapping : iterable
        Iterable of key, package pairs
    manual_insert : iterable of times, np.datetime64, or cftime.datetime
        List with times. This list will be extended with the package times if
        not present.

    Returns
    -------
    times : list
        List with times, extended with package times
    first_times : dict
        Dictionary with first timestamp per package
    """

    times = _to_list(manual_insert)

    first_times = {}
    for key, pkg in package_mapping:
        if pkg._is_periodic():
            continue  # Periodic stresses can start earlier than model time domain in projectfile
        if pkg._hastime():
            pkgtimes = _to_list(pkg["time"].values)
            first_times[key] = sorted(pkgtimes)[0]
            for var in pkg.dataset.data_vars:
                if "stress_repeats" in pkg[var].attrs:
                    stress_repeats_times = list(pkg[var].attrs["stress_repeats"].keys())
                    pkgtimes.extend(stress_repeats_times)
            times.extend(pkgtimes)

    # np.unique also sorts
    times = np.unique(np.hstack(times))

    return times, first_times


def forcing_starts(package_times, globaltimes):
    """
    Determines the stress period numbers for start for a forcing defined at a
    starting time, until the next starting time.

    Note
    ----
    This is and adapted version from imod.wq.timeutil.forcings_starts_ends

    Parameters
    ----------
    package_times : np.array, listlike
        Treated as starting time of forcing
    globaltimes : np.array, listlike
        Global times of the simulation. Defines starting time of the stress
        periods.

    Returns
    -------
    starts : list of tuples
        For every entry in the package, return index of the start.

    """
    # From searchsorted docstring:
    # Find the indices into a sorted array a such that, if the corresponding
    # elements in v were inserted before the indices, the order of a would be
    # preserved.
    # Add one because of difference in 0 vs 1 based indexing.
    starts = np.searchsorted(globaltimes, package_times) + 1
    # convert to strings, complying with iMOD-WQ
    starts = [str(start) for start in starts]
    return starts
