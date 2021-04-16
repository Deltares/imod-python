"""
This module contains time related functions and can be potentially merged with 
imod.wq's timutil.py (also used a lot in flow), to a generic set of utilities.

This utility was made because a similar logic that was contained in 
ImodflowModel.time_discretization was required in PkgGroup, hence insert_unique_package_times.
Containing this function in model.py, however, results in a circular import with PkgGroups; 
contain the function in PkgGroups felt out of place.
"""

import numpy as np
import pandas as pd


def _to_list(t):
    """Helper function to catch packages that have only one time step"""

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
        List with times. This list will be extended with the package times if not present.

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
        if pkg._hastime():
            pkgtimes = _to_list(pkg["time"].values)
            first_times[key] = sorted(pkgtimes)[0]
            for var in pkg.dataset.data_vars:
                if "timemap" in pkg[var].attrs:
                    timemap_times = list(pkg[var].attrs["timemap"].keys())
                    pkgtimes.extend(timemap_times)
            times.extend(pkgtimes)

    # np.unique also sorts
    times = np.unique(np.hstack(times))

    return times, first_times
