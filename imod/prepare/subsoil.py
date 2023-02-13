import numba
import numpy as np
import xarray as xr


@numba.njit
def _build_lookup_table(lithoclass, lithostrat, values):
    """
    Takes three arrays, read in from a .csv file giving parameter information.

    Returns an array with N x M elements, where N is the maximum lithostrat
    code, and M the maximum lithoclass code, as integers.

    The returned array is (much) larger than strictly necessary (as lithostrat
    numbering starts in the thousands, and is not sequential). However, numpy
    indexing is quick, and this method is conceptually very simple as it
    requires no further modification of lithostrat codes.

    A workaround for the lack of dictionary support in numba. See also:
    https://github.com/numba/numba/issues/2096
    """
    if not (lithoclass.shape == lithostrat.shape == values.shape):
        raise ValueError("shape mismatch between lithoclass, lithostrat, and values")
    nclasses = lithoclass.shape[0]
    nrow = int(lithostrat.max()) + 1
    ncol = int(lithoclass.max()) + 1
    array = np.full((nrow, ncol), np.nan)
    for n in range(nclasses):
        i = lithostrat[n]
        j = lithoclass[n]
        array[i, j] = values[n]
    return array


def build_lookup_table(df, lithoclass, lithostrat, kcolumn):
    return _build_lookup_table(
        df[lithoclass].values.astype(np.int8),
        df[lithostrat].values.astype(np.int16),
        df[kcolumn].values,
    )


@numba.njit
def _check_litho_strat_combinations(lithostrat, lithology, lookup_table):
    missing = [(0, 0)]  # for type inference
    for strat, litho in zip(lithostrat.flatten(), lithology.flatten()):
        if np.isnan(lookup_table[strat, litho]):
            missing.append((strat, litho))
    return missing[1:]


@numba.njit
def _fill_in_k(lithostrat, lithology, lookup_table):
    shape = lithostrat.shape
    nlay, nrow, ncol = shape
    out = np.full(shape, np.nan)
    for i in range(nlay):
        for j in range(nrow):
            for k in range(ncol):
                strat = lithostrat[i, j, k]
                if strat != -1:
                    litho = lithology[i, j, k]
                    out[i, j, k] = lookup_table[strat, litho]
    return out


def fill_in_k(lithostrat, lithology, lookup_table):
    missing = _check_litho_strat_combinations(
        lithostrat.values.flatten(), lithology.values.flatten(), lookup_table
    )
    missing = list(set(missing))
    if missing != []:
        msg = "\n".join([f"{a}: {b}" for a, b in missing])
        raise ValueError(
            "Parameter values missing for the following combinations of "
            "lithostratigraphical and lithological units:\n" + msg
        )
    k = xr.full_like(lithostrat, np.nan).astype(np.float64)
    k[...] = _fill_in_k(lithostrat.values, lithology.values, lookup_table)
    return k
