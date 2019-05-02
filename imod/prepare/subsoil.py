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
    assert lithoclass.shape == lithostrat.shape == values.shape
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
    for (strat, litho) in zip(lithostrat.flatten(), lithology.flatten()):
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


@numba.njit
def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


@numba.njit
def _formation_number(tops, bots, ztop, zbot, out):
    nlay, nrow, ncol = out.shape
    nformations = tops.shape[0]
    for i in range(nlay):
        for j in range(nrow):
            for k in range(ncol):
                zt = ztop[i]
                zb = zbot[i]
                maximum_overlap = 0.0
                formation = -2147483648
                for nf in range(nformations):
                    t = tops[nf, j, k]
                    b = bots[nf, j, k]
                    if np.isnan(t) or np.isnan(b):
                        continue
                    else:
                        overlap = _overlap((zb, zt), (b, t))
                        if overlap > maximum_overlap:
                            maximum_overlap = overlap
                            formation = nf
                out[i, j, k] = formation
    return out


@numba.njit
def _fill_in_by_formation(formation, kh, kv):
    nlay, nrow, ncol = formation.shape
    kh_out = np.full((nlay, nrow, ncol), np.nan)
    kv_out = np.full((nlay, nrow, ncol), np.nan)
    for i in range(nlay):
        for j in range(nrow):
            for k in range(ncol):
                f = formation[i, j, k]
                if f < 0:
                    continue
                else:
                    # lookup the appropriate value
                    khf = kh[f, j, k]
                    kvf = kv[f, j, k]
                    # Since GeoTOP has only kh or kv
                    # and we need both
                    if np.isnan(khf):
                        khf = kvf
                    if np.isnan(kvf):
                        kvf = khf
                    kh_out[i, j, k] = khf
                    kv_out[i, j, k] = kvf
    return kh_out, kv_out


def voxelize(regis, like, dz):
    top = regis["top"]
    bot = regis["bot"]
    kh = regis["kh"]
    kv = regis["kv"]
    # Checks on inputs
    assert top.dims == bot.dims == kh.dims == kv.dims
    assert top.dims == ("formation", "y", "x")
    assert like.dims == ("z", "y", "x")
    assert (np.diff(like.coords["z"]) == 0.50).all()
    assert np.array_equal(top.coords["y"], like.coords["y"])
    assert np.array_equal(top.coords["x"], like.coords["x"])

    formation = np.full_like(like.values, -2147483648, dtype=np.int)
    ztop = like.coords["z"] + 0.25
    zbot = like.coords["z"] - 0.25
    formation = _formation_number(
        top.values, bot.values, ztop.values, zbot.values, formation
    )
    kh_out, kv_out = _fill_in_by_formation(formation, kh.values, kv.values)
    return xr.full_like(like, kh_out), xr.full_like(like, kv_out)
