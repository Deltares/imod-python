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
def _fill_in_by_formation(formation, src):
    nlay, nrow, ncol = formation.shape
    dst = np.full((nlay, nrow, ncol), np.nan)
    for i in range(nlay):
        for j in range(nrow):
            for k in range(ncol):
                f = formation[i, j, k]
                if f < 0:
                    continue
                else:
                    # look up the appropriate value
                    v = src[f, j, k]
                    dst[i, j, k] = v
    return dst


def voxelize(top, bottom, like, *parameters):
    """
    Turn a layer model into a voxel model.

    Parameters
    ----------
    top: xr.DataArray
        Top elevation of layers
        with dimensions ("layer", "y", "x")
    bottom: xr.DataArray
        Bottom elevaion of layers
        with dimensions ("layer", "y", "x")
    like: xr.DataArray
        Example of what the output should look like.
        with dimensions ("z", "y", "x); or ("layer", "y", "x") with
        coordinate "z": ("layer", z)
    parameters: xr.DataArray(s)
        variable number of parameters to voxelize with dimensions
        ("layer", "y", "x")

    Returns
    -------
    voxelized parameters : single, or tuple of, xr.DataArray
        Have identical dimensions and coordinates as `like`.
    """

    def dim_format(dims):
        return ", ".join(dim for dim in dims)

    # Checks on inputs
    if not like.dims == ("z", "y", "x"):
        if like.dims == ("layer", "y", "x"):
            if not "z" in like.coords:
                raise ValueError('"z" has to be given in `like` coordinates')
        else:
            raise ValueError(
                '`like` coordinates need to be exactly ("z", "y", "x"); or'
                ' ("layer", "y", "x") with coordinate "z": ("layer", z).'
                f" Got instead: {dim_format(like.dims)}."
            )
    if "dz" not in like.coords:
        dzs = np.diff(like.coords["z"].values)
        dz = dzs[0]
        if not np.allclose(dzs, dz):
            raise ValueError(
                '"dz" has to be given as a coordinate in case of'
                ' non-equidistant "z" coordinate.'
            )
        like["dz"] = dz
    for da in [top, bottom, *parameters]:
        if not da.dims == ("layer", "y", "x"):
            raise ValueError(
                "Dimensions for top, bottom, and parameters have to be exactly"
                f' ("layer", "y", "x"). Got instead {dim_format(da.dims)}.'
            )
    for da in [bottom, *parameters]:
        for (k1, v1), (_, v2) in zip(top.coords.items(), da.coords.items()):
            if not v1.equals(v2):
                raise ValueError(f"Input coordinates do not match along {k1}")

    formation = np.full_like(like.values, -2147483648, dtype=np.int)
    ztop = like.coords["z"] + 0.5 * np.abs(like.coords["dz"])
    zbot = like.coords["z"] - 0.5 * np.abs(like.coords["dz"])
    formation = _formation_number(
        top.values, bottom.values, ztop.values, zbot.values, formation
    )
    arrays = [p.values for p in parameters]
    voxelized = [_fill_in_by_formation(formation, a) for a in arrays]

    if len(voxelized) > 1:
        return tuple(xr.full_like(like, a) for a in voxelized)
    else:
        return xr.full_like(like, voxelized[0])
