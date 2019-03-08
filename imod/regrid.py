import numba
import numpy as np
import xarray as xr


@numba.njit
def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


@numba.njit
def _starts(src_x, dst_x):
    """
    Calculate regridding weights for a single dimension

    Parameters
    ----------
    src_x : np.array
        vertex coordinates of source
    dst_x: np.array
        vertex coordinates of destination
    """
    i = 0
    j = 0
    while i < dst_x.size - 1:
        x = dst_x[i]
        while j < src_x.size:
            if src_x[j] > x:
                out = max(j - 1, 0)
                yield (i, out)
                break
            else:
                j += 1
        i += 1


@numba.njit
def _weights_1d(src_x, dst_x):
    """
    Calculate regridding weights and indices for a single dimension

    Parameters
    ----------
    src_x : np.array
        vertex coordinates of source
    dst_x: np.array
        vertex coordinates of destination

    Returns
    -------
    max_len : int
        maximum number of source cells to a single destination cell for this
        dimension
    dst_inds : list of int
        destination cell index
    src_inds: list of list of int
        source cell index, per destination index
    weights : list of list of float
        weight of source cell, per destination index
    """
    max_len = 0
    dst_inds = []
    src_inds = []
    weights = []
    # i is index of dst
    # j is index of src
    for i, j in _starts(src_x, dst_x):
        dst_x0 = dst_x[i]
        dst_x1 = dst_x[i + 1]

        _inds = []
        _weights = []
        has_value = False
        while j < src_x.size - 1:
            src_x0 = src_x[j]
            src_x1 = src_x[j + 1]
            overlap = _overlap((dst_x0, dst_x1), (src_x0, src_x1))
            # No longer any overlap, continue to next dst cell
            if overlap == 0:
                break
            else:
                has_value = True
                _inds.append(j)
                _weights.append(overlap)
                j += 1
        if has_value:
            dst_inds.append(i)
            src_inds.append(_inds)
            weights.append(_weights)
            # Save max number of source cells
            # So we know how much to pre-allocate later on
            inds_len = len(_inds)
            if inds_len > max_len:
                max_len = inds_len

    return max_len, dst_inds, src_inds, weights


@numba.njit
def _regrid_3d(src, dst, src_coords, dst_coords, method):
    """
    numba compiled function to regrid in three dimensions

    Parameters
    ----------
    src : np.array
    dst : np.array
    src_coords : tuple of np.arrays of edges
    dst_coords : tuple of np.arrays of edges
    method : numba.njit'ed function
    """
    src_z, src_y, src_x = src_coords
    dst_z, dst_y, dst_x = dst_coords

    max_len_z, ii, blocks_iz, blocks_weights_z = _weights_1d(src_z, dst_z)
    max_len_y, jj, blocks_iy, blocks_weights_y = _weights_1d(src_y, dst_y)
    max_len_x, kk, blocks_ix, blocks_weights_x = _weights_1d(src_x, dst_x)

    # pre-allocate storage
    alloc_len = max_len_z * max_len_y * max_len_x
    values = np.zeros(alloc_len)
    weights = np.zeros(alloc_len)

    # i, j, k are indices of dst array
    # block_i contains indices of src array
    # block_w contains weights of src array
    for i, block_iz, block_wz in zip(ii, blocks_iz, blocks_weights_z):
        for j, block_iy, block_wy in zip(jj, blocks_iy, blocks_weights_y):
            for k, block_ix, block_wx in zip(kk, blocks_ix, blocks_weights_x):
                # Add the values and weights per cell in multi-dim block
                count = 0
                for iz, wz in zip(block_iz, block_wz):
                    for iy, wy in zip(block_iy, block_wy):
                        for ix, wx in zip(block_ix, block_wx):
                            values[count] = src[iz, iy, ix]
                            weights[count] = wz * wy * wx
                            count += 1

                # aggregate
                dst[i, j, k] = method(values[:count], weights[:count])

                # reset storage
                values[:count] = 0
                weights[:count] = 0

    return dst


def _make_regridder(method):
    """
    Closure avoids numba overhead
    https://numba.pydata.org/numba-doc/dev/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
    """
    jit_method = numba.njit(method)

    @numba.njit
    def regridder(src, dst, src_coords, dst_coords):
        return _regrid_3d(src, dst, src_coords, dst_coords, method=jit_method)

    return regridder


def regrid(source, like, method):
    """
    Regridding for axis aligned coordinates
    Does both upscaling and downscaling (downsampling and upsampling, respectively)
    Equidistant and non-equidistant
    Custom aggregation methods can be defined

    """
    src_coords = [coord.values for coord in source["coords"]]
    dst_coords = [coord.values for coord in like["coords"]]

    aggregate = make_aggregate(method)

    dst = _regrid_3d(src, dst, src_coords, dst_coords, method_jit)

    # if src is 3d, regrid is 2d, then apply per layer


def mean(values, weights):
    vsum = 0.0
    wsum = 0.0
    for i in range(values.size):
        v = values[i]
        w = weights[i]
        vsum += w * v
        wsum += w
    return vsum / wsum


def harmonic_mean(values, weights):
    v_agg = 0.0
    w_sum = 0.0
    for i in range(values.size):
        w = weights[i]
        if w > 0:
            w_sum += w
            v_agg += w / values[i]
    return w_sum / v_agg


def geometric_mean(values, weights):
    v_agg = 0.0
    w_sum = 0.0
    m = 0
    for i in range(values.size):
        w = weights[i]
        v = values[i]
        if w > 0:
            v_agg += w * np.log(abs(v))
            w_sum += w
            if v < 0:
                m += 1
    return (-1.0) ** m * np.exp((1.0 / w_sum) * v_agg)


def sum(values, weights):
    v_sum = 0.0
    for i in range(values.size):
        v_sum += values[i] * weights[i]
    return v_sum


def minimum(values, weights):
    return np.min(values)


def maximum(values, weights):
    return np.max(values)


def mode(values, weights):
    ind_mode = np.argmax(np.bincount(values))
    return values[ind_mode]


def median(values, weights):
    return np.percentile(values, 50)
