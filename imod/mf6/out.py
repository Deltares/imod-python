import os
import struct

import dask
import numpy as np
import xarray as xr

import imod


def _grb_text(f, lentxt=50):
    return f.read(lentxt).decode("utf-8").strip().lower()


# Binary Grid File / DIS Grids
# https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=162
def open_disgrb(path):
    with open(path, "rb") as f:
        h1 = _grb_text(f)
        h2 = _grb_text(f)
        if h1 != "grid dis":
            raise ValueError(f'Expected "grid dis" file, got {h1}')
        if h2 != "version 1":
            raise ValueError(f"Only version 1 supported, got {h2}")

        ntxt = int(_grb_text(f).split()[1])
        lentxt = int(_grb_text(f).split()[1])

        # we don't need any information from the the text lines that follow,
        # they are definitions that aim to make the file more portable,
        # so let's skip straight to the binary data
        f.seek(ntxt * lentxt, 1)

        ncells = struct.unpack("i", f.read(4))[0]
        nlayer = struct.unpack("i", f.read(4))[0]
        nrow = struct.unpack("i", f.read(4))[0]
        ncol = struct.unpack("i", f.read(4))[0]
        f.seek(4, 1)  # skip nja
        if ncells != (nlayer * nrow * ncol):
            raise ValueError(f"Invalid file {ncells} {nlayer} {nrow} {ncol}")
        xorigin = struct.unpack("d", f.read(8))[0]
        yorigin = struct.unpack("d", f.read(8))[0]
        f.seek(8, 1)  # skip angrot
        delr = np.fromfile(f, np.float64, nrow)
        delc = np.fromfile(f, np.float64, ncol)
        # TODO verify dimension order
        top_np = np.reshape(np.fromfile(f, np.float64, nrow * ncol), (nrow, ncol))
        bottom_np = np.reshape(np.fromfile(f, np.float64, ncells), (nlayer, nrow, ncol))
        # there is more data below: ia, ja, idomain, icelltype
        # which we don't need for now
        # idomain is not needed since the heads are already marked with nodata values

    bounds = (xorigin, xorigin + delc.sum(), yorigin, yorigin + delr.sum())
    coords = imod.util._xycoords(bounds, (delc, -delr))
    top = xr.DataArray(top_np, coords, ("y", "x"), name="top")
    coords["layer"] = np.arange(1, nlayer + 1)
    bottom = xr.DataArray(bottom_np, coords, ("layer", "y", "x"), name="bottom")

    return {
        "top": top,
        "bottom": bottom,
        "coords": coords,
        "nlayer": nlayer,
        "nrow": nrow,
        "ncol": ncol,
    }


def _to_nan(a, dry_nan):
    a[a == 1e30] = np.nan
    if dry_nan:
        a[a == -1e30] = np.nan
    return a


def _read_hds(path, nlayer, nrow, ncol, dry_nan, pos):
    """
    Reads all values of one timestep.
    """
    n_per_layer = nrow * ncol
    with open(path, "rb") as f:
        f.seek(pos)
        a1d = np.empty(nlayer * nrow * ncol, dtype=np.float64)
        for k in range(nlayer):
            f.seek(52, 1)  # skip kstp, kper, pertime
            a1d[k * n_per_layer : (k + 1) * n_per_layer] = np.fromfile(
                f, np.float64, nrow * ncol
            )

    a3d = a1d.reshape((nlayer, nrow, ncol))
    return _to_nan(a3d, dry_nan)


def _read_times(path, ntime, nlayer, nrow, ncol):
    """
    Reads all total simulation times.
    """
    times = np.empty(ntime, dtype=np.float64)
    # rest of first header and data + other layers of this timestep
    nskip = 28 + nrow * ncol * 8 + (nlayer - 1) * (52 + nrow * ncol * 8)

    with open(path, "rb") as f:
        f.seek(16)
        for i in range(ntime):
            times[i] = struct.unpack("d", f.read(8))[0]  # total simulation time
            f.seek(nskip, 1)
    return times


# Dependent Variable File / DIS Grids
# https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=167
def open_hds(hds_path, grb_path, dry_nan=False):
    """
    Open head data
    """

    d = open_disgrb(grb_path)
    nlayer, nrow, ncol = d["nlayer"], d["nrow"], d["ncol"]
    filesize = os.path.getsize(hds_path)
    ntime = filesize // (nlayer * (52 + (nrow * ncol * 8)))
    times = _read_times(hds_path, ntime, nlayer, nrow, ncol)
    d["coords"]["time"] = times

    dask_list = []
    # loop over times and add delayed arrays
    for i in range(ntime):
        # TODO verify dimension order
        pos = i * (nlayer * (52 + nrow * ncol * 8))
        a = dask.delayed(_read_hds)(hds_path, nlayer, nrow, ncol, dry_nan, pos)
        x = dask.array.from_delayed(a, shape=(nlayer, nrow, ncol), dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    return xr.DataArray(daskarr, d["coords"], ("time", "layer", "y", "x"), name="head")
