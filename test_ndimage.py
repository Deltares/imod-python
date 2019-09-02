import copy
import pathlib

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imod

plt.plot(np.arange(10.0))


def imshow(a_yx, fname, title, cmap, overlays, quantile_colorscale, figsize, settings):
    fig, ax = plt.subplots(figsize=figsize)
    if quantile_colorscale:
        levels = np.unique(np.percentile(a_yx.values, np.linspace(0, 100, 101)))
        if levels.size < 3:  # let matplotlib take care of it
            norm = None
        else:
            norm = matplotlib.colors.BoundaryNorm(levels, 256)
    else:
        norm = None
    ax1 = ax.imshow(a_yx, cmap=cmap, norm=norm, **settings)
    for overlay in overlays:
        tmp = overlay.copy()
        gdf = tmp.pop("gdf")
        gdf.plot(ax=ax, **tmp)

    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(ax1, cmap=cmap, norm=norm, cax=cbar_ax)

    ax.set_title(title)
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def nd_imshow(
    da,
    name,
    directory=".",
    cmap="viridis",
    overlays=[],
    quantile_colorscale=True,
    figsize=(8, 8),
):
    """
    Automatically colors by quantile.

    Dumps PNGs into directory of choice.
    """
    if "x" not in da.dims or "y" not in da.dims:
        raise ValueError("DataArray must have dims x and y.")
    directory = pathlib.Path(directory)
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(da)
    settings = {"interpolation": "nearest", "extent": [xmin, xmax, ymin, ymax]}
    extradims = [dim for dim in da.dims if dim not in ("x", "y")]

    if len(extradims) == 0:
        fname = directory / f"{name}.png"
        imshow(da, fname, name, cmap, overlays, quantile_colorscale, figsize, settings)
    else:
        stacked = da.stack(idf=extradims)
        for coordvals, a_yx in list(stacked.groupby("idf")):

            fname_parts = []
            title_parts = []
            for key, coordval in zip(extradims, coordvals):
                title_parts.append(f"{key}: {coordval}")
                if key == "time":
                    if isinstance(coordval, np.datetime64):
                        # The following line is because numpy.datetime64[ns] does not
                        # support converting to datetime, but returns an integer instead.
                        # This solution is 20 times faster than using pd.to_datetime()
                        coordval = (
                            coordval.astype("datetime64[us]")
                            .item()
                            .strftime("%Y%m%d%H%M%S")
                        )
                    else:
                        coordval = coordval.strftime("%Y%m%d%H%M%S")
                fname_parts.append(f"{key}{coordval}")

            fname_parts = "_".join(fname_parts)
            title_parts = ", ".join(title_parts)
            fname = directory / f"{name}_{fname_parts}.png"
            title = f"{name}, {title_parts}"
            imshow(
                a_yx,
                fname,
                title,
                cmap,
                overlays,
                quantile_colorscale,
                figsize,
                settings,
            )


kd = imod.idf.open(r"c:\projects\imodx\NHI_stationary\model_v3.3.0\dbase\kd\kd_l*.idf")
# kd = kd.squeeze("layer", drop=True)

times = [np.datetime64(f"200{y}-01-01") for y in range(5)]
kd = xr.concat([kd.assign_coords(time=time) for time in times], dim="time")
nd_imshow(kd, "kD")
