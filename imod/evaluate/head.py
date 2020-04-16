import xarray as xr
import pandas as pd
import numpy as np


def convert_pointwaterhead_freshwaterhead(
    pointwaterhead, density, elevation, density_fresh=1000.0
):
    r"""Function to convert point water head (as outputted by seawat)
    into freshwater head, using Eq.3 from Guo, W., & Langevin, C. D. (2002):

    .. math:: h_{f}=\frac{\rho}{\rho_{f}}h-\frac{\rho-\rho_{f}}{\rho_{f}}Z

    An edge case arises when the head is below the cell centre, or entirely below
    the cell. Strictly applying Eq.3 would result in freshwater heads that are
    lower than the original point water head, which is physically impossible. This
    function then outputs the freshwaterhead for the uppermost underlying cell where
    the original point water head exceeds the cell centre.

    *Requires bottleneck.*

    Parameters
    ----------
    pointwaterhead : float or xr.DataArray of floats
        the point water head as outputted by SEAWAT, in m.
    density : float or xr.DataArray of floats
        the water density at the same locations as `pointwaterhead`. 
    elevation : float or xr.DataArray of floats
        elevation at the same locations as `pointwaterhead`, in m. 
    density_fresh : float, optional
        the density of freshwater (1000 kg/m3), or a different value if 
        different units are used, or a different density reference is required.

    Returns
    -------
    freshwaterhead : float or xr.DataArray of floats
    """

    freshwaterhead = (
        density / density_fresh * pointwaterhead
        - (density - density_fresh) / density_fresh * elevation
    )

    # edge case: point water head below z
    # return freshwater head of top underlying cell where elevation < pointwaterhead
    # only for xr.DataArrays
    if isinstance(pointwaterhead, xr.DataArray) and "layer" in pointwaterhead.dims:
        freshwaterhead = freshwaterhead.where(pointwaterhead > elevation).compute()
        freshwaterhead = freshwaterhead.bfill(dim="layer")

    return freshwaterhead


def calculate_gxg(head, below_surfacelevel=False):
    """Function to calculate GxG groundwater characteristics from head time series.

    GLG and GHG (average lowest and average highest groundwater level respectively) are
    calculated as the average of the three lowest (GLG) or highest (GHG) head values per
    hydrological year (april - april), for head values measured at a semi-monthly frequency
    (14th and 28th of every month). GVG (average spring groundwater level) is calculated as
    the average of groundwater level on 28th of March, 14th and 28th of April. Supplied head 
    values are resampled (nearest) to the 14/28 frequency. Hydrological years without all 24 
    14/28 dates present are discarded.

    *Requires bottleneck.*

    Parameters
    ----------
    head : xr.DataArray of floats
        Head relative to sea level, in m, or m below surface level if `below_surfacelevel` is 
        set to True. Must be of dimensions ``("time", "y", "x")``.
    below_surfacelevel : boolean, optional
        False (default) if heads are relative to sea level. If True, heads are taken as m
        below surface level.

    Returns
    -------
    gxg : xr.Dataset 
        Dataset containing `glg`: average lowest head, `ghg`: average highest head, and `gvg`:
        average spring head.

    Examples
    --------
    Load the heads, and calculate groundwater characteristics after the year 2000:

    >>> import imod
    >>> heads = imod.idf.open("head*.idf").sel(time=heads.time.dt.year >= 2000)
    >>> gxg = imod.evaluate.calculate_gxg(heads)

    Transform to meters below surface level by substracting from surface level:

    >>> surflevel = imod.idf.open("surfacelevel.idf")
    >>> gxg = surflevel - gxg

    Or calculate from groundwater level relative to surface level directly:

    >>> gwl = surflevel - heads
    >>> gxg = imod.evaluate.calculate_gxg(gwl, below_surfacelevel=True)

    """
    assert head.ndim == 3
    assert "time" in head.dims
    assert "x" in head.dims
    assert "y" in head.dims

    # reindex to GxG frequency date_range: every 14th and 28th of the month
    d13 = pd.DateOffset(days=13)
    dr_gxg = (
        pd.date_range(
            start=pd.Timestamp(head.time.values[0]) - d13,
            end=pd.Timestamp(head.time.values[-1]) - d13,
            freq="SMS",
        )
        + d13
    )
    da_gxg = head.reindex(time=dr_gxg, method="nearest")

    # calculate LG3, HG3 per hydrological year
    dr_hydroyear = pd.date_range(
        start=dr_gxg[0] - pd.DateOffset(months=1),
        end=dr_gxg[-1] + pd.DateOffset(months=1),
        freq="AS-APR",
    )
    annual = da_gxg.groupby_bins("time", bins=dr_hydroyear)
    perturb = np.random.random((24, *head.isel(time=0).shape)) * 1e-6
    glg = []
    ghg = []
    gvg = []
    years = []
    for y, da in annual:
        if len(da.time) == 24:  # only include full-years
            da = da.load() + perturb  # perturb to assure integer ranks
            dar = da.rank(dim="time")
            maxrank = dar.max()  # to allow for nodata values within timeseries
            lg3 = da.where(dar <= 3).mean(dim="time")
            hg3 = da.where(dar > maxrank - 3).mean(dim="time")
            vg3 = da.isel(time=[0, 1, -1]).mean(dim="time")
            glg.append(lg3)
            ghg.append(hg3)
            gvg.append(vg3)
            years.append(y)

    # average
    glg = xr.concat(glg, dim="time").mean(dim="time")
    ghg = xr.concat(ghg, dim="time").mean(dim="time")
    gvg = xr.concat(gvg, dim="time").mean(dim="time")

    gxg = xr.Dataset()
    if not below_surfacelevel:
        gxg["glg"] = glg
        gxg["ghg"] = ghg
    else:
        gxg["glg"] = ghg
        gxg["ghg"] = glg
    gxg["gvg"] = gvg

    gxg = gxg.assign_attrs(
        GxG_period=f"{years[0].left:%Y-%m-%d} - {years[-1].right:%Y-%m-%d}"
    )
    return gxg
