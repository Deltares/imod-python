import warnings

import numpy as np
import pandas as pd
import xarray as xr


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
        freshwaterhead = freshwaterhead.where(pointwaterhead > elevation)
        freshwaterhead = freshwaterhead.bfill(dim="layer")

    return freshwaterhead


def _calculate_gxg(
    head_bimonthly: xr.DataArray, below_surfacelevel: bool = False
) -> xr.DataArray:
    import bottleneck as bn

    # Most efficient way of finding the three highest and three lowest is via a
    # partition. See:
    # https://bottleneck.readthedocs.io/en/latest/reference.html#bottleneck.partition

    def lowest3_mean(da: xr.DataArray):
        a = bn.partition(da.values, kth=2, axis=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanmean(a[..., :3], axis=-1)

        template = da.isel(bimonth=0)
        return template.copy(data=result)

    def highest3_mean(da: xr.DataArray):
        a = bn.partition(-da.values, kth=2, axis=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = np.nanmean(-a[..., :3], axis=-1)

        template = da.isel(bimonth=0)
        return template.copy(data=result)

    timesize = head_bimonthly["time"].size
    if timesize % 24 != 0:
        raise ValueError("head is not bimonthly for a full set of years")
    n_year = int(timesize / 24)

    # First and second date of March: 4, 5; first date of April: 6.
    month_index = np.array([4, 5, 6])
    # Repeat this for every year in dataset, and increment by 24 per repetition.
    yearly_increments = (np.arange(n_year) * 24)[:, np.newaxis]
    # Broadcast to a full set
    gvg_index = xr.DataArray(
        data=(month_index + yearly_increments), dims=("hydroyear", "bimonth")
    )
    gvg_data = head_bimonthly.isel(time=gvg_index)
    # Filters years without 3 available measurments.
    gvg_years = gvg_data.count("bimonth") == 3
    gvg_data = gvg_data.where(gvg_years)

    # Hydrological years: running from 1 April to 1 April in the Netherlands.
    # Increment run from April (6th date) to April (30th date) for every year.
    # Broadcast to a full set
    newdims = ("hydroyear", "bimonth")
    gxg_index = xr.DataArray(
        data=(np.arange(6, 30) + yearly_increments[:-1]),
        dims=newdims,
    )
    gxg_data = head_bimonthly.isel(time=gxg_index)
    dims = [dim for dim in gxg_data.dims if dim not in newdims]
    dims.extend(newdims)
    gxg_data = gxg_data.transpose(*dims)

    # Filter years without 24 measurements.
    gxg_years = gxg_data.count("bimonth") == 24
    gxg_data = gxg_data.where(gxg_years)

    # First compute LG3 and HG3 per hydrological year, then compute the mean over the total.
    if gxg_data.chunks is not None:
        # If data is lazily loaded/chunked, process data of one year at a time.
        gxg_data = gxg_data.chunk({"hydroyear": 1})
        lg3 = xr.map_blocks(lowest3_mean, gxg_data, template=gxg_data.isel(bimonth=0))
        hg3 = xr.map_blocks(highest3_mean, gxg_data, template=gxg_data.isel(bimonth=0))
    else:
        # Otherwise, just compute it in a single go.
        lg3 = lowest3_mean(gxg_data)
        hg3 = highest3_mean(gxg_data)

    gxg = xr.Dataset()
    gxg["gvg"] = gvg_data.mean(("hydroyear", "bimonth"))

    ghg = hg3.mean("hydroyear")
    glg = lg3.mean("hydroyear")
    if below_surfacelevel:
        gxg["glg"] = ghg
        gxg["ghg"] = glg
    else:
        gxg["glg"] = glg
        gxg["ghg"] = ghg

    # Add the numbers of years used in the calculation
    gxg["n_years_gvg"] = gvg_years.sum("hydroyear")
    gxg["n_years_gxg"] = gxg_years.sum("hydroyear")
    return gxg


def calculate_gxg_points(
    df: pd.DataFrame,
    id: str = "id",
    time: str = "time",
    head: str = "head",
    below_surfacelevel: bool = False,
    tolerance: pd.Timedelta = pd.Timedelta(days=7),
) -> pd.DataFrame:
    """
    Calculate GxG groundwater characteristics from head time series.

    GLG and GHG (average lowest and average highest groundwater level respectively) are
    calculated as the average of the three lowest (GLG) or highest (GHG) head values per
    Dutch hydrological year (april - april), for head values measured at a semi-monthly frequency
    (14th and 28th of every month). GVG (average spring groundwater level) is calculated as
    the average of groundwater level on 14th and 28th of March, and 14th of April. Supplied head
    values are resampled (nearest) to the 14/28 frequency.

    Hydrological years without all 24 14/28 dates present are discarded for glg and ghg.
    Years without the 3 dates for gvg are discarded.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the piezometer IDs, dates of measurement, and
        measured heads.
    id: str
        Column name of piezometer ID.
    time: str
        Column name of datetime.
    head: str
        Column name of head measurement.
    below_surfacelevel: bool
        False (default) if heads are relative to a datum (e.g. sea level). If
        True, heads are taken as m below surface level.
    tolerance: pd.Timedelta, default: 7 days.
        Maximum time window allowed when searching for dates around the 14th
        and 28th of every month.

    Returns
    -------
    gxg : pd.DataFrame
        Dataframe containing ``glg``: average lowest head, ``ghg``: average
        highest head, ``gvg``: average spring head, ``n_years_gvg``: numbers of
        years used for gvg, ``n_years_gxg``: numbers of years used for glg and
        ghg.

    Examples
    --------

    Read some IPF data and compute the GxG values, while specifying the
    (non-standard) column names:

    >>> import imod
    >>> df = imod.ipf.read("piezometers.ipf")
    >>> gxg = imod.evaluate.calculate_gxg_points(
    >>>     df=df,
    >>>     id="ID-column",
    >>>     time="Date",
    >>>     head="Piezometer head (m)",
    >>> )
    """

    def bimonthly(series: pd.Series, dates: pd.DatetimeIndex, tolerance: pd.Timedelta):
        series = series[~series.index.duplicated(keep="first")]
        return series.reindex(dates, method="nearest", tolerance=tolerance)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, received: {type(df).__name__}")
    for name in (id, time, head):
        if name not in df:
            raise ValueError(f"Column {name} not present in dataframe")

    # Create a bi-monthly date range
    start = f"{df[time].min().year}-01-01"
    end = f"{df[time].max().year}-12-31"
    dates = pd.date_range(
        start=start, end=end, freq="SMS", name="time"
    ) + pd.DateOffset(days=13)

    # Convert for every location the time series to the same date range. This
    # forms a rectangular array which we can represent directly as a DataArray.
    bimonthly_series = (
        df.set_index(time).groupby(id)[head].apply(bimonthly, dates, tolerance)
    )
    head_bimonthly = bimonthly_series.to_xarray()

    # Calculate GXG values per location
    gxg = _calculate_gxg(head_bimonthly, below_surfacelevel)
    # Transform back to a DataFrame
    return gxg.to_dataframe()


def calculate_gxg(
    head: xr.DataArray,
    below_surfacelevel: bool = False,
    tolerance: pd.Timedelta = pd.Timedelta(days=7),
) -> xr.DataArray:
    """
    Calculate GxG groundwater characteristics from head time series.

    GLG and GHG (average lowest and average highest groundwater level respectively) are
    calculated as the average of the three lowest (GLG) or highest (GHG) head values per
    Dutch hydrological year (april - april), for head values measured at a semi-monthly frequency
    (14th and 28th of every month). GVG (average spring groundwater level) is calculated as
    the average of groundwater level on 14th and 28th of March, and 14th of April. Supplied head
    values are resampled (nearest) to the 14/28 frequency.

    Hydrological years without all 24 14/28 dates present are discarded for glg and ghg.
    Years without the 3 dates for gvg are discarded.

    Parameters
    ----------
    head : xr.DataArray of floats
        Head relative to sea level, in m, or m below surface level if `below_surfacelevel` is
        set to True. Must be of dimensions ``("time", "y", "x")``.
    below_surfacelevel : boolean, optional, default: False.
        False (default) if heads are relative to a datum (e.g. sea level). If
        True, heads are taken as m below surface level.
    tolerance: pd.Timedelta, default: 7 days.
        Maximum time window allowed when searching for dates around the 14th
        and 28th of every month.

    Returns
    -------
    gxg : xr.Dataset
        Dataset containing ``glg``: average lowest head, ``ghg``: average
        highest head, ``gvg``: average spring head, ``n_years_gvg``: numbers of
        years used for gvg, ``n_years_gxg``: numbers of years used for glg and
        ghg.

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
    if not head.dims == ("time", "y", "x"):
        raise ValueError('Dimensions must be ("time", "y", "x")')
    if not np.issubdtype(head["time"].dtype, np.datetime64):
        raise ValueError("Time must have dtype numpy datetime64")

    # Reindex to GxG frequency date_range: every 14th and 28th of the month.
    start = f"{int(head['time'][0].dt.year)}-01-01"
    end = f"{int(head['time'][-1].dt.year)}-12-31"
    dates = pd.date_range(start=start, end=end, freq="SMS") + pd.DateOffset(days=13)
    head_bimonthly = head.reindex(time=dates, method="nearest", tolerance=tolerance)

    gxg = _calculate_gxg(head_bimonthly, below_surfacelevel)
    return gxg
