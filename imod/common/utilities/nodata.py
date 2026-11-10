"""Utilities to report NoData where valid values are required."""

import numpy as np

from imod.typing import GridDataArray, GridDataDict

MAX_REPORTED_LOCATIONS = 5


def describe_source(da: GridDataArray | None) -> str:
    """
    Describe where data came from, based on the paths the project file reader
    attached to the array. Constants and processed (e.g. regridded) data have
    no paths.
    """
    sources = getattr(da, "attrs", {}).get("source_paths")
    if sources:
        return ", ".join(str(s) for s in sources)
    return "unknown (constant value or processed data)"


def _describe_locations(nodata: GridDataArray, n_max: int) -> str:
    """Format the (x, y) of the first ``n_max`` True cells, if known."""
    if "layer" in nodata.dims and nodata.sizes["layer"] == 1:
        nodata = nodata.isel(layer=0, drop=True)
    if set(nodata.dims) != {"y", "x"}:
        return ""
    values = nodata.transpose("y", "x").to_numpy()
    rows, cols = np.nonzero(values)
    x = nodata["x"].to_numpy()[cols[:n_max]]
    y = nodata["y"].to_numpy()[rows[:n_max]]
    locations = ", ".join(f"({xi}, {yi})" for xi, yi in zip(x, y, strict=True))
    suffix = ", ..." if len(rows) > n_max else ""
    return f" First (x, y): {locations}{suffix}."


def describe_nodata(
    name: str,
    data: GridDataDict,
    required: GridDataArray,
    source_data: GridDataDict | None = None,
    n_max_locations: int = MAX_REPORTED_LOCATIONS,
) -> str | None:
    """
    Describe NoData in ``data[name]`` in cells where ``required`` is True.

    Parameters
    ----------
    name: str
        Key of the variable to check in ``data``.
    data: GridDataDict
        Data to check. Variables that are not present are skipped.
    required: GridDataArray
        Boolean grid, True where ``data[name]`` must have a valid value.
    source_data: GridDataDict, optional
        Data to retrieve the source file paths from, as these get lost when
        processing data. Defaults to ``data``.
    n_max_locations: int
        Maximum number of cell locations to report.

    Returns
    -------
    str or None
        Description with variable name, number of cells, first cell locations
        and source file(s). ``None`` if there is no NoData in required cells.
    """
    da = data.get(name)
    if da is None:
        return None
    if source_data is None:
        source_data = data
    nodata = da.isnull() & required
    n_nodata = int(nodata.sum())
    if n_nodata == 0:
        return None
    return (
        f"'{name}': {n_nodata} cell(s) with NoData."
        f"{_describe_locations(nodata, n_max_locations)}"
        f" Source: {describe_source(source_data.get(name))}"
    )
