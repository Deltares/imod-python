from turtle import st
from types import NotImplementedType

import numpy as np
import xarray as xr
from matplotlib.pyplot import fill

from imod.mf6.lake_package import lake_api


def create_idomain(nlay, nrow, ncol):
    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 25.0
    dy = -25.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    # Discretization data
    return xr.DataArray(np.ones(shape, dtype=np.int8), coords=coords, dims=dims)


def create_gridcovering_array(idomain, lake_cells, fillvalue, dtype):
    result = xr.full_like(
        idomain, fill_value=lake_api.missing_values[np.dtype(dtype).name], dtype=dtype
    )
    for cell in lake_cells:
        result.values[cell[0], cell[1], cell[2]] = fillvalue
    return result


def create_lakelake(idomain, starting_stage, boundname, lake_cells):
    connectionType = create_gridcovering_array(
        idomain, lake_cells, lake_api.connection_types["HORIZONTAL"], np.int32
    )
    bed_leak = create_gridcovering_array(idomain, lake_cells, 0.2, np.float32)
    top_elevation = create_gridcovering_array(idomain, lake_cells, 0.3, np.float32)
    bot_elevation = create_gridcovering_array(idomain, lake_cells, 0.4, np.float32)
    connection_length = create_gridcovering_array(idomain, lake_cells, 0.5, np.float32)
    connection_width = create_gridcovering_array(idomain, lake_cells, 0.6, np.float32)
    result = lake_api.LakeLake(
        starting_stage,
        boundname,
        connectionType,
        bed_leak,
        top_elevation,
        bot_elevation,
        connection_length,
        connection_width,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    return result


def concatenate_xr_dataArrays(object_list, propertyName, lead_coord, dtype):
    outputArray_size = 0
    index_in_out_array = 0

    for object in object_list:
        for property, value in vars(object).items():
            if property == propertyName:
                outputArray_size += len(value.coords[lead_coord])
    dims = lead_coord
    coords = {lead_coord: list(range(0, outputArray_size))}
    shape = [outputArray_size]
    outputArray = xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims, coords=coords)

    for object in object_list:
        for property, value in vars(object).items():
            if property == propertyName:
                nrvalues = len(value.coords[lead_coord])
                for ivalue in range(0, nrvalues):
                    outputArray.values[index_in_out_array] = value.data[ivalue]
                    index_in_out_array += 1

    return outputArray


nconnect = 10
dimensions = ["connection_nr"]
coordinates = {"connection_nr": np.arange(0, nconnect)}
rate1 = xr.DataArray(
    np.ones(nconnect, dtype=np.float32), coords=coordinates, dims=dimensions
)
rate2 = xr.DataArray(
    np.zeros(nconnect, dtype=np.float32), coords=coordinates, dims=dimensions
)
outlet1 = lake_api.OutletSpecified(1, 1, 2, rate1)
outlet2 = lake_api.OutletSpecified(2, 2, 1, rate2)

result = concatenate_xr_dataArrays([outlet1, outlet2], "rate", "connection_nr", float)

idomain = create_idomain(4, 10, 10)
lake1 = create_lakelake(idomain, 11, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)])
lake2 = create_lakelake(
    idomain, 15, "Ijsselmeer", [(1, 12, 12), (1, 12, 13), (1, 13, 13)]
)
lake_api.from_lakes_and_outlets([lake1, lake2])
