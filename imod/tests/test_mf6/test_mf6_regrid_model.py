
import imod
import xarray as xr
import xugrid as xu
import numpy as np


        
def grid_data_structured(dtype, value, cellsize) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize)
    x = np.arange(0, horizontal_range + cellsize, cellsize)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx": cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims)

    return da

def grid_data_structured_layered(dtype, value, cellsize) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize)
    x = np.arange(0, horizontal_range + cellsize, cellsize)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx": cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype), coords=coords, dims=dims)
    for ilayer in range(1, nlayer + 1):
        layer_value = ilayer * value
        da.loc[dict(layer=ilayer)] = layer_value
    return da

def test_regrid_model():
    cellsize = 2.0

    idomain = grid_data_structured(np.int32, 1, cellsize)


    icelltype = xr.full_like(idomain, 0)
    k = xr.full_like(idomain, 1.0, dtype=float)
    k33 = k.copy()
    rch_rate = xr.full_like(idomain.sel(layer=1), 0.001, dtype=float)
    bottom =grid_data_structured_layered(np.float64, 1.0,cellsize)
    # %%
    # All the data above have been constants over the grid. For the constant head
    # boundary, we'd like to only set values on the external border. We can
    # `py:method:xugrid.UgridDataset.binary_dilation` to easily find these cells:

    chd_location = xr.zeros_like(idomain.sel(layer=2), dtype=bool)
    constant_head = xr.full_like(idomain.sel(layer=2), 1.0, dtype=float).where(chd_location)


    # %%
    # Write the model
    # ---------------
    #
    # The first step is to define an empty model, the parameters and boundary
    # conditions are added in the form of the familiar MODFLOW packages.

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["disv"] = imod.mf6.StructuredDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )
    gwf_model["chd"] = imod.mf6.ConstantHead(
        constant_head, print_input=True, print_flows=True, save_flows=True
    )
    gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        save_flows=True,
    )
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-5,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)

#make new grid
    finer_idomain = grid_data_structured(np.int32, 1, 0.4)
    gwf_model.regrid_like(finer_idomain)
