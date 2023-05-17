
import pytest
import xugrid as xu
import numpy as np
import xarray as xr
import imod 

@pytest.fixture(scope="function")
def disk():
    return xu.data.disk()["face_z"]


def quads(dx):
    xmin, ymin, xmax, ymax = xu.data.disk().ugrid.total_bounds
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    y = np.arange(ymin, ymax, dx) + 0.5 * dx

    da = xr.DataArray(
        data=np.full((y.size, x.size), np.nan),
        coords={"y": y, "x": x},
        dims=[
            "y",
            "x",
        ],
    )
    return xu.UgridDataArray.from_structured(da)




def test_regrid(disk):

    k = disk

    icelltype = disk


    npf = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
        alternative_cell_averaging="AMT-HMK",
    )

    new_npf=npf.regrid_like(quads(1) ) 
