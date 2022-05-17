import numpy as np
import pytest
import xarray as xr

def get_data_array(nlay, nrow, ncol, globaltimes, dx, dy, xmin, ymin):
    ntimes = len(globaltimes)
    shape = (ntimes, nlay, nrow, ncol)        
    dims = ("time", "layer", "y", "x")

    layer = np.array([1, 2, 3])
    xmax = dx * ncol
    ymax = abs(dy) * nrow    
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"time": globaltimes, "layer": layer, "y": y, "x": x}

    # Discretization data
    return xr.DataArray(np.ones(shape), coords=coords, dims=dims, )    

@pytest.fixture(scope="session")
def head_fc():

    globaltimes = [np.datetime64("2000-01-01"),np.datetime64("2000-01-02"),np.datetime64("2000-01-03")]
    idomain = get_data_array(3, 15, 15, globaltimes, 5000, -5000, 0, 0)

    # Constant head
    head = xr.full_like(idomain, np.nan)
    return head

@pytest.fixture(scope="session")
def concentration_fc():

    globaltimes = [np.datetime64("2000-01-01"),np.datetime64("2000-01-02"),np.datetime64("2000-01-03")]
    idomain = get_data_array(3, 15, 15, globaltimes, 5000, -5000, 0, 0)
    idomain = idomain.expand_dims(species = ["salinity",  "temperature"])

    concentration = xr.full_like(idomain, np.nan)
    return concentration
