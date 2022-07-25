import numpy as np
import xarray as xr

import imod

"""
helper function for creating an xarray dataset of a given type
"""


def get_xarray(dtype):
    shape = nlay, nrow, ncol = 3, 9, 9

    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    ibound = xr.DataArray(np.ones(shape, dtype=dtype), coords=coords, dims=dims)
    return ibound


"""
this function creates zero, one or more instances of a given package type for testing purposes
"""


def get_instances(cls):

    if cls.__name__ == imod.mf6.AdvectionUpstream.__name__:
        return [imod.mf6.AdvectionUpstream()]
    if cls.__name__ == imod.mf6.AdvectionCentral.__name__:
        return [imod.mf6.AdvectionCentral()]
    if cls.__name__ == imod.mf6.AdvectionTVD.__name__:
        return [imod.mf6.AdvectionTVD()]
    if cls.__name__ == imod.mf6.Buoyancy.__name__:

        # TODO:
        # implement when buoyancy has no attributes anymore.
        return []
    if cls.__name__ == imod.mf6.StructuredDiscretization.__name__:
        return [
            imod.mf6.StructuredDiscretization(
                get_xarray(np.float32), get_xarray(np.float32), get_xarray(np.int32)
            )
        ]
    if cls.__name__ == imod.mf6.VerticesDiscretization.__name__:
        # TODO:
        # implement when VerticesDiscretization is loadable from file.
        return []

    if cls.__name__ == imod.mf6.Dispersion.__name__:
        return [imod.mf6.Dispersion(1e-4, 10.0, 10.0, 5.0, 2.0, 4.0, False, True)]
    if cls.__name__ == imod.mf6.InitialConditions.__name__:
        return [imod.mf6.InitialConditions(start=get_xarray(np.float32))]

    if cls.__name__ == imod.mf6.Solution.__name__:
        return [
            imod.mf6.Solution(
                print_option="summary",
                csv_output=False,
                no_ptc=True,
                outer_dvclose=1.0e-4,
                outer_maximum=500,
                under_relaxation=None,
                inner_dvclose=1.0e-4,
                inner_rclose=0.001,
                inner_maximum=100,
                linear_acceleration="cg",
                scaling_method=None,
                reordering_method=None,
                relaxation_factor=0.97,
            )
        ]
    if cls.__name__ == imod.mf6.MobileStorageTransfer.__name__:
        return [imod.mf6.MobileStorageTransfer(0.35, 0.01, 0.02, 1300.0, 0.1)]
    if cls.__name__ == imod.mf6.NodePropertyFlow.__name__:
        return [
            imod.mf6.NodePropertyFlow(get_xarray(np.int32), 3.0, True, 32.0, 34.0, 7)
        ]
    if cls.__name__ == imod.mf6.OutputControl.__name__:
        # TODO:
        # implement when OutputControl can be loaded from file.
        return []
    if cls.__name__ == imod.mf6.Storage.__name__:
        return []  # deprecated
    if cls.__name__ == imod.mf6.SpecificStorage.__name__:
        return [imod.mf6.SpecificStorage(0.001, 0.1, True, get_xarray(np.int32))]
    if cls.__name__ == imod.mf6.StorageCoefficient.__name__:
        return [imod.mf6.StorageCoefficient(0.001, 0.1, True, get_xarray(np.int32))]
    if cls.__name__ == imod.mf6.TimeDiscretization.__name__:
        # return imod.mf6.TimeDiscretization(10.0, 23, 1.02)
        # TODO:
        # implement when TimeDiscretization can be written to file after initialization.
        return []
    else:
        print(f"could not create an instance of class {cls.__name__}")
        assert False
