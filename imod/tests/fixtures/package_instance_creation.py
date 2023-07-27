import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu

import imod
import imod.tests.fixtures.mf6_lake_package_fixture as mf_lake

"""
This file is used to create instances of imod packages for testing purposes.
The main usage is importing ALL_PACKAGE_INSTANCES into a test- this list contains an instance of
each packages and boundary condition in mf6.
"""


def get_structured_grid_da(dtype, value=1):
    """
    This function creates a dataarray with scalar values for a grid of 3 layers and 9 rows and columns.
    """
    shape = nlay, nrow, ncol = 3, 9, 9
    dims = ("layer", "y", "x")

    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    da = xr.DataArray(np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims)
    return da


def get_unstructured_grid_da(dtype, value=1):
    """
    This function creates an xugrid dataarray with scalar values for an unstructured grid
    """
    grid = imod.data.circle()
    nface = grid.n_face
    nlayer = 2

    dims = ("layer", grid.face_dimension)
    shape = (nlayer, nface)

    uda = xu.UgridDataArray(
        xr.DataArray(
            np.ones(shape, dtype=dtype) * value,
            coords={"layer": [1, 2]},
            dims=dims,
        ),
        grid=grid,
    )
    return uda


def get_grid_da(is_unstructured, dtype, value=1):
    """
    helper function for creating an xarray dataset of a given type
    Depending on the is_unstructured input parameter, it will create an array for a
    structured grid or for an unstructured grid.
    """

    if is_unstructured:
        return get_unstructured_grid_da(dtype, value)
    else:
        return get_structured_grid_da(dtype, value)


def create_lake_package(is_unstructured):
    is_lake1 = get_grid_da(is_unstructured, bool, False)
    times_of_numeric_timeseries = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-03-01"),
        np.datetime64("2000-05-01"),
    ]
    numeric = xr.DataArray(
        np.full((len(times_of_numeric_timeseries)), 5.0),
        coords={"time": times_of_numeric_timeseries},
        dims=["time"],
    )
    if is_unstructured:
        is_lake1[1, 0] = True
        is_lake1[1, 1] = True
        lake1 = mf_lake.create_lake_data_unstructured(
            is_lake1, 11.0, "Naardermeer", stage=numeric
        )
    else:
        is_lake1[1, 1, 0] = True
        is_lake1[1, 1, 1] = True
        lake1 = mf_lake.create_lake_data_structured(
            is_lake1, 11.0, "Naardermeer", stage=numeric
        )
    outlet1 = imod.mf6.OutletManning("Naardermeer", "", 3.0, 2.0, 3.0, 4.0)
    return imod.mf6.Lake.from_lakes_and_outlets([lake1], [outlet1])


def create_vertices_discretization():
    """
    return imod.mf6.VerticesDiscretization object
    """
    idomain = get_grid_da(True, int, value=1)
    bottom = idomain * xr.DataArray([5.0, 0.0], dims=["layer"])
    return imod.mf6.VerticesDiscretization(top=10.0, bottom=bottom, idomain=idomain)


def create_instance_packages(is_unstructured):
    """
    creates instances of those modflow packages that are not boundary conditions.
    """
    return [
        imod.mf6.Dispersion(
            diffusion_coefficient=get_grid_da(is_unstructured, np.float32, 1e-4),
            longitudinal_horizontal=get_grid_da(is_unstructured, np.float32, 10),
            transversal_horizontal1=get_grid_da(is_unstructured, np.float32, 10),
            longitudinal_vertical=get_grid_da(is_unstructured, np.float32, 5),
            transversal_horizontal2=get_grid_da(is_unstructured, np.float32, 2),
            transversal_vertical=get_grid_da(is_unstructured, np.float32, 4),
        ),
        imod.mf6.InitialConditions(start=get_grid_da(is_unstructured, np.float32)),
        imod.mf6.MobileStorageTransfer(
            porosity=get_grid_da(is_unstructured, np.float32, 0.35),
            decay=get_grid_da(is_unstructured, np.float32, 0.01),
            decay_sorbed=get_grid_da(is_unstructured, np.float32, 0.02),
            bulk_density=get_grid_da(is_unstructured, np.float32, 1300),
            distcoef=get_grid_da(is_unstructured, np.float32, 0.1),
        ),
        imod.mf6.NodePropertyFlow(
            get_grid_da(is_unstructured, np.int32), 3.0, True, 32.0, 34.0, 7
        ),
        imod.mf6.SpecificStorage(
            0.001, 0.1, True, get_grid_da(is_unstructured, np.int32)
        ),
        imod.mf6.StorageCoefficient(
            0.001, 0.1, True, get_grid_da(is_unstructured, np.int32)
        ),
    ]


def create_instance_boundary_condition_packages(is_unstructured):
    """
    creates instances of those modflow packages that are boundary conditions.
    """
    return [
        imod.mf6.ConstantConcentration(
            get_grid_da(is_unstructured, np.float32, 2),
            print_input=True,
            print_flows=True,
            save_flows=True,
        ),
        imod.mf6.ConstantHead(
            get_grid_da(is_unstructured, np.float32, 2),
            print_input=True,
            print_flows=True,
            save_flows=True,
        ),
        imod.mf6.Drainage(
            elevation=get_grid_da(is_unstructured, np.float64, 4),
            conductance=get_grid_da(is_unstructured, np.float64, 1e-3),
        ),
        imod.mf6.Evapotranspiration(
            surface=get_grid_da(is_unstructured, np.float64, 3),
            rate=get_grid_da(is_unstructured, np.float64, 2),
            depth=get_grid_da(is_unstructured, np.float64, 1),
            proportion_rate=get_grid_da(is_unstructured, np.float64, 0.2),
            proportion_depth=get_grid_da(is_unstructured, np.float64, 0.2),
            fixed_cell=True,
        ),
        imod.mf6.GeneralHeadBoundary(
            head=get_grid_da(is_unstructured, np.float64, 3),
            conductance=get_grid_da(is_unstructured, np.float64, 0.33),
        ),
        imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic(
            hydraulic_characteristic=get_grid_da(is_unstructured, np.float64, 0.33),
            idomain=get_grid_da(is_unstructured, int, 1),
            print_input=True,
        ),
        imod.mf6.HorizontalFlowBarrierMultiplier(
            multiplier=get_grid_da(is_unstructured, np.float64, 0.33),
            idomain=get_grid_da(is_unstructured, int, 1),
            print_input=True,
        ),
        imod.mf6.HorizontalFlowBarrierResistance(
            resistance=get_grid_da(is_unstructured, np.float64, 0.33),
            idomain=get_grid_da(is_unstructured, int, 1),
            print_input=True,
        ),
        imod.mf6.Recharge(
            rate=get_grid_da(is_unstructured, np.float64, 0.33),
        ),
        imod.mf6.River(
            stage=get_grid_da(is_unstructured, np.float64, 0.33),
            conductance=get_grid_da(is_unstructured, np.float64, 0.33),
            bottom_elevation=get_grid_da(is_unstructured, np.float64, 0.33),
        ),
        imod.mf6.SourceSinkMixing(
            package_names=["a", "b"],
            concentration_boundary_type=["a", "b"],
            auxiliary_variable_name=["a", "b"],
        ),
        create_lake_package(is_unstructured),
    ]


STRUCTURED_GRID_PACKAGES = [
    imod.mf6.StructuredDiscretization(
        2.0, get_grid_da(False, np.float32), get_grid_da(False, np.int32)
    ),
    imod.mf6.WellDisStructured(
        layer=[3, 2, 1],
        row=[1, 2, 3],
        column=[2, 2, 2],
        rate=[-5.0] * 3,
    ),
    imod.mf6.MassSourceLoading(
        rate=get_grid_da(False, np.float64, 0.33),
        print_input=True,
        print_flows=False,
        save_flows=False,
    ),
] + [
    *create_instance_packages(is_unstructured=False),
    *create_instance_boundary_condition_packages(False),
]


UNSTRUCTURED_GRID_PACKAGES = (
    [
        imod.mf6.WellDisVertices(
            layer=[1, 2, 1],
            cell2d=[3, 12, 23],
            rate=[-0.1, 0.2, 0.3],
            print_input=False,
            print_flows=False,
            save_flows=False,
        )
    ]
    + [create_vertices_discretization()]
    + [
        *create_instance_packages(is_unstructured=True),
        *create_instance_boundary_condition_packages(True),
    ]
)

GRIDLESS_PACKAGES = [
    imod.mf6.adv.Advection("upstream"),
    imod.mf6.Buoyancy(
        reference_density=1000.0,
        reference_concentration=[4.0, 25.0],
        density_concentration_slope=[0.7, -0.375],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
    ),
    imod.mf6.OutputControl(),
    imod.mf6.SolutionPresetSimple(modelnames=["gwf-1"]),
    imod.mf6.TimeDiscretization(
        xr.DataArray(
            data=[0.001, 7.0, 365.0],
            coords={"time": pd.date_range("2000-01-01", "2000-01-03")},
            dims=["time"],
        ),
        23,
        1.02,
    ),
    imod.mf6.Well(
        screen_top=[0.0, 0.0],
        screen_bottom=[-10.0, -10.0],
        x=[1.0, 6002.0],
        y=[3.0, 5004.0],
        rate=[1.0, 3.0],
        print_flows=True,
        validate=True,
    ),
]

ALL_PACKAGE_INSTANCES = (
    GRIDLESS_PACKAGES + STRUCTURED_GRID_PACKAGES + UNSTRUCTURED_GRID_PACKAGES
)
