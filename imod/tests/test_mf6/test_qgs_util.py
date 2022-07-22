import numpy as np
import pandas as pd
import pytest
import xarray as xr

import imod
import imod.mf6.qgs_util as qgs_util


@pytest.fixture(scope="module")
def simple_model():
    # Initiate simple_model
    gwf_model = imod.mf6.GroundwaterFlowModel()

    # Create discretication
    shape = nlay, nrow, ncol = 1, 9, 9
    nper = 12
    time = pd.date_range("2018-01-01", periods=nper, freq="H")

    dx = 1000.0
    dy = -1000.0
    dz = np.array([2])
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    like = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    idomain = like.astype(np.int32)
    idomain[:, slice(0, 3), slice(0, 3)] = 0

    top = 0.0
    bottom = xr.DataArray(
        np.cumsum(layer.astype(np.float64) * -1 * dz),
        coords={"layer": layer},
        dims="layer",
    )

    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        idomain=idomain, top=top, bottom=bottom
    )

    # Create constant head
    head = xr.full_like(like, 0.0)
    head[..., 0] = -2.0
    head[..., -1] = -2.0
    head = head.where(idomain == 1)
    head = head.expand_dims(time=time)

    gwf_model["chd"] = imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )

    # Create nodeproperty flow
    icelltype = xr.full_like(idomain, 0)
    k = 10.0
    k33 = 1.0
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=False,
        perched=False,
        save_flows=True,
    )

    # Create initial conditions
    shd = -2.0

    gwf_model["ic"] = imod.mf6.InitialConditions(head=shd)

    # Storage
    Ss = xr.full_like(like, 1e-5)
    Sy = xr.full_like(like, 0.1)
    iconvert = xr.full_like(idomain, 0)

    gwf_model["sto"] = imod.mf6.SpecificStorage(Ss, Sy, True, iconvert)

    # Set output control
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")

    # Attach it to a simulation
    simulation = imod.mf6.Modflow6Simulation("test")
    simulation["GWF_1"] = gwf_model
    # Define solver settings
    simulation["solver"] = imod.mf6.Solution(
        modelnames=["GWF_1"],
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
    # Collect time discretization
    simulation.create_time_discretization(additional_times=time)

    return simulation


@pytest.fixture(scope="module")
def qgs_tree(simple_model):
    gwf = simple_model["GWF_1"]
    pkgnames = ["chd", "sto"]
    data_paths = [".", "."]
    data_vars_ls = [["head"], ["convertible"]]
    epsg = "epsg:28992"

    return qgs_util._create_qgis_tree(gwf, pkgnames, data_paths, data_vars_ls, crs=epsg)


def test_get_color_hexes_cmap():
    n = 5
    actual = qgs_util._get_color_hexes_cmap(n, cmap_name="magma")
    expected = ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"]

    assert len(actual) == n
    assert actual == expected


def test_create_colorramp():
    n = 5
    actual = qgs_util._create_colorramp(0, 1, n, cmap_name="magma")

    expected_attrs = [
        "colorramp",
        "item",
        "classificationMode",
        "colorRampType",
        "clip",
    ]
    actual_attrs = list(actual.__dict__.keys())

    expected_item = {
        "label": "0.00",
        "value": "0.00",
        "color": "#000004",
        "alpha": "255",
    }

    assert actual_attrs == expected_attrs
    assert len(actual.item) == n
    assert actual.item[0].__dict__ == expected_item


def test_generate_layer_ids():
    pkgnames = ["RCH", "STO"]
    actual = qgs_util._generate_layer_ids(pkgnames)

    assert len(actual) == len(pkgnames)
    assert len(actual[0]) == 40


def test_create_groups(simple_model):
    gwf = simple_model["GWF_1"]
    data_vars = ["specific_storage", "specific_yield", "convertible"]
    actual = qgs_util._create_groups(gwf["sto"], data_vars, aggregate_layers=False)
    expected = [("specific_storage", 1), ("specific_yield", 1), ("convertible", 1)]

    assert actual == expected


def test_data_range_per_data_var(simple_model):
    gwf = simple_model["GWF_1"]
    actual = qgs_util._data_range_per_data_var(gwf["chd"], ["head"])
    expected = ({"head": -2.0}, {"head": 0.0})

    assert actual == expected


def test_get_time_range(simple_model):
    gwf = simple_model["GWF_1"]
    actual = qgs_util._get_time_range(gwf)
    actual_str = [str(i) for i in actual]
    expected = ["2018-01-01T00:00:00.000000000", "2018-01-01T11:00:00.000000000"]
    assert actual_str == expected


def test_create_mesh_dataset_group(simple_model):
    gwf = simple_model["GWF_1"]
    data_vars = ["specific_storage", "specific_yield", "convertible"]
    groups = qgs_util._create_groups(gwf["sto"], data_vars, aggregate_layers=False)
    actual = qgs_util._create_mesh_dataset_group(gwf["sto"], groups)
    actual = list(actual.__dict__.items())[0]

    expected_d_item = {
        "provider_name": "specific_storage_layer:1",
        "display_name": "",
        "dataset_index": "0",
        "is_vector": "0",
        "is_enabled": "1",
    }

    assert actual[0] == "mesh_dataset_group_tree_item"
    assert len(actual[1]) == 4
    assert actual[1][1].__dict__ == expected_d_item


def test_create_qgis_tree(qgs_tree):
    actual = qgs_tree

    xmax_first_layer = actual.projectlayers.maplayer[0].extent.xmax

    assert len(actual.__dict__) == 27
    assert len(actual.projectlayers.maplayer) == 2
    assert np.isclose(xmax_first_layer, 9000.0)


def test_make_processor(qgs_tree):
    # Due to this issue with optional aggregates:
    # https://github.com/gatkin/declxml/issues/27
    # we cannot read the object from string itsself,
    # which would be preferred.
    import imod.qgs as qgs
    from imod import declxml as xml

    processor = qgs.make_processor(qgs.Qgis)
    to_string = xml.serialize_to_string(processor, qgs_tree, indent="  ")

    # TODO: create more robust test than this
    assert len(to_string.splitlines()) == 225
