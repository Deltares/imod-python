import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod


def make_circle_model():
    grid = imod.data.circle()

    nface = grid.n_face

    nlayer = 2

    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": [1, 2]},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )
    icelltype = xu.full_like(idomain, 0)
    k = xu.full_like(idomain, 1.0, dtype=np.float64)
    k33 = k.copy()
    rch_rate = xu.full_like(k.sel(layer=1), 0.001, dtype=float)
    bottom = k * xr.DataArray([5.0, 0.0], dims=["layer"])
    chd_location = xu.zeros_like(k.sel(layer=2), dtype=bool).ugrid.binary_dilation(
        border_value=True
    )
    constant_head = xu.full_like(k.sel(layer=2), 1.0).where(chd_location)

    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["disv"] = imod.mf6.VerticesDiscretization(
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
        save_flows=False,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)

    simulation = imod.mf6.Modflow6Simulation("circle")
    simulation["GWF_1"] = gwf_model
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
    simulation.create_time_discretization(additional_times=["2000-01-01", "2000-01-02"])
    return simulation


@pytest.fixture(scope="function")
def circle_model():
    return make_circle_model()


@pytest.mark.usefixtures("circle_model")
@pytest.fixture(scope="session")
def circle_result(tmpdir_factory):
    # Using a tmpdir_factory is the canonical way of sharing a tempory pytest
    # directory between different testing modules.
    modeldir = tmpdir_factory.mktemp("circle")
    simulation = make_circle_model()
    simulation.write(modeldir)
    simulation.run()
    return modeldir


def make_circle_model_evt():
    simulation = make_circle_model()
    gwf_model = simulation["GWF_1"]

    idomain = gwf_model["disv"].dataset["idomain"]
    like = idomain.sel(layer=1).astype(np.float64)
    face_dim = idomain.ugrid.grid.face_dimension

    rate = xu.full_like(like, 0.001)
    # Lay surface on chd level
    surface = xu.full_like(like, 1.0)
    depth = xu.full_like(like, 2.0)

    segments = xr.DataArray(
        data=[1, 2, 3], coords={"segment": [1, 2, 3]}, dims=("segment",)
    )
    segments_reversed = segments.copy()
    segments_reversed.values = [3, 2, 1]

    proportion_depth = xu.full_like(like, 0.3) * segments
    proportion_rate = xu.full_like(like, 0.3) * segments_reversed

    proportion_depth = proportion_depth.transpose("segment", face_dim)
    proportion_rate = proportion_rate.transpose("segment", face_dim)

    gwf_model["evt"] = imod.mf6.Evapotranspiration(
        surface, rate, depth, proportion_rate, proportion_depth
    )

    simulation["GWF_1"] = gwf_model

    return simulation


@pytest.fixture(scope="session")
def circle_model_evt():
    return make_circle_model_evt()


@pytest.mark.usefixtures("circle_model_evt")
@pytest.fixture(scope="session")
def circle_result_evt(tmpdir_factory):
    # Using a tmpdir_factory is the canonical way of sharing a tempory pytest
    # directory between different testing modules.
    modeldir = tmpdir_factory.mktemp("circle_evt")
    simulation = make_circle_model_evt()
    simulation.write(modeldir)
    simulation.run()
    return modeldir


def make_circle_model_save_sto():
    simulation = make_circle_model()
    gwf_model = simulation["GWF_1"]

    gwf_model["sto"].dataset["save_flows"] = True
    return simulation


@pytest.fixture(scope="session")
def circle_result_sto(tmpdir_factory):
    """
    Circle result with storage fluxes, which are saved as METH1 instead of METH6
    """
    # Using a tmpdir_factory is the canonical way of sharing a tempory pytest
    # directory between different testing modules.
    modeldir = tmpdir_factory.mktemp("circle_sto")
    simulation = make_circle_model_save_sto()
    simulation.write(modeldir)
    simulation.run()
    return modeldir
