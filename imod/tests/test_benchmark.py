# keep track of performance using pytest-benchmark
# https://pytest-benchmark.readthedocs.io/en/stable/

import flopy
import numpy as np
import pytest
import xarray as xr

import imod


def setup_mf6_basic_simulation_flopy():
    """
    Defines a basic FloPy Modflow 6 model.

    The code is based on this example notebook from the FloPy documentation:
    https://github.com/modflowpy/flopy/blob/develop/examples/Notebooks/flopy3_mf6_A_simple-model.ipynb

    The only real change is to increase the number of layers from 10 to 100 to get 1 million cells.
    """
    # For this example, we will set up a model workspace.
    # Model input files and output files will reside here.

    name = "mf6lake"
    h1 = 100.0
    h2 = 90.0
    Nlay = 10
    N = 101
    L = 400.0
    H = 50.0
    k = 1.0

    # Create the Flopy simulation object
    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name="mf6", version="mf6")

    # Create the Flopy temporal discretization object
    flopy.mf6.modflow.mftdis.ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{name}.nam"
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)

    # Create the Flopy iterative model solver (ims) Package object
    flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")

    # Create the discretization package
    bot = np.linspace(-H / Nlay, -H, Nlay)
    delrow = delcol = L / (N - 1)
    flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=Nlay,
        nrow=N,
        ncol=N,
        delr=delrow,
        delc=delcol,
        top=0.0,
        botm=bot,
    )

    # Create the initial conditions package
    start = h1 * np.ones((Nlay, N, N))
    flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=start)

    # Create the node property flow package
    flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf, pname="npf", icelltype=1, k=k, save_flows=True
    )

    # Create the constant head package.
    chd_rec = []
    chd_rec.append(((0, int(N / 4), int(N / 4)), h2))
    for layer in range(0, Nlay):
        for row_col in range(0, N):
            chd_rec.append(((layer, row_col, 0), h1))
            chd_rec.append(((layer, row_col, N - 1), h1))
            if row_col != 0 and row_col != N - 1:
                chd_rec.append(((layer, 0, row_col), h1))
                chd_rec.append(((layer, N - 1, row_col), h1))
    flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
        gwf,
        pname="chd",
        maxbound=len(chd_rec),
        stress_period_data=chd_rec,
        save_flows=True,
    )

    # Create the output control package
    headfile = f"{name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{name}.cbc"
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord = [("HEAD", "LAST")]
    flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
    )
    return sim


def setup_mf6_basic_simulation_imod():
    """
    Defines a basic Modflow 6 model, to benchmark against the FloPy version above.
    """

    h1 = 100.0
    h2 = 90.0
    nlay = 10
    nrow = 101
    ncol = 101
    shape = (nlay, nrow, ncol)
    dy = -4.0
    dx = 4.0
    height = 50.0
    k = 1.0

    xmax = dx * ncol
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, 0.0, dy) + 0.5 * dy
    x = np.arange(0.0, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    # Discretization data
    like = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    idomain = like.astype(np.int32)
    bottom = xr.DataArray(
        np.linspace(-height / nlay, -height, nlay), {"layer": layer}, ("layer",)
    )

    # Constant head
    head = xr.full_like(like, np.nan)
    # set all side edges to h1
    head[:, [0, -1], :] = h1
    head[:, :, [0, -1]] = h1
    # set a single cell to h2
    head[0, nrow // 4, ncol // 4] = h2

    # Create and fill the groundwater model.
    gwf_model = imod.mf6.GroundwaterFlowModel()
    gwf_model["dis"] = imod.mf6.StructuredDiscretization(
        top=0.0, bottom=bottom, idomain=idomain
    )
    gwf_model["chd"] = imod.mf6.ConstantHead(head, save_flows=True)
    gwf_model["ic"] = imod.mf6.InitialConditions(head=100.0)
    gwf_model["npf"] = imod.mf6.NodePropertyFlow(icelltype=1, k=k, save_flows=True)
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="last", save_budget="last")
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1.0e-15,
        specific_yield=0.15,
        convertible=0,
        transient=False,
    )

    # Attach it to a simulation
    simulation = imod.mf6.Modflow6Simulation("mf6basic")
    simulation["gwf"] = gwf_model
    # Define solver settings
    simulation["solver"] = imod.mf6.SolutionPresetSimple(
        print_option="summary", csv_output=False, no_ptc=True
    )

    # Collect time discretization
    simulation.create_time_discretization(additional_times=["2000-01-01", "2000-01-02"])
    return simulation


@pytest.fixture(scope="module")
def mf6_basic_simulation_flopy():
    return setup_mf6_basic_simulation_flopy()


@pytest.fixture(scope="module")
def mf6_basic_simulation_imod():
    return setup_mf6_basic_simulation_imod()


def write_basic_flopy(mf6_basic_simulation_flopy, tmp_path):
    simulation = mf6_basic_simulation_flopy
    simulation.set_sim_path(str(tmp_path))
    simulation.write_simulation()


def write_basic_imod_binary(mf6_basic_simulation_imod, tmp_path):
    simulation = mf6_basic_simulation_imod
    modeldir = tmp_path / "mf6-basic"
    simulation.write(modeldir)


def write_basic_imod_text(mf6_basic_simulation_imod, tmp_path):
    simulation = mf6_basic_simulation_imod
    modeldir = tmp_path / "mf6-basic"
    simulation.write(modeldir, binary=False)


def test_setup_basic_flopy(benchmark, mf6_basic_simulation_flopy, tmp_path):
    benchmark(setup_mf6_basic_simulation_flopy)


def test_setup_basic_imod(benchmark, mf6_basic_simulation_imod, tmp_path):
    benchmark(setup_mf6_basic_simulation_imod)


def test_write_basic_flopy(benchmark, mf6_basic_simulation_flopy, tmp_path):
    benchmark(write_basic_flopy, mf6_basic_simulation_flopy, tmp_path)


def test_write_basic_imod_binary(benchmark, mf6_basic_simulation_imod, tmp_path):
    benchmark(write_basic_imod_binary, mf6_basic_simulation_imod, tmp_path)


def test_write_basic_imod_text(benchmark, mf6_basic_simulation_imod, tmp_path):
    benchmark(write_basic_imod_text, mf6_basic_simulation_imod, tmp_path)
