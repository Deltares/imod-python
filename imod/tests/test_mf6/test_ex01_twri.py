import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.validation_context import ValidationContext
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.typing.grid import ones_like


@pytest.mark.usefixtures("twri_model")
def test_dis_render(twri_model, tmp_path):
    simulation = twri_model
    dis = simulation["GWF_1"]["dis"]
    actual = dis.render(
        directory=tmp_path,
        pkgname="dis",
        globaltimes=None,
        binary=True,
    )
    path = Path(tmp_path).as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
              xorigin 0.0
              yorigin 0.0
            end options

            begin dimensions
              nlay 3
              nrow 15
              ncol 15
            end dimensions

            begin griddata
              delr
                constant 5000.0
              delc
                constant 5000.0
              top
                constant 200.0
              botm layered
                constant -200.0
                constant -300.0
                constant -450.0
              idomain
                open/close {path}/dis/idomain.bin (binary)
            end griddata
            """
    )
    assert actual == expected
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)

    dis.write(pkgname="dis", globaltimes=None, write_context=write_context)
    assert (tmp_path / "dis.dis").is_file()
    assert (tmp_path / "dis").is_dir()
    assert (tmp_path / "dis" / "idomain.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_chd_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    chd = simulation["GWF_1"]["chd"]
    actual = chd.render(
        directory=tmp_path,
        pkgname="chd",
        globaltimes=globaltimes,
        binary=True,
    )
    path = Path(tmp_path).as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
              print_input
              print_flows
              save_flows
            end options

            begin dimensions
              maxbound 30
            end dimensions

            begin period 1
              open/close {path}/chd/chd.bin (binary)
            end period
            """
    )
    assert actual == expected
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    chd.write(pkgname="chd", globaltimes=None, write_context=write_context)
    assert (tmp_path / "chd.chd").is_file()
    assert (tmp_path / "chd").is_dir()
    assert (tmp_path / "chd" / "chd.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_drn_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    drn = simulation["GWF_1"]["drn"]
    actual = drn.render(
        directory=tmp_path,
        pkgname="drn",
        globaltimes=globaltimes,
        binary=True,
    )
    path = Path(tmp_path).as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
              print_input
              print_flows
              save_flows
            end options

            begin dimensions
              maxbound 9
            end dimensions

            begin period 1
              open/close {path}/drn/drn.bin (binary)
            end period
            """
    )
    assert actual == expected
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    drn.write(pkgname="drn", globaltimes=None, write_context=write_context)
    assert (tmp_path / "drn.drn").is_file()
    assert (tmp_path / "drn").is_dir()
    assert (tmp_path / "drn" / "drn.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_ic_render(twri_model, tmp_path):
    simulation = twri_model
    ic = simulation["GWF_1"]["ic"]
    actual = ic.render(
        directory=tmp_path,
        pkgname="ic",
        globaltimes=None,
        binary=True,
    )
    expected = textwrap.dedent(
        """\
            begin options
            end options

            begin griddata
              strt
                constant 0.0
            end griddata
            """
    )
    assert actual == expected
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    ic.write(pkgname="ic", globaltimes=None, write_context=write_context)
    assert (tmp_path / "ic.ic").is_file()


@pytest.mark.usefixtures("twri_model")
def test_npf_render(twri_model, tmp_path):
    simulation = twri_model
    npf = simulation["GWF_1"]["npf"]
    actual = npf.render(
        directory=tmp_path, pkgname="npf", globaltimes=None, binary=True
    )
    expected = textwrap.dedent(
        """\
            begin options
              save_flows
              variablecv dewatered
              perched
            end options

            begin griddata
              icelltype layered
                constant 1
                constant 0
                constant 0
              k layered
                constant 0.001
                constant 0.0001
                constant 0.0002
              k33 layered
                constant 2e-08
                constant 2e-08
                constant 2e-08
            end griddata
            """
    )
    assert actual == expected
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    npf.write(pkgname="npf", globaltimes=None, write_context=write_context)
    assert (tmp_path / "npf.npf").is_file()


@pytest.mark.usefixtures("twri_model")
def test_oc_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    oc = simulation["GWF_1"]["oc"]

    # An absolute path should be maintained in the fileout entries!
    path = tmp_path.as_posix()
    actual = oc.render(
        directory=tmp_path, pkgname="oc", globaltimes=globaltimes, binary=True
    )
    expected = textwrap.dedent(
        f"""\
          begin options
            budget fileout {path}/{tmp_path.stem}.cbc
            head fileout {path}/{tmp_path.stem}.hds
          end options

          begin period 1
            save head all
            save budget all
          end period
          """
    )
    assert actual == expected

    path = Path(tmp_path.stem)
    actual = oc.render(
        directory=path, pkgname="oc", globaltimes=globaltimes, binary=True
    )
    expected = textwrap.dedent(
        f"""\
          begin options
            budget fileout {path}/{path}.cbc
            head fileout {path}/{path}.hds
          end options

          begin period 1
            save head all
            save budget all
          end period
          """
    )
    assert actual == expected


@pytest.mark.usefixtures("twri_model")
def test_rch_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    rch = simulation["GWF_1"]["rch"]
    actual = rch.render(
        directory=tmp_path,
        pkgname="rch",
        globaltimes=globaltimes,
        binary=True,
    )
    path = Path(tmp_path).as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
            end options

            begin dimensions
              maxbound 225
            end dimensions

            begin period 1
              open/close {path}/rch/rch.bin (binary)
            end period
            """
    )
    assert actual == expected
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    rch.write(pkgname="rch", globaltimes=None, write_context=write_context)
    assert (tmp_path / "rch.rch").is_file()
    assert (tmp_path / "rch").is_dir()
    assert (tmp_path / "rch" / "rch.bin").is_file()


def test_wel_render(twri_model, tmp_path):
    simulation = twri_model
    simulation.write(
        tmp_path,
    )
    wel_path = tmp_path / "GWF_1/wel.wel"
    with open(wel_path, "r") as wel_file:
        actual = wel_file.read()

    expected = textwrap.dedent(
        """\
            begin options
              save_flows
            end options

            begin dimensions
              maxbound 45
            end dimensions

            begin period 1
              open/close GWF_1/wel/wel.bin (binary)
            end period
            """
    )
    assert actual == expected
    assert (tmp_path / "GWF_1/wel").is_dir()
    assert (tmp_path / "GWF_1/wel" / "wel.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_solver_render(twri_model, tmp_path):
    simulation = twri_model
    solver = simulation["solver"]
    actual = solver.render(
        directory=tmp_path,
        pkgname="solver",
        globaltimes=None,
        binary=True,
    )
    expected = textwrap.dedent(
        """\
            begin options
              print_option summary
            end options

            begin nonlinear
              outer_dvclose 0.0001
              outer_maximum 500
            end nonlinear

            begin linear
              inner_maximum 100
              inner_dvclose 0.0001
              inner_rclose 0.001
              linear_acceleration cg
              relaxation_factor 0.97
            end linear
            """
    )
    assert actual == expected
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=True)
    solver.write(pkgname="solver", globaltimes=None, write_context=write_context)
    assert (tmp_path / "solver.ims").is_file()


@pytest.mark.usefixtures("twri_model")
def test_gwfmodel_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    path = Path(tmp_path.stem).as_posix()
    validation_context = ValidationContext(tmp_path)
    write_context = WriteContext(tmp_path)
    actual = gwfmodel.render(path, write_context)
    expected = textwrap.dedent(
        f"""\
            begin options
            end options

            begin packages
              wel6 {path}/wel.wel wel
              dis6 {path}/dis.dis dis
              chd6 {path}/chd.chd chd
              drn6 {path}/drn.drn drn
              ic6 {path}/ic.ic ic
              npf6 {path}/npf.npf npf
              oc6 {path}/oc.oc oc
              rch6 {path}/rch.rch rch
              sto6 {path}/sto.sto sto
              api6 {path}/api.api api
            end packages
            """
    )
    assert actual == expected
    gwfmodel.write("GWF_1", globaltimes, write_context, validation_context)
    assert (tmp_path / "GWF_1" / "GWF_1.nam").is_file()
    assert (tmp_path / "GWF_1").is_dir()


@pytest.mark.usefixtures("twri_model")
def test_simulation_render(twri_model):
    simulation = twri_model
    write_context = WriteContext(".")
    actual = simulation.render(write_context)

    expected = textwrap.dedent(
        """\
            begin options
            end options

            begin timing
              tdis6 time_discretization.tdis
            end timing

            begin models
              gwf6 GWF_1/GWF_1.nam GWF_1
            end models

            begin exchanges

            end exchanges

            begin solutiongroup 1
              ims6 solver.ims GWF_1
            end solutiongroup
            """
    )
    assert actual == expected


@pytest.mark.usefixtures("twri_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write_and_run(twri_model, tmp_path):
    simulation = twri_model

    with pytest.raises(
        RuntimeError, match="Simulation ex01-twri has not been written yet."
    ):
        twri_model.run()

    modeldir = tmp_path / "ex01-twri"
    simulation.write(
        modeldir,
        binary=False,
    )
    simulation.run()

    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds",
        modeldir / "GWF_1/dis.dis.grb",
        simulation_start_time="01-01-1999",
        time_unit="d",
    )
    assert isinstance(head, xr.DataArray)
    assert head.dims == ("time", "layer", "y", "x")
    assert head.shape == (1, 3, 15, 15)
    assert np.all(
        head["time"].values
        == np.array("1999-01-02T00:00:00.000000000", dtype="datetime64[ns]")
    )
    meanhead_layer = head.groupby("layer").mean(dim=xr.ALL_DIMS)
    mean_answer = np.array([59.79181509, 30.44132373, 24.88576811])
    assert np.allclose(meanhead_layer, mean_answer)


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write_storage(transient_twri_model, tmp_path):
    simulation = transient_twri_model
    modeldir = tmp_path / "ex01-twri-transient"
    simulation.write(modeldir, binary=True)
    simulation.run()

    cbc = imod.mf6.open_cbc(
        modeldir / "GWF_1/GWF_1.cbc", modeldir / "GWF_1/dis.dis.grb"
    )

    # Test if storage fluxes properly read
    assert "sto-ss" in cbc.keys()
    sto_ss = cbc["sto-ss"]

    assert isinstance(sto_ss, xr.DataArray)
    assert sto_ss.dims == ("time", "layer", "y", "x")
    assert sto_ss.shape == (30, 3, 15, 15)

    meanflux_layer = sto_ss.groupby("layer").mean(dim=xr.ALL_DIMS).compute()
    mean_answer = np.array([-1.993060e-05, -2.536776e-06, -3.110719e-06])
    np.testing.assert_allclose(meanflux_layer.values, mean_answer, rtol=2e-7)


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_wrong_dtype(transient_twri_model, tmp_path):
    simulation = transient_twri_model

    # Make input in wrong dtype
    simulation["GWF_1"]["rch"].dataset["rate"] = (
        simulation["GWF_1"]["rch"].dataset["rate"].astype(np.int16)
    )

    modeldir = tmp_path / "ex01-twri-transient"

    with pytest.raises(ValidationError):
        simulation.write(modeldir, binary=True)


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_validate_false(transient_twri_model, tmp_path):
    simulation = transient_twri_model

    # Make input in wrong dtype
    simulation["GWF_1"]["rch"].dataset["rate"] = (
        simulation["GWF_1"]["rch"].dataset["rate"].astype(np.int16)
    )

    modeldir = tmp_path / "ex01-twri-transient"
    simulation.write(modeldir, binary=True, validate=False)


@pytest.mark.usefixtures("twri_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write_errors(twri_model, tmp_path):
    simulation = twri_model
    model = simulation["GWF_1"]
    model.pop("sto")
    modeldir = tmp_path / "ex01-twri"

    expected_message = "No sto package found in model GWF_1"
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        simulation.write(modeldir, binary=True)


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_slice_and_run(transient_twri_model, tmp_path):
    # TODO: bring back well once slicing is implemented...
    transient_twri_model["GWF_1"].pop("wel")
    simulation = transient_twri_model.clip_box(
        time_min="2000-01-10",
        time_max="2000-01-20",
        layer_min=1,
        layer_max=2,
        x_min=None,
        x_max=65_000.0,
        y_min=10_000.0,
        y_max=65_000.0,
    )
    modeldir = tmp_path / "ex01-twri-transient-slice"
    simulation.write(modeldir, binary=True)
    simulation.run()


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_slice_and_run_with_state(transient_twri_model, tmp_path):
    # TODO: bring back well once slicing is implemented...
    transient_twri_model["GWF_1"].pop("wel")
    transient_twri_model.write(tmp_path, False, True, False)
    transient_twri_model.run()
    head = transient_twri_model.open_head()

    # set the boundary to  recognizable value, in this case 1.23
    state_for_boundary = {"GWF_1": 1.23 * ones_like(head)}
    clipped_simulation = transient_twri_model.clip_box(
        time_min="2000-01-10",
        time_max="2000-01-20",
        layer_min=1,
        layer_max=2,
        x_min=None,
        x_max=65_000.0,
        y_min=10_000.0,
        y_max=65_000.0,
        states_for_boundary=state_for_boundary,
    )

    # check that the value 1.23 occurrs in the chd_clipped package of the clipped simulation
    clipped_boundary = clipped_simulation["GWF_1"]["chd_clipped"].dataset.sel(
        time=2, layer=1
    )
    np_array = clipped_boundary["head"].values
    assert (np_array == 1.23).sum() == 33


@pytest.mark.usefixtures("transient_twri_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_slice_and_run_purge_empty_package(transient_twri_model, tmp_path):
    # TODO: bring back well once slicing is implemented...
    transient_twri_model["GWF_1"].pop("wel")

    # add a package out of the clipping area
    faraway_constant_head = xr.full_like(
        transient_twri_model["GWF_1"].domain, np.nan, dtype=float
    )
    faraway_constant_head.isel(x=14, y=0, layer=0).values[()] = 1
    faraway_constant_head_package = imod.mf6.ConstantHead(
        faraway_constant_head, print_input=True, print_flows=True, save_flows=True
    )
    assert not faraway_constant_head_package.is_empty()
    transient_twri_model["GWF_1"]["chd_far"] = faraway_constant_head_package

    simulation = transient_twri_model.clip_box(
        time_min="2000-01-10",
        time_max="2000-01-20",
        layer_min=1,
        layer_max=2,
        x_min=None,
        x_max=65_000.0,
        y_min=None,
        y_max=12500.0,
    )
    assert "chd_far" not in list(simulation["GWF_1"].keys())
    modeldir = tmp_path / "ex01-twri-transient-slice"
    simulation.write(modeldir, binary=True)
    simulation.run()
