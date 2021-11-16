import subprocess
import sys
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.mark.usefixtures("twri_model")
def test_dis_render(twri_model, tmp_path):
    simulation = twri_model
    dis = simulation["GWF_1"]["dis"]
    actual = dis.render(tmp_path, "dis")
    path = tmp_path.as_posix()
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
    dis.write(tmp_path, "dis")
    assert (tmp_path / "dis.dis").is_file()
    assert (tmp_path / "dis").is_dir()
    assert (tmp_path / "dis" / "idomain.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_chd_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    chd = simulation["GWF_1"]["chd"]
    actual = chd.render(tmp_path, "chd", globaltimes)
    path = tmp_path.as_posix()
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
    chd.write(tmp_path, "chd", globaltimes)
    assert (tmp_path / "chd.chd").is_file()
    assert (tmp_path / "chd").is_dir()
    assert (tmp_path / "chd" / "chd.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_drn_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    drn = simulation["GWF_1"]["drn"]
    actual = drn.render(tmp_path, "drn", globaltimes)
    path = tmp_path.as_posix()
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
    drn.write(tmp_path, "drn", globaltimes)
    assert (tmp_path / "drn.drn").is_file()
    assert (tmp_path / "drn").is_dir()
    assert (tmp_path / "drn" / "drn.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_ic_render(twri_model, tmp_path):
    simulation = twri_model
    ic = simulation["GWF_1"]["ic"]
    actual = ic.render(tmp_path, "ic")
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
    ic.write(tmp_path, "ic")
    assert (tmp_path / "ic.ic").is_file()


@pytest.mark.usefixtures("twri_model")
def test_npf_render(twri_model, tmp_path):
    simulation = twri_model
    npf = simulation["GWF_1"]["npf"]
    actual = npf.render(tmp_path, "npf")
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
    npf.write(tmp_path, "npf")
    assert (tmp_path / "npf.npf").is_file()


@pytest.mark.usefixtures("twri_model")
def test_oc_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    oc = simulation["GWF_1"]["oc"]
    path = tmp_path.as_posix()
    actual = oc.render(tmp_path, "oc", globaltimes)
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
    oc.write(tmp_path, "oc", globaltimes)
    assert (tmp_path / "oc.oc").is_file()


@pytest.mark.usefixtures("twri_model")
def test_rch_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    rch = simulation["GWF_1"]["rch"]
    actual = rch.render(tmp_path, "rch", globaltimes)
    path = tmp_path.as_posix()
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
    rch.write(tmp_path, "rch", globaltimes)
    assert (tmp_path / "rch.rch").is_file()
    assert (tmp_path / "rch").is_dir()
    assert (tmp_path / "rch" / "rch.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_wel_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    wel = simulation["GWF_1"]["wel"]
    actual = wel.render(tmp_path, "wel", globaltimes)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
              print_input
              print_flows
              save_flows
            end options

            begin dimensions
              maxbound 15
            end dimensions

            begin period 1
              open/close {path}/wel/wel.bin (binary)
            end period
            """
    )
    assert actual == expected
    wel.write(tmp_path, "wel", globaltimes)
    assert (tmp_path / "wel.wel").is_file()
    assert (tmp_path / "wel").is_dir()
    assert (tmp_path / "wel" / "wel.bin").is_file()


@pytest.mark.usefixtures("twri_model")
def test_solver_render(twri_model, tmp_path):
    simulation = twri_model
    solver = simulation["solver"]
    actual = solver.render()
    expected = textwrap.dedent(
        """\
            begin options
              print_option summary
            end options

            begin nonlinear
              outer_hclose 0.0001
              outer_maximum 500
            end nonlinear

            begin linear
              inner_maximum 100
              inner_hclose 0.0001
              inner_rclose 0.001
              linear_acceleration cg
              relaxation_factor 0.97
            end linear
            """
    )
    assert actual == expected
    solver.write(tmp_path, "solver")
    assert (tmp_path / "solver.ims").is_file()


@pytest.mark.usefixtures("twri_model")
def test_gwfmodel_render(twri_model, tmp_path):
    simulation = twri_model
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    actual = gwfmodel.render(tmp_path)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
            end options

            begin packages
              dis6 {path}/dis.dis
              chd6 {path}/chd.chd
              drn6 {path}/drn.drn
              ic6 {path}/ic.ic
              npf6 {path}/npf.npf
              oc6 {path}/oc.oc
              rch6 {path}/rch.rch
              wel6 {path}/wel.wel
            end packages
            """
    )
    assert actual == expected
    gwfmodel.write(tmp_path, "GWF_1", globaltimes)
    assert (tmp_path / "GWF_1" / "GWF_1.nam").is_file()
    assert (tmp_path / "GWF_1").is_dir()


@pytest.mark.usefixtures("twri_model")
def test_simulation_render(twri_model):
    simulation = twri_model
    actual = simulation.render()
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
    simulation.write(modeldir)
    simulation.run()

    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds", modeldir / "GWF_1/dis.dis.grb"
    )
    assert head.dims == ("time", "layer", "y", "x")
    assert head.shape == (1, 3, 15, 15)
    meanhead_layer = head.groupby("layer").mean(dim=xr.ALL_DIMS)
    mean_answer = np.array([59.79181509, 30.44132373, 24.88576811])
    assert np.allclose(meanhead_layer, mean_answer)
