import sys
import textwrap

import pytest
import xugrid as xu

import imod
from imod.mf6.write_context import WriteContext


@pytest.mark.usefixtures("circle_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write_and_run(circle_model, tmp_path):
    simulation = circle_model

    with pytest.raises(
        RuntimeError, match="Simulation circle has not been written yet."
    ):
        circle_model.run()

    modeldir = tmp_path / "circle"
    simulation.write(modeldir, binary=False)
    simulation.run()

    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds", modeldir / "GWF_1/disv.disv.grb"
    )
    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (1, 2, 216)


@pytest.mark.usefixtures("circle_model")
def test_gwfmodel_render(circle_model, tmp_path):
    simulation = circle_model
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    actual = gwfmodel.render(tmp_path)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
            end options

            begin packages
              disv6 {path}/disv.disv disv
              chd6 {path}/chd.chd chd
              ic6 {path}/ic.ic ic
              npf6 {path}/npf.npf npf
              sto6 {path}/sto.sto sto
              oc6 {path}/oc.oc oc
              rch6 {path}/rch.rch rch
            end packages
            """
    )
    assert actual == expected
    context = WriteContext(tmp_path)
    gwfmodel.write("GWF_1", globaltimes, context)
    assert (tmp_path / "GWF_1" / "GWF_1.nam").is_file()
    assert (tmp_path / "GWF_1").is_dir()


@pytest.mark.usefixtures("circle_model_evt")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write_and_run_evt(circle_model_evt, tmp_path):
    simulation = circle_model_evt

    with pytest.raises(
        RuntimeError, match="Simulation circle has not been written yet."
    ):
        circle_model_evt.run()

    modeldir = tmp_path / "circle_evt"
    simulation.write(modeldir, binary=False)
    simulation.run()

    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds", modeldir / "GWF_1/disv.disv.grb"
    )
    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (1, 2, 216)


@pytest.mark.usefixtures("circle_model_evt")
def test_gwfmodel_render_evt(circle_model_evt, tmp_path):
    simulation = circle_model_evt
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    actual = gwfmodel.render(tmp_path)
    path = tmp_path.as_posix()
    expected = textwrap.dedent(
        f"""\
            begin options
            end options

            begin packages
              disv6 {path}/disv.disv disv
              chd6 {path}/chd.chd chd
              ic6 {path}/ic.ic ic
              npf6 {path}/npf.npf npf
              sto6 {path}/sto.sto sto
              oc6 {path}/oc.oc oc
              rch6 {path}/rch.rch rch
              evt6 {path}/evt.evt evt
            end packages
            """
    )
    assert actual == expected
    context = WriteContext(tmp_path)
    gwfmodel.write("GWF_1", globaltimes, context)
    assert (tmp_path / "GWF_1" / "GWF_1.nam").is_file()
    assert (tmp_path / "GWF_1").is_dir()
