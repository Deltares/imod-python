import sys
import textwrap
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
import xugrid as xu

import imod
from imod.logging import LoggerType, LogLevel
from imod.mf6.validation_context import ValidationContext
from imod.mf6.write_context import WriteContext


@pytest.mark.usefixtures("circle_model")
@pytest.mark.skipif(sys.version_info < (3, 7), reason="capture_output added in 3.7")
def test_simulation_write_and_run(circle_model, tmp_path):
    imod.logging.configure(
        LoggerType.PYTHON,
        log_level=LogLevel.DEBUG,
        add_default_file_handler=True,
        add_default_stream_handler=True,
    )
    simulation = circle_model

    with pytest.raises(
        RuntimeError, match="Simulation circle has not been written yet."
    ):
        circle_model.run()

    modeldir = tmp_path / "circle"

    mask = deepcopy(circle_model["GWF_1"].domain)
    mask.values[:, 133] = -1
    simulation.mask_all_models(mask)

    simulation.write(modeldir, binary=False, use_absolute_paths=True)
    simulation.run()

    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds",
        modeldir / "GWF_1/disv.disv.grb",
        simulation_start_time="01-01-1999",
        time_unit="s",
    )
    assert isinstance(head, xu.UgridDataArray)
    assert np.all(
        head["time"].values[0]
        == np.array(datetime(1999, 1, 1, 0, 0, 7), dtype="datetime64[ns]")
    )
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (52, 2, 216)


@pytest.mark.usefixtures("circle_model")
def test_gwfmodel_render(circle_model, tmp_path):
    simulation = circle_model
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    write_context1 = WriteContext()
    actual = gwfmodel.render("GWF_1", write_context1)
    path = "GWF_1"
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
    validation_context = ValidationContext(True)
    write_context2 = WriteContext(tmp_path)
    gwfmodel._write("GWF_1", globaltimes, write_context2, validation_context)
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
    assert head.shape == (52, 2, 216)


@pytest.mark.usefixtures("circle_model_evt")
def test_gwfmodel_render_evt(circle_model_evt, tmp_path):
    simulation = circle_model_evt
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    write_context1 = WriteContext()
    actual = gwfmodel.render("GWF_1", write_context1)
    path = "GWF_1"
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
    validation_context = ValidationContext(True)
    write_context2 = WriteContext(tmp_path)
    gwfmodel._write("GWF_1", globaltimes, write_context2, validation_context)
    assert (tmp_path / "GWF_1" / "GWF_1.nam").is_file()
    assert (tmp_path / "GWF_1").is_dir()
