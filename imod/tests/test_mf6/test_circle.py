import textwrap
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
import xugrid as xu

import imod
from imod.logging import LoggerType, LogLevel
from imod.mf6.validation_settings import ValidationSettings
from imod.mf6.write_context import WriteContext


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


def test_gwfmodel_render(circle_model, tmp_path):
    simulation = circle_model
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    write_context1 = WriteContext()
    actual = gwfmodel._render("GWF_1", write_context1)
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
    validation_context = ValidationSettings(True)
    write_context2 = WriteContext(tmp_path)
    gwfmodel._write("GWF_1", globaltimes, write_context2, validation_context)
    assert (tmp_path / "GWF_1" / "GWF_1.nam").is_file()
    assert (tmp_path / "GWF_1").is_dir()

def test_simulation_write_and_run_evt(circle_model_evt, tmp_path):
    simulation = circle_model_evt

    with pytest.raises(
        RuntimeError, match="Simulation circle has not been written yet."
    ):
        circle_model_evt.run()

    modeldir = tmp_path / "circle_evt"
    simulation.write(modeldir, binary=True)
    simulation.run()

    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds", modeldir / "GWF_1/disv.disv.grb"
    )
    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (52, 2, 216)

def test_simulation_write_and_run_evt__no_segments(circle_model_evt, tmp_path):
    simulation = circle_model_evt

    evt = circle_model_evt["GWF_1"].pop("evt")
    evt_ds = evt.dataset
    evt_ds = evt_ds.drop_vars(["proportion_rate", "proportion_depth"])
    evt_dict = {key: evt_ds[key] for key in evt_ds.keys()}
    circle_model_evt["GWF_1"]["evt"] = imod.mf6.Evapotranspiration(**evt_dict)

    with pytest.raises(
        RuntimeError, match="Simulation circle has not been written yet."
    ):
        circle_model_evt.run()

    modeldir = tmp_path / "circle_evt"
    simulation.write(modeldir, binary=True)
    simulation.run()

    head = imod.mf6.open_hds(
        modeldir / "GWF_1/GWF_1.hds", modeldir / "GWF_1/disv.disv.grb"
    )
    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (52, 2, 216)



def test_gwfmodel_render_evt(circle_model_evt, tmp_path):
    simulation = circle_model_evt
    globaltimes = simulation["time_discretization"]["time"].values
    gwfmodel = simulation["GWF_1"]
    write_context1 = WriteContext()
    actual = gwfmodel._render("GWF_1", write_context1)
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
    validation_context = ValidationSettings(True)
    write_context2 = WriteContext(tmp_path)
    gwfmodel._write("GWF_1", globaltimes, write_context2, validation_context)
    assert (tmp_path / "GWF_1" / "GWF_1.nam").is_file()
    assert (tmp_path / "GWF_1").is_dir()


def test_simulation_write_and_run_transport(circle_model_transport, tmp_path):
    simulation = circle_model_transport

    with pytest.raises(
        RuntimeError, match="Simulation circle has not been written yet."
    ):
        circle_model_transport.run()

    modeldir = tmp_path / "circle_transport"
    simulation.write(modeldir)
    simulation.run()
    head = simulation.open_head()
    concentration = simulation.open_concentration()

    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (52, 2, 216)

    assert isinstance(concentration, xu.UgridDataArray)
    assert concentration.dims == ("time", "layer", "mesh2d_nFaces")
    assert concentration.shape == (52, 2, 216)


def test_simulation_write_and_run_transport_vsc(circle_model_transport_vsc, tmp_path):
    simulation = circle_model_transport_vsc

    with pytest.raises(
        RuntimeError, match="Simulation circle has not been written yet."
    ):
        circle_model_transport_vsc.run()

    modeldir = tmp_path / "circle_transport"
    simulation.write(modeldir)
    simulation.run()
    head = simulation.open_head()
    concentration = simulation.open_concentration()

    assert isinstance(head, xu.UgridDataArray)
    assert head.dims == ("time", "layer", "mesh2d_nFaces")
    assert head.shape == (52, 2, 216)

    assert isinstance(concentration, xu.UgridDataArray)
    assert concentration.dims == ("time", "layer", "mesh2d_nFaces")
    assert concentration.shape == (52, 2, 216)


def test_simulation_clip_and_state_at_boundary(circle_model_transport, tmp_path):
    # Arrange
    simulation = circle_model_transport
    idomain = simulation["GWF_1"]["disv"]["idomain"].compute()

    simulation.write(tmp_path / "full")
    simulation.run()
    head = simulation.open_head().compute().reindex_like(idomain)
    concentration = simulation.open_concentration().compute().reindex_like(idomain)

    states_for_boundary = {
        "GWF_1": head.isel(time=-1, drop=True),
        "transport": concentration.isel(time=-1, drop=True),
    }
    # Act
    half_simulation = simulation.clip_box(
        x_max=0.1, states_for_boundary=states_for_boundary
    )
    # Assert
    # Test if model dims halved
    idomain_half = half_simulation["GWF_1"]["disv"]["idomain"]
    dim = idomain_half.grid.face_dimension
    np.testing.assert_array_equal(idomain_half.sizes[dim] / idomain.sizes[dim], 0.5)
    assert (
        half_simulation["transport"]["cnc_clipped"]["concentration"].notnull().sum()
        == 20
    )
    assert half_simulation["GWF_1"]["chd_clipped"]["head"].notnull().sum() == 20
    # Test if model runs
    half_simulation.write(tmp_path / "half")
    half_simulation.run()
    # Test if the clipped model output has the correct dimension
    head_half = half_simulation.open_head().compute().reindex_like(idomain_half)
    concentration_half = (
        half_simulation.open_concentration().compute().reindex_like(idomain_half)
    )
    assert head_half.shape == (52, 2, 108)
    assert concentration_half.shape == (52, 2, 108)
