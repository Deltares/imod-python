import pathlib
import re
import textwrap
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

import imod
from imod.mf6.utilities.regridding_types import RegridderType
from imod.schemata import ValidationError


def test_render():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    npf = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
        alternative_cell_averaging="AMT-HMK",
    )
    directory = pathlib.Path("mymodel")
    actual = npf.render(directory, "npf", None, True)
    expected = textwrap.dedent(
        """\
        begin options
          save_flows
          alternative_cell_averaging AMT-HMK
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


def test_wrong_dtype():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    with pytest.raises(ValidationError):
        imod.mf6.NodePropertyFlow(
            icelltype=icelltype.astype(np.float64),
            k=k,
            k33=k33,
            variable_vertical_conductance=True,
            dewatered=True,
            perched=True,
            save_flows=True,
        )


def test_validate_false():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    imod.mf6.NodePropertyFlow(
        icelltype=icelltype.astype(np.float64),
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
        validate=False,
    )


def test_incompatible_setting():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    message = textwrap.dedent(
        """
        * rhs_option
        \t- Incompatible setting: xt3d_option should be True"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.NodePropertyFlow(
            icelltype=icelltype,
            k=k,
            k33=k33,
            variable_vertical_conductance=True,
            dewatered=True,
            perched=True,
            save_flows=True,
            xt3d_option=False,
            rhs_option=True,
            alternative_cell_averaging="AMT-HMK",
        )

    message = textwrap.dedent(
        """
        * alternative_cell_averaging
        \t- Incompatible setting: xt3d_option should be False"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.NodePropertyFlow(
            icelltype=icelltype,
            k=k,
            k33=k33,
            variable_vertical_conductance=True,
            dewatered=True,
            perched=True,
            save_flows=True,
            xt3d_option=True,
            alternative_cell_averaging="AMT-HMK",
        )


def test_wrong_dim():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray(
        [[1, 0, 0]],
        {"time": [1], "layer": layer},
        (
            "time",
            "layer",
        ),
    )
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    message = textwrap.dedent(
        """
        * icelltype
        \t- No option succeeded:
        \tdim mismatch: expected ('layer', 'y', 'x'), got ('time', 'layer')
        \tdim mismatch: expected ('layer', '{face_dim}'), got ('time', 'layer')
        \tdim mismatch: expected ('layer',), got ('time', 'layer')
        \tdim mismatch: expected (), got ('time', 'layer')"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.NodePropertyFlow(
            icelltype=icelltype,
            k=k,
            k33=k33,
            variable_vertical_conductance=True,
            dewatered=True,
            perched=True,
            save_flows=True,
        )


def test_configure_xt3d(tmp_path):
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    npf = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
        alternative_cell_averaging="AMT-HMK",
    )

    # assert xt3d off by default
    rendered = npf.render(tmp_path, "npf", None, True)
    assert "xt3d" not in rendered
    assert not npf.get_xt3d_option()

    # assert xt3d can be turned on
    npf.set_xt3d_option(True, True)
    rendered = npf.render(tmp_path, "npf", None, True)
    assert "xt3d" in rendered
    assert "rhs" in rendered
    assert npf.get_xt3d_option()

    # assert xt3d can be turned off
    npf.set_xt3d_option(False, False)
    rendered = npf.render(tmp_path, "npf", None, True)
    assert "xt3d" not in rendered
    assert "rhs" not in rendered
    assert not npf.get_xt3d_option()


@pytest.mark.usefixtures("imod5_dataset")
def test_npf_from_imod5_isotropic(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])
    # throw out kva (=vertical anisotropy array) and ani (=horizontal anisotropy array)
    data.pop("kva")
    data.pop("ani")

    target_grid = data["khv"]["kh"]
    npf = imod.mf6.NodePropertyFlow.from_imod5_data(data, target_grid)

    # Test array values are the same  for k ( disregarding the locations where k == np.nan)
    k_nan_removed = xr.where(np.isnan(npf.dataset["k"]), 0, npf.dataset["k"])
    np.testing.assert_allclose(k_nan_removed, data["khv"]["kh"].values)

    rendered_npf = npf.render(tmp_path, "npf", None, None)
    assert "k22" not in rendered_npf
    assert "k33" not in rendered_npf
    assert "angle1" not in rendered_npf
    assert "angle2" not in rendered_npf
    assert "angle3" not in rendered_npf


@pytest.mark.usefixtures("imod5_dataset")
def test_npf_from_imod5_horizontal_anisotropy(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])
    # throw out kva (=vertical anisotropy array)
    data.pop("kva")

    target_grid = data["khv"]["kh"]
    data["ani"]["angle"].values[:, :, :] = 135.0
    data["ani"]["factor"].values[:, :, :] = 0.1
    npf = imod.mf6.NodePropertyFlow.from_imod5_data(data, target_grid)

    # Test array values  for k22 and angle1
    for layer in npf.dataset["k"].coords["layer"].values:
        ds_layer = npf.dataset.sel({"layer": layer})

        ds_layer = ds_layer.fillna(0.0)

        if layer in data["ani"]["factor"].coords["layer"].values:
            np.testing.assert_allclose(
                ds_layer["k"].values * 0.1, ds_layer["k22"].values, atol=1e-10
            )
            assert np.all(ds_layer["angle1"].values == 315.0)
        else:
            assert np.all(ds_layer["k"].values == ds_layer["k22"].values)
            assert np.all(ds_layer["angle1"].values == 0.0)

    rendered_npf = npf.render(tmp_path, "npf", None, None)
    assert "k22" in rendered_npf
    assert "k33" not in rendered_npf
    assert "angle1" in rendered_npf
    assert "angle2" not in rendered_npf
    assert "angle3" not in rendered_npf


@pytest.mark.usefixtures("imod5_dataset")
def test_npf_from_imod5_vertical_anisotropy(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])
    # throw out ani (=horizontal anisotropy array)
    data.pop("ani")

    data["kva"]["vertical_anisotropy"].values[:] = 0.1
    target_grid = data["khv"]["kh"]

    npf = imod.mf6.NodePropertyFlow.from_imod5_data(data, target_grid)

    # Test array values  for k33
    for layer in npf.dataset["k"].coords["layer"].values:
        k_layer = npf.dataset["k"].sel({"layer": layer})
        k33_layer = npf.dataset["k33"].sel({"layer": layer})

        k_layer = xr.where(np.isnan(k_layer), 0.0, k_layer)
        k33_layer = xr.where(np.isnan(k33_layer), 0.0, k33_layer)
        np.testing.assert_allclose(k_layer.values * 0.1, k33_layer.values, atol=1e-10)

    rendered_npf = npf.render(tmp_path, "npf", None, None)
    assert "k22" not in rendered_npf
    assert "k33" in rendered_npf
    assert "angle1" not in rendered_npf
    assert "angle2" not in rendered_npf
    assert "angle3" not in rendered_npf


@pytest.mark.usefixtures("imod5_dataset")
def test_npf_from_imod5_settings(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])

    # move the coordinates a bit so that it doesn't match the grid of k (and the regridding settings will matter)
    target_grid = data["khv"]["kh"]
    x = target_grid["x"].values
    x += 50
    y = target_grid["y"].values
    y += 50
    target_grid = target_grid.assign_coords({"x": x, "y": y})

    settings = imod.mf6.NodePropertyFlow.get_regrid_methods()
    settings_1 = deepcopy(settings)
    settings_1.k = (
        RegridderType.OVERLAP,
        "harmonic_mean",
    )
    npf_1 = imod.mf6.NodePropertyFlow.from_imod5_data(data, target_grid, settings_1)

    settings_2 = deepcopy(settings)
    settings_2.k = (
        RegridderType.OVERLAP,
        "mode",
    )
    npf_2 = imod.mf6.NodePropertyFlow.from_imod5_data(data, target_grid, settings_2)

    # assert that different settings lead to different results.
    diff = npf_1.dataset["k"] - npf_2.dataset["k"]
    diff = xr.where(np.isnan(diff), 0, diff)
    assert diff.values.max() > 0.1
