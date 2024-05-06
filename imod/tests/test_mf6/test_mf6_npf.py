import pathlib
import re
import textwrap
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError
from imod.tests.test_mf6.test_mf6_dis import _load_imod5_data_in_memory


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

    _load_imod5_data_in_memory(data)
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

    _load_imod5_data_in_memory(data)
    target_grid = data["khv"]["kh"]
    data["ani"]["angle"].values[:, :, :] = 135.0
    data["ani"]["factor"].values[:, :, :] = 0.1
    npf = imod.mf6.NodePropertyFlow.from_imod5_data(data, target_grid)

    # Test array values  for k22 and angle1
    for layer in npf.dataset["k"].coords["layer"].values:
        k_layer = npf.dataset["k"].sel({"layer": layer})
        k22_layer = npf.dataset["k22"].sel({"layer": layer})
        angle1_layer = npf.dataset["angle1"].sel({"layer": layer})

        k_layer = xr.where(np.isnan(k_layer), 0.0, k_layer)
        k22_layer = xr.where(np.isnan(k22_layer), 0.0, k22_layer)
        angle1_layer = xr.where(np.isnan(angle1_layer), 0.0, angle1_layer)

        if layer in data["ani"]["factor"].coords["layer"].values:
            np.testing.assert_allclose(
                k_layer.values * 0.1, k22_layer.values, atol=1e-10
            )
            assert np.all(angle1_layer.values == 315.0)
        else:
            assert np.all(k_layer.values == k22_layer.values)
            assert np.all(angle1_layer.values == 0.0)

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

    _load_imod5_data_in_memory(data)
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
