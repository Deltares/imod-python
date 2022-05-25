import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


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
    )
    directory = pathlib.Path("mymodel")
    actual = npf.render(directory, "npf", None, True)
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


def test_wrong_dtype():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    with pytest.raises(TypeError):
        imod.mf6.NodePropertyFlow(
            icelltype=icelltype.astype(np.float64),
            k=k,
            k33=k33,
            variable_vertical_conductance=True,
            dewatered=True,
            perched=True,
            save_flows=True,
        )


def test_to_disu():
    layer = [1, 2, 3]
    template = xr.full_like(
        imod.util.empty_3d(10.0, 0.0, 100.0, 10.0, 0.0, 100.0, layer), 1, dtype=np.int32
    )
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",)) * template
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",)) * template
    k33 = (
        xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",)) * template
    )

    npf = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
    )

    disu_npf = npf.to_disu()
    assert np.array_equal(disu_npf.dataset["node"], np.arange(300))

    # Test again, with partially active domain
    active = xr.full_like(template, True, dtype=bool)
    active[:, :, 0] = False
    n_active = active.sum()
    cell_ids = np.full((3, 10, 10), -1)
    cell_ids[active.values] = np.arange(n_active)
    cell_ids = cell_ids.ravel()

    disu_npf = npf.to_disu(cell_ids)
    assert np.array_equal(disu_npf.dataset["node"], np.arange(270))

    with pytest.raises(TypeError, match="cell_ids should be integer"):
        cell_ids = np.arange(3 * 5 * 5, dtype=np.float64)
        npf.to_disu(cell_ids)
