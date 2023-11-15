import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


@pytest.fixture()
def idomain():
    nlay = 3
    nrow = 15
    ncol = 15
    shape = (nlay, nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    # Discretization data
    return xr.DataArray(np.ones(shape), coords=coords, dims=dims)


@pytest.fixture()
def sy_layered(idomain):
    layer = idomain["layer"].values
    return xr.DataArray([0.16, 0.15, 0.14], {"layer": layer}, ("layer",))


@pytest.fixture()
def convertible(idomain):
    return xr.full_like(idomain, 0, dtype=int)


@pytest.fixture(scope="function")
def dis(idomain):
    top = idomain.sel(layer=1)
    bottom = idomain - xr.DataArray(
        data=[1.0, 2.0], dims=("layer",), coords={"layer": [2, 3]}
    )

    return imod.mf6.StructuredDiscretization(
        top=top, bottom=bottom, idomain=idomain.astype(int)
    )


def test_render_specific_storage(sy_layered, convertible):
    # for better coverage, use full (conv) layered (sy) and constant (ss)
    sto = imod.mf6.SpecificStorage(
        specific_storage=0.0003,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
    )

    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = sto.render(directory, "sto", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin griddata
          iconvert
            open/close mymodel/sto/iconvert.bin (binary)
          ss
            constant 0.0003
          sy layered
            constant 0.16
            constant 0.15
            constant 0.14
        end griddata

        begin period 1
          transient
        end period
        """
    )
    assert actual == expected


def test_render_specific_storage_save_flows(sy_layered, convertible):
    # for better coverage, use full (conv) layered (sy) and constant (ss)
    sto = imod.mf6.SpecificStorage(
        specific_storage=0.0003,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
        save_flows=True,
    )

    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = sto.render(directory, "sto", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          save_flows
        end options

        begin griddata
          iconvert
            open/close mymodel/sto/iconvert.bin (binary)
          ss
            constant 0.0003
          sy layered
            constant 0.16
            constant 0.15
            constant 0.14
        end griddata

        begin period 1
          transient
        end period
        """
    )
    assert actual == expected


def test_render_specific_storage_three_periods(sy_layered, convertible):
    # again but starting with two steady-state periods, followed by a transient stress period
    times = np.array(["2000-01-01", "2000-01-03"], dtype="datetime64[ns]")
    transient = xr.DataArray([False, True], {"time": times}, ("time",))

    sto = imod.mf6.SpecificStorage(
        specific_storage=0.0003,
        specific_yield=sy_layered,
        transient=transient,
        convertible=convertible,
    )

    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )
    actual = sto.render(directory, "sto", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin griddata
          iconvert
            open/close mymodel/sto/iconvert.bin (binary)
          ss
            constant 0.0003
          sy layered
            constant 0.16
            constant 0.15
            constant 0.14
        end griddata

        begin period 1
          steady-state
        end period
        begin period 3
          transient
        end period
        """
    )
    assert actual == expected


def test_render_storage_coefficient(sy_layered, convertible):
    # for better coverage, use full (conv) layered (sy) and constant (ss)
    sto = imod.mf6.StorageCoefficient(
        storage_coefficient=0.0003,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
    )

    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = sto.render(directory, "sto", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          storagecoefficient
        end options

        begin griddata
          iconvert
            open/close mymodel/sto/iconvert.bin (binary)
          ss
            constant 0.0003
          sy layered
            constant 0.16
            constant 0.15
            constant 0.14
        end griddata

        begin period 1
          transient
        end period
        """
    )
    assert actual == expected


def test_render_storage_coefficient_save_flows(sy_layered, convertible):
    # for better coverage, use full (conv) layered (sy) and constant (ss)
    sto = imod.mf6.StorageCoefficient(
        storage_coefficient=0.0003,
        specific_yield=sy_layered,
        transient=True,
        save_flows=True,
        convertible=convertible,
    )

    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = sto.render(directory, "sto", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          save_flows
          storagecoefficient
        end options

        begin griddata
          iconvert
            open/close mymodel/sto/iconvert.bin (binary)
          ss
            constant 0.0003
          sy layered
            constant 0.16
            constant 0.15
            constant 0.14
        end griddata

        begin period 1
          transient
        end period
        """
    )
    assert actual == expected


def test_storage_deprecation_warning(sy_layered, convertible):
    with pytest.raises(NotImplementedError):
        imod.mf6.Storage(
            specific_storage=0.0003,
            specific_yield=sy_layered,
            transient=True,
            convertible=convertible,
        )


def test_wrong_dtype_sc(sy_layered, convertible):
    with pytest.raises(ValidationError):
        imod.mf6.StorageCoefficient(
            storage_coefficient=0,
            specific_yield=sy_layered,
            transient=True,
            convertible=convertible,
        )


def test_wrong_dtype_ss(sy_layered, convertible):
    with pytest.raises(ValidationError):
        imod.mf6.SpecificStorage(
            specific_storage=0,
            specific_yield=sy_layered,
            transient=True,
            convertible=convertible,
        )


def test_validate_false(sy_layered, convertible):
    imod.mf6.SpecificStorage(
        specific_storage=0,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
        validate=False,
    )

    imod.mf6.StorageCoefficient(
        storage_coefficient=0,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
        validate=False,
    )


def test_validate_transient_wrong_dtype(sy_layered, convertible):
    # Wrong dtype for transient (string instead of bool)
    times = np.array(["2000-01-01", "2000-01-03"], dtype="datetime64[ns]")
    transient = xr.DataArray(["False", "True"], {"time": times}, ("time",))

    with pytest.raises(ValidationError):
        imod.mf6.SpecificStorage(
            specific_storage=0.0003,
            specific_yield=sy_layered,
            transient=transient,
            convertible=convertible,
        )


def test_validate_transient_wrong_dim(sy_layered, convertible):
    # Wrong shape for transient {"time", "layer"} instead of {"time",}
    times = np.array(["2000-01-01", "2000-01-03"], dtype="datetime64[ns]")
    transient = xr.DataArray([False, True], {"time": times}, ("time",))
    # Create boolean DataArray with layer coordinate for broadcasting
    layer = sy_layered.coords["layer"]
    boolean_shape = layer == layer
    # Broadcast
    transient = transient & boolean_shape

    with pytest.raises(ValidationError):
        imod.mf6.SpecificStorage(
            specific_storage=0.0003,
            specific_yield=sy_layered,
            transient=transient,
            convertible=convertible,
        )


def test_wrong_value_ss(sy_layered, convertible, dis):
    sto = imod.mf6.SpecificStorage(
        specific_storage=-0.1,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
    )

    errors = sto._validate(sto._write_schemata, **dis.dataset)

    assert len(errors) == 1

    for var, error in errors.items():
        assert var == "specific_storage"


def test_wrong_value_sc(sy_layered, convertible, dis):
    sto = imod.mf6.StorageCoefficient(
        storage_coefficient=-0.1,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
    )

    errors = sto._validate(sto._write_schemata, **dis.dataset)

    assert len(errors) == 1

    for var, error in errors.items():
        assert var == "storage_coefficient"


def test_check_nan_in_active_cell(sy_layered, convertible, dis):
    storage_coefficient = dis["idomain"] * 0.1

    sto = imod.mf6.StorageCoefficient(
        storage_coefficient=storage_coefficient,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
    )

    errors = sto._validate(sto._write_schemata, **dis.dataset)

    assert len(errors) == 0

    storage_coefficient[1, 1, 1] = np.nan

    sto = imod.mf6.StorageCoefficient(
        storage_coefficient=storage_coefficient,
        specific_yield=sy_layered,
        transient=True,
        convertible=convertible,
    )

    errors = sto._validate(sto._write_schemata, **dis.dataset)

    assert len(errors) == 1

    for var, error in errors.items():
        assert var == "storage_coefficient"
