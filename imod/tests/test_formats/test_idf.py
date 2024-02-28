import numpy as np
import pytest
import xarray as xr
from pytest import approx
from pytest_cases import case, parametrize, parametrize_with_cases

from imod import idf, util


@pytest.fixture(scope="module", params=[np.float32, np.float64])
def test_da(request):
    nrow, ncol = 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "test", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=request.param)
    da = xr.DataArray(data, **kwargs)
    return da


@pytest.fixture(scope="module")
def test_da_nonequidistant():
    nrow, ncol = 3, 4
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.3, -0.7, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    kwargs = {"name": "nonequidistant", "coords": coords, "dims": ("y", "x")}
    data = np.ones((nrow, ncol), dtype=np.float32)
    return xr.DataArray(data, **kwargs)


@pytest.fixture(scope="module")
def test_layerda():
    nlay, nrow, ncol = 5, 3, 4
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    coords["layer"] = np.arange(nlay) + 1
    kwargs = {"name": "layer", "coords": coords, "dims": ("layer", "y", "x")}
    data = np.ones((nlay, nrow, ncol), dtype=np.float32)
    da = xr.DataArray(data, **kwargs)
    return da


def dxdy(equidistant: bool):
    if equidistant:
        dx, dy = (1.0, -1.0)
    else:
        dy = np.array([-0.5, -1.5] * 2)
        dx = np.array([1.0] * 5)
    return dx, dy


def dxdy_full(equidistant: bool):
    if equidistant:
        dx, dy = (1.0, -1.0)
    else:
        dy = np.array([-0.5, -1.5] * 3)
        dx = np.array([1.0] * 8)
    return dx, dy


class SubdomainCases:
    def create_da(self, subdomain_factor: int = 0, equidistant: bool = True):
        nspecies, nlayer, nrow, ncol = (2, 3, 4, 5)
        dx, dy = dxdy(equidistant=equidistant)
        layer = [1, 2, 3]
        species = [1, 2]
        xmin = (0.0, 3.0, 3.0, 0.0)
        xmax = (5.0, 8.0, 8.0, 5.0)
        ymin = (0.0, 2.0, 0.0, 2.0)
        ymax = (4.0, 6.0, 4.0, 6.0)
        data = np.ones((nspecies, nlayer, nrow, ncol), dtype=np.float32)

        kwargs = {"name": "subdomains", "dims": ("species", "layer", "y", "x")}

        das = []
        for i, subd_extent in enumerate(zip(xmin, xmax, ymin, ymax)):
            kwargs["coords"] = util._xycoords(subd_extent, (dx, dy))
            kwargs["coords"]["layer"] = layer
            kwargs["coords"]["species"] = species
            da_data = data + i * subdomain_factor
            das.append(xr.DataArray(da_data, **kwargs))

        return das

    @case(tags="no_species")
    @parametrize(equidistant=[True, False])
    def case_constant(self, equidistant):
        das = [da.sel(species=1, drop=True) for da in self.create_da(0, equidistant)]
        expected = np.ones((3, 6, 8))
        return das, expected, equidistant

    @case(tags="no_species")
    @parametrize(equidistant=[True, False])
    def case_labeled(self, equidistant):
        das = [da.sel(species=1, drop=True) for da in self.create_da(1, equidistant)]
        expected = np.ones((3, 6, 8))
        expected[..., 0:4, 3:] = 2
        expected[..., 2:, 3:] = 3
        expected[..., 0:4, 0:5] = 4
        return das, expected, equidistant

    @case(tags="species")
    @parametrize(equidistant=[True, False])
    def case_constant_species(self, equidistant):
        das = self.create_da(0, equidistant)
        expected = np.ones((2, 3, 6, 8))
        return das, expected, equidistant

    @case(tags="species")
    @parametrize(equidistant=[True, False])
    def case_labeled_species(self, equidistant):
        das = self.create_da(1, equidistant)
        expected = np.ones((2, 3, 6, 8))
        expected[..., 0:4, 3:] = 2
        expected[..., 2:, 3:] = 3
        expected[..., 0:4, 0:5] = 4
        return das, expected, equidistant


def _save_subdomains_no_species(subdomains, tmp_path):
    for i, subdomain in enumerate(subdomains):
        for layer, da in subdomain.groupby("layer"):
            idf.write(tmp_path / f"subdomains_20000101_l{layer}_p00{i}.idf", da)


def _save_subdomains_species(subdomains, tmp_path):
    for i, subdomain in enumerate(subdomains):
        for species, das in subdomain.groupby("species"):
            for layer, da in das.groupby("layer"):
                idf.write(
                    tmp_path / f"subdomains_c{species}_20000101_l{layer}_p00{i}.idf", da
                )


@parametrize_with_cases(
    "subdomains,expected,equidistant", cases=SubdomainCases, has_tag="no_species"
)
def test_open_subdomains(subdomains, expected, equidistant, tmp_path):
    _save_subdomains_no_species(subdomains, tmp_path)

    da = idf.open_subdomains(tmp_path / "subdomains_*.idf").load()

    dx, dy = dxdy_full(equidistant)
    expected_coords = util._xycoords((0.0, 8.0, 0.0, 6.0), (dx, dy))

    assert da.dims == ("time", "layer", "y", "x")

    assert np.all(da.isel(time=0) == expected)
    assert len(da.x) == 8
    assert len(da.y) == 6

    assert np.all(da["y"].values == expected_coords["y"])
    assert np.all(da["x"].values == expected_coords["x"])
    assert np.all(da["dx"].values == dx)
    assert np.all(da["dy"].values == dy)

    assert da.values.dtype == np.float32

    assert isinstance(da, xr.DataArray)


@parametrize_with_cases(
    "subdomains,expected,equidistant", cases=SubdomainCases, has_tag="no_species"
)
def test_open_subdomains_pattern_None(subdomains, expected, equidistant, tmp_path):
    """Read without provided pattern, function should interpet dimensions correctly"""
    _save_subdomains_no_species(subdomains, tmp_path)
    # Test with pattern is None
    da = idf.open_subdomains(tmp_path / "subdomains_*.idf").load()

    assert da.dims == ("time", "layer", "y", "x")

    assert np.all(da.isel(time=0) == expected)


@parametrize_with_cases(
    "subdomains,expected,equidistant", cases=SubdomainCases, has_tag="species"
)
def test_open_subdomains_species(subdomains, expected, equidistant, tmp_path):
    _save_subdomains_species(subdomains, tmp_path)

    # Test with pattern
    pattern = r"{name}_c{species}_{time}_l{layer}_p{subdomain}"

    da = idf.open_subdomains(tmp_path / "subdomains_*.idf", pattern=pattern).load()

    dx, dy = dxdy_full(equidistant)
    expected_coords = util._xycoords((0.0, 8.0, 0.0, 6.0), (dx, dy))

    assert da.dims == ("species", "time", "layer", "y", "x")

    assert np.all(da.isel(time=0) == expected)
    assert len(da.x) == 8
    assert len(da.y) == 6

    assert np.all(da["y"].values == expected_coords["y"])
    assert np.all(da["x"].values == expected_coords["x"])
    assert np.all(da["dx"].values == dx)
    assert np.all(da["dy"].values == dy)

    assert da.values.dtype == np.float32

    assert isinstance(da, xr.DataArray)


@parametrize_with_cases(
    "subdomains,expected,_", cases=SubdomainCases, has_tag="species"
)
def test_open_subdomains_species_pattern_None(subdomains, expected, _, tmp_path):
    """Read without provided pattern, function should interpet dimensions correctly"""
    _save_subdomains_species(subdomains, tmp_path)

    # Test with pattern is None
    da = idf.open_subdomains(tmp_path / "subdomains_*.idf").load()

    assert da.dims == ("species", "time", "layer", "y", "x")

    assert np.all(da.isel(time=0) == expected)


@parametrize_with_cases(
    "subdomains,expected,equidistant", cases=SubdomainCases, has_tag="no_species"
)
def test_open_subdomains_error(subdomains, expected, equidistant, tmp_path):
    _save_subdomains_no_species(subdomains, tmp_path)

    # Add an additional subdomain with only one layer
    idf.write(tmp_path / "subdomains_20000101_l1_p010.idf", subdomains[0].sel(layer=1))

    with pytest.raises(ValueError):
        idf.open_subdomains(tmp_path / "subdomains_*.idf")


def test_xycoords_equidistant():
    dx, dy = 1.0, -1.0
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    assert np.allclose(coords["x"], np.arange(xmin + dx / 2.0, xmax, dx))
    assert np.allclose(coords["y"], np.arange(ymax + dy / 2.0, ymin, dy))
    assert coords["dx"] == dx
    assert coords["dy"] == dy


def test_xycoords_nonequidistant():
    dx = np.array([0.9, 1.1, 0.8, 1.2])
    dy = np.array([-1.3, -0.7, -1.0])
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 3.0
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    assert np.allclose(coords["x"], np.array([0.45, 1.45, 2.4, 3.4]))
    assert np.allclose(coords["y"], np.array([2.35, 1.35, 0.5]))
    assert coords["dx"][0] == "x"
    assert np.allclose(coords["dx"][1], dx)
    assert coords["dy"][0] == "y"
    assert np.allclose(coords["dy"][1], dy)


def test_xycoords_equidistant_array():
    dx = np.array([2.0, 2.0, 2.0, 2.0])
    dy = np.array([-0.5, -0.500001, -0.5])
    xmin, xmax = 0.0, 8.0
    ymin, ymax = 0.0, 1.5
    coords = util._xycoords((xmin, xmax, ymin, ymax), (dx, dy))
    assert np.allclose(coords["x"], np.arange(xmin + 1.0, xmax, 2.0))
    assert np.allclose(coords["y"], np.arange(ymax - 0.25, ymin, -0.5))
    assert coords["dx"] == approx(2.0)
    assert coords["dy"] == approx(-0.5)


def test_saveopen__nonequidistant(test_da_nonequidistant, tmp_path):
    idf.save(tmp_path / "nonequidistant", test_da_nonequidistant)
    assert (tmp_path / "nonequidistant.idf").exists()
    da = idf.open(tmp_path / "nonequidistant.idf")
    assert isinstance(da, xr.DataArray)
    assert np.array_equal(da, test_da_nonequidistant)
    # since the coordinates are created in float64 and stored in float32,
    # we lose some precision, which we have to allow for here
    xr.testing.assert_allclose(da, test_da_nonequidistant)


def test_save_topbot__single_layer(test_da, tmp_path):
    da = test_da
    da = da.assign_coords(z=0.5)
    da = da.assign_coords(dz=1.0)
    idf.save(tmp_path / "test", da)
    da_read = idf.open(tmp_path / "test.idf")
    assert da_read["z"] == approx(0.5)
    assert da_read["dz"] == approx(1.0)


def test_save_topbot__layers(test_layerda, tmp_path):
    da = test_layerda
    da = da.assign_coords(z=("layer", np.arange(1.0, 6.0) - 0.5))
    idf.save(tmp_path / "layer", da)
    da_l1 = idf.open(tmp_path / "layer_l1.idf")
    assert da_l1["z"] == approx(0.5)
    assert da_l1["dz"] == approx(1.0)
    da_l2 = idf.open(tmp_path / "layer_l2.idf")
    assert da_l2["z"] == approx(1.5)
    assert da_l2["dz"] == approx(1.0)
    # Read multiple idfs
    actual = idf.open(tmp_path / "layer_l*.idf")
    assert np.allclose(actual["z"], da["z"])
    assert actual["dz"] == approx(1.0)


def test_save_topbot__layers_nonequidistant(test_layerda, tmp_path):
    da = test_layerda
    dz = np.arange(-1.0, -6.0, -1.0)
    z = np.cumsum(dz) - 0.5 * dz
    da = da.assign_coords(z=("layer", z))
    da = da.assign_coords(dz=("layer", dz))
    idf.save(tmp_path / "layer", da)
    # Read multiple idfs
    actual = idf.open(tmp_path / "layer_l*.idf")
    assert np.allclose(actual["z"], da["z"])
    assert np.allclose(actual["dz"], da["dz"])


def test_save_topbot__only_z(test_layerda, tmp_path):
    da = test_layerda
    da = da.assign_coords(z=("layer", np.arange(1.0, 6.0) - 0.5))
    da = da.swap_dims({"layer": "z"})
    da = da.drop_vars("layer")
    idf.save(tmp_path / "layer", da)
    da_l1 = idf.open(tmp_path / "layer_l1.idf")
    assert da_l1["z"] == approx(0.5)
    assert da_l1["dz"] == approx(1.0)
    da_l2 = idf.open(tmp_path / "layer_l2.idf")
    assert da_l2["z"] == approx(1.5)
    assert da_l2["dz"] == approx(1.0)


def test_save_topbot__errors(test_layerda, tmp_path):
    da = test_layerda
    # non-equidistant, cannot infer dz
    z = np.array([0.0, -1.0, -3.0, -4.5, -5.0])
    da = da.assign_coords(z=("layer", z))
    with pytest.raises(ValueError):
        idf.save(tmp_path / "layer", da)


def test_saveopen_dtype(test_da, tmp_path):
    da = test_da
    idf.save(tmp_path / "dtype", da, dtype=np.float32)
    backda = idf.open(tmp_path / "dtype.idf")
    assert backda.dtype == np.float32

    idf.save(tmp_path / "dtype", da, dtype=np.float64)
    backda = idf.open(tmp_path / "dtype.idf")
    assert backda.dtype == np.float64


def test_dtype_error(test_da, tmp_path):
    da = test_da
    with pytest.raises(ValueError):
        idf.save(tmp_path / "integer", da, dtype=np.int32)

    with pytest.raises(ValueError):
        idf.write(tmp_path / "integer.idf", da, dtype=np.int32)


def test_nodata(test_da, tmp_path):
    da = test_da
    test_da[...] = np.nan
    path = tmp_path / "nodata.idf"
    idf.write(path, da, dtype=np.float32, nodata=1.0e20)

    header = idf.header(path, pattern=None)
    size = header["nrow"] * header["ncol"]
    with open(path) as f:
        f.seek(header["headersize"])
        a = np.fromfile(f, np.float32, size)

    assert not np.isnan(a).any()


def test_save_open_arbitrary_4D(tmp_path):
    da = xr.DataArray(
        data=np.ones((5, 3, 3, 2)),
        coords={
            "bndvalue": np.arange(5.0),
            "porosity": ["0low", "1mid", "2high"],
            "y": [2.5, 1.5, 0.5],
            "x": [0.0, 2.0],
        },
        dims=["bndvalue", "porosity", "y", "x"],
    )

    idf.save(
        tmp_path / "interface", da, pattern="{name}_{bndvalue:0f}_{porosity}{extension}"
    )
    back = idf.open(tmp_path / "interface*.idf", pattern="{name}_{bndvalue}_{porosity}")

    # Test whether it has worked properly, value testing is annoying:
    # coordinates may be shuffled since IDF doesn't provide any conventions
    # here.
    assert isinstance(back, xr.DataArray)
    # Might get shuffled dimensions for the same reason.
    assert set(da.dims) == set(back.dims)
