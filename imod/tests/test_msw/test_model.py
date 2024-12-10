from copy import copy
from pathlib import Path

import pytest
import xarray as xr
from numpy.testing import assert_almost_equal, assert_equal
from pytest_cases import parametrize_with_cases

from imod import mf6, msw
from imod.mf6.utilities.regrid import RegridderWeightsCache
from imod.msw.model import DEFAULT_SETTINGS
from imod.typing import GridDataArray, Imod5DataDict


def test_msw_model_write(msw_model, coupled_mf6_model, coupled_mf6wel, tmp_path):
    mf6_dis = coupled_mf6_model["GWF_1"]["dis"]

    output_dir = tmp_path / "metaswap"
    msw_model.write(output_dir, mf6_dis, coupled_mf6wel)

    assert len(list(output_dir.rglob(r"*.inp"))) == 16
    assert len(list(output_dir.rglob(r"*.asc"))) == 4


def test_get_starttime(msw_model):
    year, time_since_start_year = msw_model._get_starttime()

    assert_equal(year, 1971)
    assert_almost_equal(time_since_start_year, 0.0)


def test_get_pkgkey(msw_model):
    pkg_id = msw_model._get_pkg_key(msw.GridData)

    assert pkg_id == "grid"


def test_check_required_packages(msw_model):
    # Should not throw error
    msw_model._check_required_packages()

    # Remove essential package
    msw_model.pop("grid")

    with pytest.raises(ValueError):
        msw_model._check_required_packages()


def test_check_vegetation_indices_in_annual_crop_factors(msw_model):
    msw_model._check_vegetation_indices_in_annual_crop_factors()

    # Remove one vegetation index from crop factors
    msw_model["crop_factors"].dataset = msw_model["crop_factors"].dataset.sel(
        vegetation_index=[1, 2]
    )

    with pytest.raises(ValueError):
        msw_model._check_vegetation_indices_in_annual_crop_factors()


def test_check_landuse_indices_in_lookup_options(msw_model):
    msw_model._check_landuse_indices_in_lookup_options()

    # Remove one vegetation index from crop factors
    msw_model["landuse_options"].dataset = msw_model["landuse_options"].dataset.sel(
        landuse_index=[2, 3]
    )

    with pytest.raises(ValueError):
        msw_model._check_landuse_indices_in_lookup_options()


def test_render_unsat_database_path(msw_model, tmp_path):
    rel_path = msw_model._render_unsaturated_database_path("./unsat_database")

    assert rel_path[0] == '"'
    assert rel_path[1] == "$"
    assert rel_path[-2] == "\\"
    assert rel_path[-1] == '"'
    assert rel_path == '"$unsat_database\\"'

    abs_path = msw_model._render_unsaturated_database_path(tmp_path.resolve())

    assert abs_path[0] == '"'
    assert abs_path[-2] == "\\"
    assert abs_path[-1] == '"'

    assert Path(abs_path.replace('"', "")).is_absolute()


def get_target_mf6_discretization():
    x = [1.0, 1.5, 2.0, 2.5, 3.0]
    y = [3.0, 2.5, 2.0, 1.5, 1.0]
    dx = 0.5
    dy = -0.5
    layer = [1, 2, 3]

    idomain = xr.DataArray(
        1,
        dims=("layer", "y", "x"),
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
    )

    top = 0.0
    bottom = xr.DataArray([-1.0, -21.0, -321.0], coords={"layer": layer}, dims="layer")

    dis = mf6.StructuredDiscretization(top=top, bottom=bottom, idomain=idomain)
    return dis


def test_model_regrid(msw_model, coupled_mf6wel, tmp_path):
    """
    Test where only msw model is regridded, modflow 6 wells placed in same
    row/col number, thus change spatially.
    """
    mf6_discretization = get_target_mf6_discretization()

    regrid_context = RegridderWeightsCache()
    regridded_msw_model = msw_model.regrid_like(mf6_discretization, regrid_context)
    regridded_msw_model.write(tmp_path, mf6_discretization, coupled_mf6wel)


def test_coupled_model_regrid(msw_model, coupled_mf6_model, tmp_path):
    """
    Test where only msw model is regridded, modflow 6 wells placed in same
    row/col number, thus change spatially.
    """
    mf6_discretization = get_target_mf6_discretization()

    regrid_context = RegridderWeightsCache()
    regridded_msw_model = msw_model.regrid_like(mf6_discretization, regrid_context)
    regridded_mf6_model = coupled_mf6_model.regrid_like(
        "regridded", mf6_discretization["idomain"]
    )
    regridded_npf = regridded_mf6_model["GWF_1"]["npf"]
    grid_agnostic_well = coupled_mf6_model["GWF_1"]["well_msw"]
    regridded_mf6_wel = grid_agnostic_well.to_mf6_pkg(
        mf6_discretization["idomain"],
        mf6_discretization["top"],
        mf6_discretization["bottom"],
        regridded_npf["k"],
    )

    regridded_msw_model.write(tmp_path, mf6_discretization, regridded_mf6_wel)


def setup_written_meteo_grids(
    meteo_grids: tuple[GridDataArray], tmp_path: Path
) -> Path:
    precipitation, _ = meteo_grids
    meteo_grid = msw.MeteoGrid(precipitation, precipitation)
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir(exist_ok=True, parents=True)
    meteo_grid.write(grid_dir)
    return grid_dir


def setup_parasim_inp(directory: Path):
    settings = copy(DEFAULT_SETTINGS)
    settings["iybg"] = 2000
    settings["tdbg"] = 200.45
    settings["unsa_svat_path"] = str(directory)

    filename = directory / "para_sim.inp"
    with open(filename, "w") as f:
        rendered = msw.MetaSwapModel._template.render(settings=settings)
        f.write(rendered)
    return filename


def write_test_files(directory: Path, filenames: list[str]) -> list[Path]:
    paths = [directory / filename for filename in filenames]
    for p in paths:
        with open(p, mode="w") as f:
            f.write("test")
    return paths


def setup_extra_files(meteo_grids: tuple[GridDataArray], directory: Path):
    grid_dir = setup_written_meteo_grids(meteo_grids, directory)
    setup_parasim_inp(grid_dir)
    write_test_files(grid_dir, ["a.inp", "b.inp"])
    return {
        "paths": [
            [str(grid_dir / fn)]
            for fn in ["a.inp", "mete_grid.inp", "para_sim.inp", "b.inp"]
        ]
    }


class Imod5DataCases:
    def case_grid(self, imod5_cap_data: Imod5DataDict) -> tuple[Imod5DataDict, bool]:
        has_scaling_factor = True
        return imod5_cap_data, has_scaling_factor

    def case_no_scaling_factors(
        self, imod5_cap_data: Imod5DataDict
    ) -> tuple[Imod5DataDict, bool]:
        has_scaling_factor = False
        cap_data = imod5_cap_data["cap"]
        # open_projectfile_data adds layer kwargs to constants
        layer_kwargs = {"coords": {"layer": [1]}, "dims": ("layer",)}
        cap_data["perched_water_table_level"] = xr.DataArray([-9999.0], **layer_kwargs)
        cap_data["soil_moisture_fraction"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["conductivitiy_factor"] = xr.DataArray([1.0], **layer_kwargs)
        return imod5_cap_data, has_scaling_factor

    def case_constants(
        self, imod5_cap_data: Imod5DataDict
    ) -> tuple[Imod5DataDict, bool]:
        has_scaling_factor = False
        cap_data = imod5_cap_data["cap"]
        # open_projectfile_data adds layer kwargs to constants
        layer_kwargs = {"coords": {"layer": [1]}, "dims": ("layer",)}
        cap_data["perched_water_table_level"] = xr.DataArray([-9999.0], **layer_kwargs)
        cap_data["soil_moisture_fraction"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["conductivitiy_factor"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["urban_ponding_depth"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["rural_ponding_depth"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["urban_runoff_resistance"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["rural_runoff_resistance"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["urban_runon_resistance"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["rural_runon_resistance"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["urban_infiltration_capacity"] = xr.DataArray([1.0], **layer_kwargs)
        cap_data["rural_infiltration_capacity"] = xr.DataArray([1.0], **layer_kwargs)
        return imod5_cap_data, has_scaling_factor


@parametrize_with_cases("imod5_data, has_scaling_factor", cases=Imod5DataCases)
def test_import_from_imod5(
    imod5_data: Imod5DataDict,
    has_scaling_factor: bool,
    meteo_grids: tuple[GridDataArray],
    coupled_mf6_model: mf6.Modflow6Simulation,
    tmp_path: Path,
):
    # Arrange
    imod5_data["extra"] = setup_extra_files(meteo_grids, tmp_path)
    times = coupled_mf6_model["time_discretization"].dataset.coords["time"]
    dis_pkg = coupled_mf6_model["GWF_1"]["dis"]
    # Act
    model = msw.MetaSwapModel.from_imod5_data(imod5_data, dis_pkg, times)
    # Assert
    grid_packages = {
        "grid",
        "infiltration",
        "ponding",
        "sprinkling",
        "idf_mapping",
    }
    expected_keys = grid_packages | {
        "meteo_grid",
        "prec_mapping",
        "evt_mapping",
        "coupling",
        "extra_files",
        "time_oc",
    }
    missing_keys = expected_keys - set(model.keys())
    assert not missing_keys
    for pkgname in grid_packages:
        # Test if all grid packages broadcasted to grid.
        missing_dims = {"y", "x"} - set(model[pkgname].dataset.dims.keys())
        assert not missing_dims
    assert "time" in model["time_oc"].dataset.dims.keys()
    assert len(model["meteo_grid"].dataset.dims) == 0
    assert ("scaling_factor" in model.keys()) == has_scaling_factor


@parametrize_with_cases("imod5_data, has_scaling_factor", cases=Imod5DataCases)
def test_import_from_imod5_and_write(
    imod5_data: Imod5DataDict,
    has_scaling_factor: bool,
    meteo_grids: tuple[GridDataArray],
    coupled_mf6_model: mf6.Modflow6Simulation,
    tmp_path: Path,
):
    # Arrange
    imod5_data["extra"] = setup_extra_files(meteo_grids, tmp_path)
    times = coupled_mf6_model["time_discretization"].dataset.coords["time"]
    dis_pkg = coupled_mf6_model["GWF_1"]["dis"]
    npf_pkg = coupled_mf6_model["GWF_1"]["npf"]
    active = dis_pkg["idomain"] == 1
    modeldir = tmp_path / "modeldir"
    # Act
    model = msw.MetaSwapModel.from_imod5_data(imod5_data, dis_pkg, times)
    well_pkg = mf6.LayeredWell.from_imod5_cap_data(imod5_data)
    mf6_wel_pkg = well_pkg.to_mf6_pkg(
        active, dis_pkg["top"], dis_pkg["bottom"], npf_pkg["k"]
    )
    model.write(modeldir, dis_pkg, mf6_wel_pkg, validate=False)

    # Assert
    expected_n_files = 13
    if has_scaling_factor:
        expected_n_files += 1
    assert len(list(modeldir.rglob(r"*.inp"))) == expected_n_files
