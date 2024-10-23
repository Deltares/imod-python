from pathlib import Path

import pytest
import xarray as xr
from numpy.testing import assert_almost_equal, assert_equal

from imod import mf6, msw
from imod.mf6.utilities.regrid import RegridderWeightsCache


def test_msw_model_write(msw_model, tmp_path):
    output_dir = tmp_path / "metaswap"
    msw_model.write(output_dir)

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


def test_model_regrid(msw_model, tmp_path):
    # sprinkling not supported yet for regridding
    msw_model.pop("sprinkling")

    mf6_discretization = get_target_mf6_discretization()

    regrid_context = RegridderWeightsCache()
    regridded_model = msw_model.regrid_like(mf6_discretization, regrid_context)
    regridded_model.write(tmp_path)
