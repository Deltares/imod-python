from pathlib import Path

import pytest
from numpy.testing import assert_almost_equal, assert_equal

from imod import msw


@pytest.mark.usefixtures("msw_model")
def test_msw_model_write(msw_model, tmp_path):
    output_dir = tmp_path / "metaswap"
    msw_model.write(output_dir)

    assert len(list(output_dir.rglob(r"*.inp"))) == 16
    assert len(list(output_dir.rglob(r"*.asc"))) == 4


@pytest.mark.usefixtures("msw_model")
def test_get_starttime(msw_model):
    year, time_since_start_year = msw_model._get_starttime()

    assert_equal(year, 1971)
    assert_almost_equal(time_since_start_year, 0.0)


@pytest.mark.usefixtures("msw_model")
def test_get_pkgkey(msw_model):
    pkg_id = msw_model._get_pkg_key(msw.GridData)

    assert pkg_id == "grid"


@pytest.mark.usefixtures("msw_model")
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
