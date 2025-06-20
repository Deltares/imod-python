"""
QGIS user acceptance test, these are pytest-marked with 'qgis_export'.

"""

from pathlib import Path

import pytest
import xugrid as xu

import imod


@pytest.mark.qgis_export
def test_mf6_qgis_export__structured(tmp_path):
    mf6_sim = imod.data.hondsrug_simulation(tmp_path / "unzipped")

    path_dumped = Path(tmp_path) / "hondsrug_MDAL"
    mf6_sim.dump(path_dumped, mdal_compliant=True, crs="EPSG:28992")
    print(f"Dumped to: {path_dumped}")


@pytest.mark.qgis_export
def test_mf6_qgis_export__unstructured(tmp_path):
    mf6_sim = imod.data.hondsrug_simulation(tmp_path / "unzipped")
    idomain = mf6_sim["GWF"]["dis"]["idomain"]
    grid = xu.UgridDataArray.from_structured2d(idomain)

    mf6_sim_unstructured = mf6_sim.regrid_like("unstructured_model", grid)
    path_dumped = Path(tmp_path) / "hondsrug_MDAL"
    mf6_sim_unstructured.dump(path_dumped, mdal_compliant=True, crs="EPSG:28992")
    print(f"Dumped to: {path_dumped}")
