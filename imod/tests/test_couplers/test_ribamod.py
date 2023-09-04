from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

import imod
from imod.couplers.ribamod import RibaMod
from imod.couplers.ribamod.ribamod import DriverCoupling
from imod.mf6.model import GroundwaterFlowModel
from imod.mf6 import Drainage, River
from imod.mf6.simulation import Modflow6Simulation


# tomllib part of Python 3.11, else use tomli
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@pytest.fixture
def basin_definition(coupled_ribasim_mf6_model, ribasim_model) -> gpd.GeoDataFrame:
    _, mf6_model = get_mf6_gwf_modelnames(coupled_ribasim_mf6_model)[0]
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(
        mf6_model["dis"]["idomain"]
    )
    node_id = ribasim_model.basin.static["node_id"].unique()
    polygon = Polygon(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ]
    )
    return gpd.GeoDataFrame(data={"node_id": node_id}, geometry=[polygon])


def test_validate_keys(coupled_ribasim_mf6_model):
    with pytest.raises(ValueError, match="active and passive keys share members"):
        RibaMod.validate_keys(
            coupled_ribasim_mf6_model,
            active_keys=["riv-1"],
            passive_keys=["riv-1"],
            expected_type=River,
        )

    with pytest.raises(ValueError, match="keys with expected type"):
        RibaMod.validate_keys(
            coupled_ribasim_mf6_model,
            active_keys=["riv-1"],
            passive_keys=[],
            expected_type=Drainage,
        )


def test_derive_river_drainage_coupling():
    x = [0.5, 1.5, 2.5]
    y = [19.5, 18.5, 17.5, 16.5]
    data = np.array(
        [
            [0, 0, 0],
            [2, 2, 2],
            [5, 5, 5],
            [11, 11, 11],
        ]
    )
    gridded_basin = xr.DataArray(data=data, coords={"y": y, "x": x}, dims=("y", "x"))
    # nothing linked to 1 and 7
    basin_ids = np.array([0, 1, 2, 5, 7, 11])
    conductance = xr.DataArray(
        data=np.full((2, 4, 3), np.nan),
        coords={"layer": [1, 2], "y": y, "x": x},
        dims=("layer", "y", "x"),
    )
    conductance[:, :, 1] = 1.0

    actual = RibaMod.derive_river_drainage_coupling(
        gridded_basin, basin_ids, conductance
    )
    assert np.array_equal(actual["basin_index"], [0, 2, 3, 5, 0, 2, 3, 5])
    assert np.array_equal(actual["bound_index"], [0, 1, 2, 3, 4, 5, 6, 7])


def test_ribamod_write(
    ribasim_model, coupled_ribasim_mf6_model, basin_definition, tmp_path
):
    mf6_modelname, mf6_model = get_mf6_gwf_modelnames(coupled_ribasim_mf6_model)[0]
    mf6_river_packages = get_mf6_river_packagenames(mf6_model)

    driver_coupling = DriverCoupling(
        mf6_model=mf6_modelname,
        mf6_active_river_packages=mf6_river_packages,
    )

    coupled_models = RibaMod(
        ribasim_model,
        coupled_ribasim_mf6_model,
        coupling_list=[driver_coupling],
        basin_definition=basin_definition,
    )
    output_dir = tmp_path / "ribamod"
    coupling_dict = coupled_models.write_exchanges(output_dir)

    exchange_path = Path("exchanges") / "riv-1.tsv"
    expected_dict = {
        "mf6_model": "GWF_1",
        "mf6_active_river_packages": {"riv-1": exchange_path.as_posix()},
        "mf6_passive_river_packages": {},
        "mf6_active_drainage_packages": {},
        "mf6_passive_drainage_packages": {},
    }
    assert coupling_dict == expected_dict

    assert (output_dir / exchange_path).exists()
    exchange_df = pd.read_csv(output_dir / exchange_path, sep="\t")
    expected_df = pd.DataFrame(data={"basin_index": [0], "bound_index": [0]})
    assert exchange_df.equals(expected_df)


def test_ribamod_write_toml(
    ribasim_model, coupled_ribasim_mf6_model, basin_definition, tmp_path
):
    mf6_modelname, mf6_model = get_mf6_gwf_modelnames(coupled_ribasim_mf6_model)[0]
    mf6_river_packages = get_mf6_river_packagenames(mf6_model)

    driver_coupling = DriverCoupling(
        mf6_model=mf6_modelname,
        mf6_active_river_packages=mf6_river_packages,
    )

    coupled_models = RibaMod(
        ribasim_model,
        coupled_ribasim_mf6_model,
        coupling_list=[driver_coupling],
        basin_definition=basin_definition,
    )

    output_dir = tmp_path / "ribamod"
    coupling_dict = coupled_models.write_exchanges(output_dir)

    coupled_models.write_toml(
        output_dir, coupling_dict, "./modflow6.dll", "./ribasim.dll", "./ribasim-bin"
    )

    with open(output_dir / "imod_coupler.toml", mode="rb") as f:
        toml_dict = tomllib.load(f)

    # This contains empty tupled, which are removed in the TOML
    dict_coupling_expected = {k: v for k, v in coupling_dict.items()}
    dict_expected = {
        "timing": False,
        "log_level": "INFO",
        "driver_type": "ribamod",
        "driver": {
            "kernels": {
                "modflow6": {
                    "dll": "./modflow6.dll",
                    "work_dir": f".\\{coupled_models._modflow6_model_dir}",
                },
                "ribasim": {
                    "dll": "./ribasim.dll",
                    "dll_dep_dir": "./ribasim-bin",
                    "config_file": "ribasim\\trivial.toml",
                },
            },
            "coupling": [dict_coupling_expected],
        },
    }
    assert toml_dict == dict_expected


def get_mf6_gwf_modelnames(
    mf6_simulation: Modflow6Simulation,
) -> list[tuple[str, GroundwaterFlowModel]]:
    """
    Get names of gwf models in mf6 simulation
    """
    return [
        (key, value)
        for key, value in mf6_simulation.items()
        if isinstance(value, GroundwaterFlowModel)
    ]


def get_mf6_river_packagenames(mf6_model: GroundwaterFlowModel) -> list[str]:
    """
    Get names of river packages in mf6 simulation
    """
    return [key for key, value in mf6_model.items() if isinstance(value, River)]
