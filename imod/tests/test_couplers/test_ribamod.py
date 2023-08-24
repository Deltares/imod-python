import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

import imod
from imod.couplers.ribamod import RibaMod
from imod.couplers.ribamod.ribamod import DriverCoupling
from imod.mf6.model import GroundwaterFlowModel
from imod.mf6.riv import River
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
    return gpd.GeoDataFrame(data={"basin_id": node_id}, geometry=[polygon])


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

    exchange_path = output_dir / "exchanges" / "riv-1.tsv"
    expected_dict = {"mf6_active_river_packages": {"riv-1": exchange_path.as_posix()}}
    assert coupling_dict == expected_dict

    assert exchange_path.exists()
    exchange_df = pd.read_csv(exchange_path, sep="\t")
    expected_df = pd.DataFrame(data={"basin_id": [1], "bound_id": [8]})
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
    dict_coupling_expected = {k: v for k, v in coupling_dict.items() if v}
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
                    "config_file": str(
                        output_dir / coupled_models._ribasim_model_dir / "trivial.toml"
                    ),
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
