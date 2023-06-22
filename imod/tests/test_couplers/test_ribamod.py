from dataclasses import asdict

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


def test_ribamod_write_toml(ribasim_model, coupled_ribasim_mf6_model, tmp_path):
    mf6_modelname, mf6_model = get_mf6_gwf_modelnames(coupled_ribasim_mf6_model)[0]
    mf6_river_packages = get_mf6_river_packagenames(mf6_model)

    driver_coupling = DriverCoupling(
        mf6_model=mf6_modelname,
        mf6_river_packages=mf6_river_packages,
        mf6_drainage_packages=[],
    )

    coupled_models = RibaMod(
        ribasim_model, coupled_ribasim_mf6_model, coupling_list=[driver_coupling]
    )

    output_dir = tmp_path / "ribamod"

    coupled_models.write_toml(
        output_dir, "./modflow6.dll", "./ribasim.dll", "./ribasim-bin"
    )

    with open(output_dir / "imod_coupler.toml", mode="rb") as f:
        toml_dict = tomllib.load(f)

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
                    "config_file": str(output_dir
                    / coupled_models._ribasim_model_dir
                    / "trivial.toml"),
                },
            },
            "coupling": [asdict(driver_coupling)],
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
