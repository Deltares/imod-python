from imod.couplers.ribamod import RibaMod

# tomllib part of Python 3.11, else use tomli
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_ribamod_write(ribasim_model, coupled_ribasim_mf6_model, tmp_path):
    output_dir = tmp_path / "ribamod"

    coupled_models = RibaMod(
        ribasim_model,
        coupled_ribasim_mf6_model,
    )

    coupled_models.write(output_dir, "./modflow6.dll", "./ribasim.dll", "./ribasim-bin")


def test_ribamod_write_toml(ribasim_model, coupled_ribasim_mf6_model, tmp_path):
    output_dir = tmp_path / "ribamod"

    coupled_models = RibaMod(
        ribasim_model,
        coupled_ribasim_mf6_model,
    )
    coupling_dict = {}

    coupled_models.write_toml(
        output_dir, "./modflow6.dll", "./ribasim.dll", "./ribasim-bin", coupling_dict
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
                    "config_file": f"{coupled_models._ribasim_model_dir}/trivial.toml",
                },
            },
            "coupling": [coupling_dict],
        },
    }
    assert toml_dict == dict_expected
