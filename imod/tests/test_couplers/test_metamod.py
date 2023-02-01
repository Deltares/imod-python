import numpy as np
import pytest
from numpy.testing import assert_equal

from imod.couplers.metamod import (
    MetaMod,
    NodeSvatMapping,
    RechargeSvatMapping,
    WellSvatMapping,
)

# tomllib part of Python 3.11, else use tomli
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_metamod_write(msw_model, coupled_mf6_model, tmp_path):
    output_dir = tmp_path / "metamod"

    coupled_models = MetaMod(
        msw_model,
        coupled_mf6_model,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey="wells_msw",
    )

    coupled_models.write(output_dir, "./modflow6.dll", "./metaswap.dll", "./metaswap")

    # Test metaswap files written
    assert len(list(output_dir.rglob(r"*.inp"))) == 16
    assert len(list(output_dir.rglob(r"*.asc"))) == 4
    # Test exchanges written
    assert len(list(output_dir.rglob(r"*.dxc"))) == 3
    assert len(list(output_dir.rglob(r"*.toml"))) == 1
    # Test mf6 files written
    assert len(list(output_dir.rglob(r"*.bin"))) == 6
    assert len(list(output_dir.rglob(r"*.nam"))) == 2
    assert len(list(output_dir.rglob(r"*.chd"))) == 1
    assert len(list(output_dir.rglob(r"*.rch"))) == 1
    assert len(list(output_dir.rglob(r"*.wel"))) == 1


def test_metamod_write_exhange(
    msw_model, coupled_mf6_model, fixed_format_parser, tmp_path
):
    output_dir = tmp_path / "exhanges"
    output_dir.mkdir(exist_ok=True, parents=True)

    coupled_models = MetaMod(
        msw_model,
        coupled_mf6_model,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey="wells_msw",
    )

    coupled_models.write_exchanges(
        output_dir,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey="wells_msw",
    )

    nodes_dxc = fixed_format_parser(
        output_dir / NodeSvatMapping._file_name,
        NodeSvatMapping._metadata_dict,
    )

    rch_dxc = fixed_format_parser(
        output_dir / RechargeSvatMapping._file_name,
        RechargeSvatMapping._metadata_dict,
    )

    wel_dxc = fixed_format_parser(
        output_dir / WellSvatMapping._file_name,
        WellSvatMapping._metadata_dict,
    )

    assert_equal(nodes_dxc["mod_id"], np.array([2, 8, 2, 5]))
    assert_equal(nodes_dxc["svat"], np.array([1, 2, 3, 4]))
    assert_equal(nodes_dxc["layer"], np.array([1, 1, 1, 1]))

    assert_equal(rch_dxc["rch_id"], np.array([1, 3, 1, 2]))
    assert_equal(rch_dxc["svat"], np.array([1, 2, 3, 4]))
    assert_equal(rch_dxc["layer"], np.array([1, 1, 1, 1]))

    assert_equal(wel_dxc["wel_id"], np.array([2, 8, 2, 5]))
    assert_equal(wel_dxc["svat"], np.array([1, 2, 3, 4]))
    assert_equal(wel_dxc["layer"], np.array([3, 3, 3, 3]))


def test_metamod_write_exhange_no_sprinkling(
    msw_model, coupled_mf6_model, fixed_format_parser, tmp_path
):
    # Remove sprinkling package
    msw_model.pop("sprinkling")

    output_dir = tmp_path / "exhanges"
    output_dir.mkdir(exist_ok=True, parents=True)

    coupled_models = MetaMod(
        msw_model,
        coupled_mf6_model,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey=None,
    )

    coupled_models.write_exchanges(
        output_dir,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey=None,
    )

    nodes_dxc = fixed_format_parser(
        output_dir / NodeSvatMapping._file_name,
        NodeSvatMapping._metadata_dict,
    )

    rch_dxc = fixed_format_parser(
        output_dir / RechargeSvatMapping._file_name,
        RechargeSvatMapping._metadata_dict,
    )

    well_dxc_written = (output_dir / WellSvatMapping._file_name).exists()

    assert well_dxc_written is False

    assert_equal(nodes_dxc["mod_id"], np.array([2, 8, 2, 5]))
    assert_equal(nodes_dxc["svat"], np.array([1, 2, 3, 4]))
    assert_equal(nodes_dxc["layer"], np.array([1, 1, 1, 1]))

    assert_equal(rch_dxc["rch_id"], np.array([1, 3, 1, 2]))
    assert_equal(rch_dxc["svat"], np.array([1, 2, 3, 4]))
    assert_equal(rch_dxc["layer"], np.array([1, 1, 1, 1]))


def test_metamod_write_toml(msw_model, coupled_mf6_model, tmp_path):
    output_dir = tmp_path
    output_dir.mkdir(exist_ok=True, parents=True)

    coupled_models = MetaMod(
        msw_model,
        coupled_mf6_model,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey="wells_msw",
    )

    coupling_dict = {
        "mf6_model": "GWF_1",
        "mf6_msw_node_map": "./exchanges/nodenr2svat.dxc",
        "mf6_msw_recharge_map": "./exchanges/rchindex2svat.dxc",
        "mf6_msw_recharge_pkg": "rch_msw",
        "enable_sprinkling": True,
        "mf6_msw_well_pkg": "wells_msw",
        "mf6_msw_sprinkling_map": "./exchanges/wellindex2svat.dxc",
    }

    coupled_models.write_toml(
        output_dir, "./modflow6.dll", "./metaswap.dll", "./metaswap", coupling_dict
    )

    with open(output_dir / "imod_coupler.toml", mode="rb") as f:
        toml_dict = tomllib.load(f)

    dict_expected = {
        "timing": False,
        "log_level": "INFO",
        "driver_type": "metamod",
        "driver": {
            "kernels": {
                "modflow6": {
                    "dll": "./modflow6.dll",
                    "work_dir": f".\\{coupled_models._modflow6_model_dir}",
                },
                "metaswap": {
                    "dll": "./metaswap.dll",
                    "work_dir": f".\\{coupled_models._metaswap_model_dir}",
                    "dll_dep_dir": "./metaswap",
                },
            },
            "coupling": [coupling_dict],
        },
    }

    assert toml_dict == dict_expected


def test_metamod_get_coupling_dict(msw_model, coupled_mf6_model, tmp_path):
    output_dir = tmp_path / "exchanges"

    coupled_models = MetaMod(
        msw_model,
        coupled_mf6_model,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey="wells_msw",
    )

    dict_expected = {
        "mf6_model": "GWF_1",
        "mf6_msw_node_map": "./exchanges/nodenr2svat.dxc",
        "mf6_msw_recharge_map": "./exchanges/rchindex2svat.dxc",
        "mf6_msw_recharge_pkg": "rch_msw",
        "enable_sprinkling": True,
        "mf6_msw_well_pkg": "wells_msw",
        "mf6_msw_sprinkling_map": "./exchanges/wellindex2svat.dxc",
    }

    coupled_dict = coupled_models._get_coupling_dict(
        output_dir,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey="wells_msw",
    )

    assert dict_expected == coupled_dict


def test_metamod_get_coupling_dict_no_sprinkling(
    msw_model, coupled_mf6_model, tmp_path
):
    output_dir = tmp_path / "exchanges"

    # Remove sprinkling package
    msw_model.pop("sprinkling")

    coupled_models = MetaMod(
        msw_model,
        coupled_mf6_model,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey=None,
    )

    dict_expected = {
        "mf6_model": "GWF_1",
        "mf6_msw_node_map": "./exchanges/nodenr2svat.dxc",
        "mf6_msw_recharge_map": "./exchanges/rchindex2svat.dxc",
        "mf6_msw_recharge_pkg": "rch_msw",
        "enable_sprinkling": False,
    }

    coupled_dict = coupled_models._get_coupling_dict(
        output_dir,
        mf6_rch_pkgkey="rch_msw",
        mf6_wel_pkgkey=None,
    )

    assert dict_expected == coupled_dict


def test_metamod_init_no_sprinkling_fail(msw_model, coupled_mf6_model):
    # Remove sprinkling package
    msw_model.pop("sprinkling")

    with pytest.raises(ValueError):
        MetaMod(
            msw_model,
            coupled_mf6_model,
            mf6_rch_pkgkey="rch_msw",
            mf6_wel_pkgkey="wells_msw",
        )


def test_metamod_init_no_mf6_well_fail(msw_model, coupled_mf6_model):
    with pytest.raises(ValueError):
        MetaMod(
            msw_model,
            coupled_mf6_model,
            mf6_rch_pkgkey="rch_msw",
            mf6_wel_pkgkey="does_not_exist",
        )


def test_metamod_init_no_mf6_well_fail2(msw_model, coupled_mf6_model):
    with pytest.raises(ValueError):
        MetaMod(
            msw_model,
            coupled_mf6_model,
            mf6_rch_pkgkey="rch_msw",
            mf6_wel_pkgkey=None,
        )


def test_metamod_init_no_mf6_rch_fail(msw_model, coupled_mf6_model):
    with pytest.raises(ValueError):
        MetaMod(
            msw_model,
            coupled_mf6_model,
            mf6_rch_pkgkey="does_not_exist",
            mf6_wel_pkgkey="wells_msw",
        )
