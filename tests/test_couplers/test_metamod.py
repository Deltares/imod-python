from imod.couplers.metamod import (
    MetaMod,
    NodeSvatMapping,
    RechargeSvatMapping,
    WellSvatMapping,
)

import pytest
import numpy as np
from numpy.testing import assert_equal


@pytest.mark.usefixtures("msw_model", "coupled_mf6_model")
def test_metamod_write(msw_model, coupled_mf6_model, tmp_path):
    output_dir = tmp_path / "metamod"

    coupled_models = MetaMod(msw_model, coupled_mf6_model)

    coupled_models.write(output_dir)

    # Test metaswap files written
    assert len(list(output_dir.rglob(r"*.inp"))) == 16
    assert len(list(output_dir.rglob(r"*.asc"))) == 4
    # Test exchanges written
    assert len(list(output_dir.rglob(r"*.dxc"))) == 3
    # Test mf6 files written
    assert len(list(output_dir.rglob(r"*.bin"))) == 6
    assert len(list(output_dir.rglob(r"*.nam"))) == 2
    assert len(list(output_dir.rglob(r"*.chd"))) == 1
    assert len(list(output_dir.rglob(r"*.rch"))) == 1
    assert len(list(output_dir.rglob(r"*.wel"))) == 1


@pytest.mark.usefixtures("msw_model", "coupled_mf6_model")
def test_metamod_write_exhange(
    msw_model, coupled_mf6_model, fixed_format_parser, tmp_path
):
    output_dir = tmp_path / "exhanges"
    output_dir.mkdir(exist_ok=True, parents=True)

    coupled_models = MetaMod(msw_model, coupled_mf6_model)

    coupled_models.write_exchanges(output_dir)

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
