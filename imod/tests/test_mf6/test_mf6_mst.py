import pathlib
import textwrap

import numpy as np
import pytest

from imod.mf6.mst import MobileStorageTransfer


def test_render_simple():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    m = MobileStorageTransfer(0.3)
    actual = m.render(directory, "mst", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin griddata
          porosity
            constant 0.3
        end griddata"""
    )
    assert actual == expected


@pytest.mark.usefixtures(
    "porosity_fc",
    "decay_fc",
    "decay_sorbed_fc",
    "bulk_density_fc",
    "distcoef_fc",
    "sp2_fc",
)
def test_render_first_order_decay(
    porosity_fc, decay_fc, decay_sorbed_fc, bulk_density_fc, distcoef_fc, sp2_fc
):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    m = MobileStorageTransfer(
        porosity_fc,
        decay=decay_fc,
        decay_sorbed=decay_sorbed_fc,
        bulk_density=bulk_density_fc,
        distcoef=distcoef_fc,
        sp2=sp2_fc,
        first_order_decay=True,
        sorption="langmuir",
    )
    actual = m.render(directory, "mst", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          first_order_decay
          sorption langmuir
        end options

        begin griddata
          porosity
            open/close mymodel/mst/porosity.bin (binary)
          decay
            open/close mymodel/mst/decay.bin (binary)
          decay_sorbed
            open/close mymodel/mst/decay_sorbed.bin (binary)
          bulk_density
            open/close mymodel/mst/bulk_density.bin (binary)
          distcoef
            open/close mymodel/mst/distcoef.bin (binary)
          sp2
            open/close mymodel/mst/sp2.bin (binary)
        end griddata"""
    )

    assert actual == expected


def test_render_zero_order_decay(
    porosity_fc, decay_fc, decay_sorbed_fc, bulk_density_fc, distcoef_fc, sp2_fc
):
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    m = MobileStorageTransfer(
        porosity_fc,
        decay=decay_fc,
        decay_sorbed=decay_sorbed_fc,
        bulk_density=bulk_density_fc,
        distcoef=distcoef_fc,
        sp2=sp2_fc,
        sorption="langmuir",
        zero_order_decay=True,
    )
    actual = m.render(directory, "mst", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          zero_order_decay
          sorption langmuir
        end options

        begin griddata
          porosity
            open/close mymodel/mst/porosity.bin (binary)
          decay
            open/close mymodel/mst/decay.bin (binary)
          decay_sorbed
            open/close mymodel/mst/decay_sorbed.bin (binary)
          bulk_density
            open/close mymodel/mst/bulk_density.bin (binary)
          distcoef
            open/close mymodel/mst/distcoef.bin (binary)
          sp2
            open/close mymodel/mst/sp2.bin (binary)
        end griddata"""
    )

    assert actual == expected


def test_wrong_decay_order(
    porosity_fc, decay_fc, decay_sorbed_fc, bulk_density_fc, distcoef_fc, sp2_fc
):
    with pytest.raises(ValueError):
        MobileStorageTransfer(
            porosity_fc,
            decay=decay_fc,
            decay_sorbed=decay_sorbed_fc,
            bulk_density=bulk_density_fc,
            distcoef=distcoef_fc,
            sp2=sp2_fc,
            sorption="langmuir",
            zero_order_decay=True,
            first_order_decay=True,
        )
