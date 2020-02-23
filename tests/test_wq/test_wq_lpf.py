import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.wq import LayerPropertyFlow


@pytest.fixture(scope="module")
def layerpropertyflow():
    def create_lpf(bed=False):
        layer = np.arange(1, 4)
        y = np.arange(4.5, 0.0, -1.0)
        x = np.arange(0.5, 5.0, 1.0)
        k_horizontal = xr.DataArray(
            np.full((3, 5, 5), 1.0),
            coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
            dims=("layer", "y", "x"),
        )

        lpf = LayerPropertyFlow(
            k_horizontal=k_horizontal,
            k_vertical=k_horizontal.copy(),
            horizontal_anisotropy=k_horizontal.copy(),
            interblock=k_horizontal.copy(),
            layer_type=k_horizontal.copy(),
            specific_storage=k_horizontal.copy(),
            specific_yield=k_horizontal.copy(),
            save_budget=False,
            layer_wet=k_horizontal.copy(),
            interval_wet=0.01,
            method_wet="wetfactor",
            head_dry=1.0e20,
        )
        return lpf

    return create_lpf


def test_render(layerpropertyflow):
    create_lpf = layerpropertyflow
    lpf = create_lpf(bed=False)
    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [lpf]
            ilpfcb = 0
            hdry = 1e+20
            layvka_l? = 0
            laytyp_l1 = layer_type_l1.idf
            laytyp_l2 = layer_type_l2.idf
            laytyp_l3 = layer_type_l3.idf
            layavg_l1 = interblock_l1.idf
            layavg_l2 = interblock_l2.idf
            layavg_l3 = interblock_l3.idf
            chani_l1 = horizontal_anisotropy_l1.idf
            chani_l2 = horizontal_anisotropy_l2.idf
            chani_l3 = horizontal_anisotropy_l3.idf
            hk_l1 = k_horizontal_l1.idf
            hk_l2 = k_horizontal_l2.idf
            hk_l3 = k_horizontal_l3.idf
            vka_l1 = k_vertical_l1.idf
            vka_l2 = k_vertical_l2.idf
            vka_l3 = k_vertical_l3.idf
            ss_l1 = specific_storage_l1.idf
            ss_l2 = specific_storage_l2.idf
            ss_l3 = specific_storage_l3.idf
            sy_l1 = specific_yield_l1.idf
            sy_l2 = specific_yield_l2.idf
            sy_l3 = specific_yield_l3.idf
            laywet_l1 = layer_wet_l1.idf
            laywet_l2 = layer_wet_l2.idf
            laywet_l3 = layer_wet_l3.idf"""
    )

    assert lpf._render(directory) == compare
