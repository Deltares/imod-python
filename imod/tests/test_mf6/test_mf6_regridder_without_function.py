
import pytest
import imod
from imod.mf6.utilities.regridding_types import RegridderType

@pytest.mark.usefixtures("circle_model")
def test_simulation_can_be_regridded_with_methods_without_functions(circle_model, tmp_path):
    simulation = circle_model
    idomain = circle_model["GWF_1"].domain
    #redefine the default regridding method for the constant head package
    old_regrid_method =  imod.mf6.ConstantHead._regrid_method
    imod.mf6.ConstantHead._regrid_method =  {
        "head": (RegridderType.BARYCENTRIC),
        "concentration": ( RegridderType.BARYCENTRIC),
    }
    regridding_succeeded = False
    try:
        simulation.regrid_like("regridded", idomain)
        regridding_succeeded = True
    finally:
        imod.mf6.ConstantHead._regrid_method = old_regrid_method

    assert regridding_succeeded