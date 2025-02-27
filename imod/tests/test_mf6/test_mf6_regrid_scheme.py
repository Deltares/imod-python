from dataclasses import asdict

import pytest
from pydantic import ValidationError
from pytest_cases import parametrize, parametrize_with_cases

from imod.common.utilities.regrid import RegridderType
from imod.mf6.regrid.regrid_schemes import (
    ConstantHeadRegridMethod,
    DiscretizationRegridMethod,
    DispersionRegridMethod,
    DrainageRegridMethod,
    EvapotranspirationRegridMethod,
    GeneralHeadBoundaryRegridMethod,
    InitialConditionsRegridMethod,
    MobileStorageTransferRegridMethod,
    NodePropertyFlowRegridMethod,
    RechargeRegridMethod,
    RiverRegridMethod,
    SpecificStorageRegridMethod,
    StorageCoefficientRegridMethod,
)

ALL_REGRID_METHODS = [
    ConstantHeadRegridMethod,
    DiscretizationRegridMethod,
    DispersionRegridMethod,
    DrainageRegridMethod,
    EvapotranspirationRegridMethod,
    GeneralHeadBoundaryRegridMethod,
    InitialConditionsRegridMethod,
    MobileStorageTransferRegridMethod,
    NodePropertyFlowRegridMethod,
    RechargeRegridMethod,
    RiverRegridMethod,
    SpecificStorageRegridMethod,
    StorageCoefficientRegridMethod,
]


# Forward regrid methods to case functions to generate readable test ids
@parametrize(regrid_method=ALL_REGRID_METHODS)
def case_regrid(regrid_method):
    return regrid_method


def tuple_centroid():
    return (RegridderType.CENTROIDLOCATOR, "max")


def tuple_barycentric():
    return (RegridderType.BARYCENTRIC,)


@parametrize_with_cases("regrid_method", cases=".")
def test_regrid_method(regrid_method):
    regrid_method_instance = regrid_method()
    assert isinstance(regrid_method_instance, regrid_method)


@parametrize_with_cases("regrid_method", cases=".")
@parametrize_with_cases("tuple_values", cases=".", prefix="tuple_")
def test_regrid_method_custom(regrid_method, tuple_values):
    for key in asdict(regrid_method()).keys():
        kwargs = {key: tuple_values}
        regrid_method_instance = regrid_method(**kwargs)
        assert isinstance(regrid_method_instance, regrid_method)


@parametrize_with_cases("regrid_method", cases=".")
def test_regrid_method_incorrect_input(regrid_method):
    with pytest.raises(ValidationError):
        regrid_method(non_existent_var=999)

    for key in asdict(regrid_method()).keys():
        kwargs = {key: "wrong_value_type"}
        with pytest.raises(ValidationError):
            regrid_method(**kwargs)
