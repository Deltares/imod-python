from contextlib import nullcontext as does_not_raise

import pytest

from imod.mf6.exchangebase import ExchangeBase, _pkg_id_to_type


class DummyExchange(ExchangeBase):
    _pkg_id = "gwfgwt"

    def __init__(self, model_id1: str = None, model_id2: str = None):
        variables_to_merge = {}
        if model_id1:
            variables_to_merge["model_name_1"] = model_id1
        if model_id2:
            variables_to_merge["model_name_2"] = model_id2

        super().__init__(variables_to_merge)


def test_package_name_construct_name():
    # Arrange.
    model_name1 = "testmodel1"
    model_name2 = "testmodel2"
    exchange = DummyExchange(model_name1, model_name2)

    # Act.
    package_name = exchange.package_name()

    # Assert.
    assert model_name1 in package_name
    assert model_name2 in package_name


@pytest.mark.parametrize(
    ("model_name1", "model_name2", "expectation"),
    (
        [None, None, pytest.raises(ValueError)],
        ["testmodel1", None, pytest.raises(ValueError)],
        [None, "testmodel2", pytest.raises(ValueError)],
        ["testmodel1", "testmodel2", does_not_raise()],
    ),
)
def test_package_name_missing_name(model_name1, model_name2, expectation):
    # Arrange
    exchange = DummyExchange(model_name1, model_name2)

    # Act/Assert
    with expectation:
        exchange.package_name()


def test_get_specification():
    # Arrange.
    model_name1 = "testmodel1"
    model_name2 = "testmodel2"
    exchange = DummyExchange(model_name1, model_name2)

    # Act.
    (
        spec_exchange_type,
        spec_filename,
        spec_model_name1,
        spec_model_name2,
    ) = exchange.get_specification()

    # Assert
    assert spec_exchange_type is _pkg_id_to_type[DummyExchange._pkg_id]
    assert model_name1 in spec_filename
    assert model_name2 in spec_filename
    assert DummyExchange._pkg_id in spec_filename
    assert spec_model_name1 == model_name1
    assert spec_model_name2 == model_name2
