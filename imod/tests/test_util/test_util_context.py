import pytest

from imod.util.context import print_if_error


def test_print_if_error():
    with print_if_error(TypeError):
        1 + "a"

    with pytest.raises(TypeError):
        with print_if_error(ValueError):
            1 + "a"
