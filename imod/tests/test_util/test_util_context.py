import pytest

from imod.util.context import print_if_error

# Comment added to test if sonarcloud ignores this file.

def test_print_if_error():
    with print_if_error(TypeError):
        1 + "a"

    with pytest.raises(TypeError):
        with print_if_error(ValueError):
            1 + "a"
