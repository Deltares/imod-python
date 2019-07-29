import pytest

import imod


def test_mf6dummy():
    assert imod.mf6.mf6dummy(3) == 5
