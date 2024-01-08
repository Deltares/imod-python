import pytest

from imod.couplers import MetaMod


def test_metamod_init():
    # Functionality has been moved to primod, tests have been moved to
    # https://github.com/Deltares/imod_coupler/tree/main/tests/test_primod

    with pytest.raises(NotImplementedError):
        MetaMod()
