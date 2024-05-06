import pytest

import imod


@pytest.fixture(scope="module")
def imod5_dataset():
    tmp_path = imod.util.temporary_directory()
    return imod.data.imod5_projectfile_data(tmp_path)
