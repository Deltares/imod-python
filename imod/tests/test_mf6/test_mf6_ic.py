from copy import deepcopy
import pathlib
import textwrap

import pytest

import imod
from imod.schemata import ValidationError
from imod.tests.test_mf6.test_mf6_dis import _load_imod5_data_in_memory


def test_render():
    ic_head = imod.mf6.InitialConditions(start=0.0)
    ic_start = imod.mf6.InitialConditions(start=0.0)
    directory = pathlib.Path("mymodel")
    actual_head = ic_head.render(directory, "ic", None, True)
    actual_start = ic_start.render(directory, "ic", None, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin griddata
          strt
            constant 0.0
        end griddata
        """
    )
    assert actual_head == expected
    assert actual_start == expected


def test_wrong_dtype():
    with pytest.raises(ValidationError):
        imod.mf6.InitialConditions(start=0)


def test_validate_false():
    imod.mf6.InitialConditions(start=0, validate=False)


def test_wrong_arguments():
    with pytest.raises(ValueError):
        imod.mf6.InitialConditions(head=0.0, start=1.0)

@pytest.mark.usefixtures("imod5_dataset")
def test_from_imod5( imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])

    _load_imod5_data_in_memory(data)
    target_grid = data["khv"]["kh"]

    ic = imod.mf6.InitialConditions.from_imod5_data(data, target_grid)

    ic._validate_init_schemata(True)


    rendered_ic = ic.render(tmp_path, "ic", None, False)
    assert "strt" in rendered_ic