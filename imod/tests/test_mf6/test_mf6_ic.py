import pathlib
import textwrap
from copy import deepcopy

import pytest

import imod
from imod.schemata import ValidationError


def test_render():
    ic_start = imod.mf6.InitialConditions(start=0.0)
    directory = pathlib.Path("mymodel")
    actual_start = ic_start._render(directory, "ic", None, True)
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
    assert actual_start == expected


def test_wrong_dtype():
    with pytest.raises(ValidationError):
        imod.mf6.InitialConditions(start=0)


def test_validate_false():
    imod.mf6.InitialConditions(start=0, validate=False)


def test_from_imod5(imod5_dataset, tmp_path):
    data = deepcopy(imod5_dataset[0])

    target_grid = data["khv"]["kh"]

    ic = imod.mf6.InitialConditions.from_imod5_data(data, target_grid)

    ic._validate_init_schemata(True)

    rendered_ic = ic._render(tmp_path, "ic", None, False)
    assert "strt" in rendered_ic
