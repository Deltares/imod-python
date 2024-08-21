from datetime import datetime
from shutil import copyfile
from textwrap import dedent
from typing import Union

import numpy as np
import pytest
from pytest_cases import (
    get_all_cases,
    get_parametrize_args,
    parametrize,
    parametrize_with_cases,
)

from imod.formats.prj import open_projectfile_data
from imod.mf6 import LayeredWell, Well


class WellPrjCases:
    """Cases for projectfile well records"""

    def case_simple__steady_state(self):
        return dedent(
            """
            0001,(WEL),1
            steady-state
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_associated__steady_state(self):
        return dedent(
            """
            0001,(WEL),1
            steady-state
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_mixed__steady_state(self):
        return dedent(
            """
            0001,(WEL),1
            steady-state
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__first(self):
        return dedent(
            """
            0001,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__first_multi_layer1(self):
        return dedent(
            """
            0001,(WEL),1
            1982-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 002, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__first_multi_layer2(self):
        return dedent(
            """
            0001,(WEL),1
            1982-01-01
            001,002
            1,2, 000, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__all_same(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-02-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-03-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__all_same_multi_layer1(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 002, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-02-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 002, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-03-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 002, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__all_same_multi_layer2(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,002
            1,2, 000, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-02-01
            001,002
            1,2, 000, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-03-01
            001,002
            1,2, 000, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__all_different1(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-02-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple2.ipf"
            1982-03-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple3.ipf"
            """
        )

    def case_simple__all_different2(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-02-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple2.ipf"
            1982-03-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple3.ipf"
            """
        )

    def case_simple__all_different3(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-02-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple2.ipf"
            1982-03-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple3.ipf"
            """
        )

    def case_associated__first(self):
        return dedent(
            """
            0001,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_associated__all(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1982-02-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1982-03-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_associated__all_varying_factors(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1982-02-01
            001,001
            1,2, 001, 0.5, 0.0, -999.9900 ,"ipf/associated.ipf"
            1982-03-01
            001,001
            1,2, 001, 0.2, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_associated__multiple_layers_different_factors(self):
        return dedent(
            """
            0001,(WEL),1
            1982-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1,2, 002, 0.75, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_mixed__first(self):
        return dedent(
            """
            0001,(WEL),1
            1982-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_mixed__all(self):
        return dedent(
            """
            0003,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1982-02-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-03-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_mixed__associated_second(self):
        return dedent(
            """
            0002,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1982-02-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )


class WellReadCases:
    """Expected cases as interpreted by ``imod.formats.prj.open_projectfile_data``"""

    def case_simple__steady_state(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [None],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_associated__steady_state(self):
        return {
            "wel-associated": {
                "has_associated": True,
                "time": [None],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_mixed__steady_state(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [None],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-associated": {
                "has_associated": True,
                "time": [None],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_simple__first(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_simple__first_multi_layer1(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 1, 1), datetime(1982, 1, 1)],
                "layer": [1, 2],
                "factor": [1.0, 1.0],
                "addition": [0.0, 0.0],
            },
        }

    def case_simple__first_multi_layer2(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 1, 1), datetime(1982, 1, 1)],
                "layer": [0, 1],
                "factor": [1.0, 1.0],
                "addition": [0.0, 0.0],
            },
        }

    def case_simple__all_same(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [
                    datetime(1982, 1, 1),
                    datetime(1982, 2, 1),
                    datetime(1982, 3, 1),
                ],
                "layer": [1, 1, 1],
                "factor": [1.0, 1.0, 1.0],
                "addition": [0.0, 0.0, 0.0],
            },
        }

    def case_simple__all_same_multi_layer1(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [
                    datetime(1982, 1, 1),
                    datetime(1982, 1, 1),
                    datetime(1982, 2, 1),
                    datetime(1982, 2, 1),
                    datetime(1982, 3, 1),
                    datetime(1982, 3, 1),
                ],
                "layer": [1, 2, 1, 2, 1, 2],
                "factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "addition": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        }

    def case_simple__all_same_multi_layer2(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [
                    datetime(1982, 1, 1),
                    datetime(1982, 1, 1),
                    datetime(1982, 2, 1),
                    datetime(1982, 2, 1),
                    datetime(1982, 3, 1),
                    datetime(1982, 3, 1),
                ],
                "layer": [0, 1, 0, 1, 0, 1],
                "factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "addition": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        }

    def case_simple__all_different1(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-simple2": {
                "has_associated": False,
                "time": [datetime(1982, 2, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-simple3": {
                "has_associated": False,
                "time": [datetime(1982, 3, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_simple__all_different2(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 1, 1), datetime(1982, 2, 1)],
                "layer": [1, 1],
                "factor": [1.0, 1.0],
                "addition": [0.0, 0.0],
            },
            "wel-simple2": {
                "has_associated": False,
                "time": [datetime(1982, 2, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-simple3": {
                "has_associated": False,
                "time": [datetime(1982, 3, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_simple__all_different3(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 1, 1), datetime(1982, 3, 1)],
                "layer": [1, 1],
                "factor": [1.0, 1.0],
                "addition": [0.0, 0.0],
            },
            "wel-simple2": {
                "has_associated": False,
                "time": [datetime(1982, 2, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-simple3": {
                "has_associated": False,
                "time": [datetime(1982, 3, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_associated__first(self):
        return {
            "wel-associated": {
                "has_associated": True,
                "time": [datetime(1982, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            }
        }

    def case_associated__all(self):
        return {
            "wel-associated": {
                "has_associated": True,
                "time": [
                    datetime(1982, 1, 1),
                    datetime(1982, 2, 1),
                    datetime(1982, 3, 1),
                ],
                "layer": [1, 1, 1],
                "factor": [1.0, 1.0, 1.0],
                "addition": [0.0, 0.0, 0.0],
            },
        }

    def case_associated__all_varying_factors(self):
        return {
            "wel-associated": {
                "has_associated": True,
                "time": [
                    datetime(1982, 1, 1),
                    datetime(1982, 2, 1),
                    datetime(1982, 3, 1),
                ],
                "layer": [1, 1, 1],
                "factor": [1.0, 0.5, 0.2],
                "addition": [0.0, 0.0, 0.0],
            },
        }

    def case_associated__multiple_layers_different_factors(self):
        return {
            "wel-associated": {
                "has_associated": True,
                "time": [
                    datetime(1982, 1, 1),
                    datetime(1982, 1, 1),
                ],
                "layer": [1, 2],
                "factor": [1.0, 0.75],
                "addition": [0.0, 0.0],
            },
        }

    def case_mixed__first(self):
        return {
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-associated": {
                "has_associated": True,
                "time": [datetime(1982, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_mixed__all(self):
        return {
            "wel-associated": {
                "has_associated": True,
                "time": [
                    datetime(1982, 1, 1),
                    datetime(1982, 2, 1),
                    datetime(1982, 3, 1),
                ],
                "layer": [1, 1, 1],
                "factor": [1.0, 1.0, 1.0],
                "addition": [0.0, 0.0, 0.0],
            },
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 2, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

    def case_mixed__associated_second(self):
        return {
            "wel-associated": {
                "has_associated": True,
                "time": [datetime(1982, 2, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-simple1": {
                "has_associated": False,
                "time": [datetime(1982, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }


class WellPackageCases:
    """
    Expected cases as loaded with from_imod5_data.
    Returns a tuple  with as first element a bool whether the import is expected to fail
    The second element specifies in which timesteps the rates are set to zero.

    Returns
    -------
    {wellname: (fails, has_time, datetimes_set_to_zero)}
    """

    def case_simple__steady_state(self):
        return {
            "wel-simple1": (False, False, []),
        }

    def case_associated__steady_state(self):
        return {
            "wel-associated": (False, True, []),
        }

    def case_mixed__steady_state(self):
        return {
            "wel-associated": (False, True, []),
            "wel-simple1": (False, False, []),
        }

    def case_simple__first(self):
        return {
            "wel-simple1": (False, True, [datetime(1982, 2, 1), datetime(1982, 3, 1)]),
        }

    def case_simple__first_multi_layer1(self):
        return {
            "wel-simple1": (False, True, [datetime(1982, 2, 1), datetime(1982, 3, 1)]),
        }

    def case_simple__first_multi_layer2(self):
        return {
            "wel-simple1": (True, False, []),
        }

    def case_simple__all_same(self):
        return {
            "wel-simple1": (False, True, []),
        }

    def case_simple__all_same_multi_layer1(self):
        return {
            "wel-simple1": (False, True, []),
        }

    def case_simple__all_same_multi_layer2(self):
        return {
            "wel-simple1": (True, False, []),
        }

    def case_simple__all_different1(self):
        return {
            "wel-simple1": (False, True, [datetime(1982, 2, 1), datetime(1982, 3, 1)]),
            "wel-simple2": (False, True, [datetime(1982, 3, 1)]),
            "wel-simple3": (False, True, []),
        }

    def case_simple__all_different2(self):
        return {
            "wel-simple1": (False, True, [datetime(1982, 3, 1)]),
            "wel-simple2": (False, True, [datetime(1982, 3, 1)]),
            "wel-simple3": (False, True, []),
        }

    def case_simple__all_different3(self):
        return {
            "wel-simple1": (False, True, [datetime(1982, 2, 1)]),
            "wel-simple2": (False, True, [datetime(1982, 3, 1)]),
            "wel-simple3": (False, True, []),
        }

    def case_associated__first(self):
        return {"wel-associated": (False, True, [])}

    def case_associated__all(self):
        return {"wel-associated": (False, True, [])}

    def case_associated__all_varying_factors(self):
        return {"wel-associated": (True, False, [])}

    def case_associated__multiple_layers_different_factors(self):
        return {"wel-associated": (True, False, [])}

    def case_mixed__first(self):
        return {
            "wel-simple1": (False, True, [datetime(1982, 2, 1), datetime(1982, 3, 1)]),
            "wel-associated": (False, True, []),
        }

    def case_mixed__all(self):
        return {
            "wel-simple1": (False, True, [datetime(1982, 3, 1)]),
            "wel-associated": (False, True, []),
        }

    def case_mixed__associated_second(self):
        return {
            "wel-simple1": (False, True, [datetime(1982, 2, 1), datetime(1982, 3, 1)]),
            "wel-associated": (True, False, []),
        }


# pytest_cases doesn't support any "zipped test cases", instead it takes the
# outer product of cases, when providing multiple case sets.
# https://github.com/smarie/python-pytest-cases/issues/284
# To support this, we would like to retrieve all function arguments from the
# case classes and to zip them together, something like
# zip(input_args,expected).
def case_args_to_parametrize(cases, prefix):
    """Manually retrieve all case args of a set in cases."""

    # Decorate some dummy function to be able to call ``get_all_cases``. For some
    # reason, pytest_cases requires a decorated function (despite telling us
    # differently in the docs.)
    @parametrize_with_cases("case", cases=cases)
    def f(case):
        return case

    all_cases = get_all_cases(f, cases=cases)
    return get_parametrize_args(f, all_cases, prefix)


PRJ_ARGS = case_args_to_parametrize(WellPrjCases, "case_")
READ_ARGS = case_args_to_parametrize(WellReadCases, "case_")
PKG_ARGS = case_args_to_parametrize(WellPackageCases, "case_")


def setup_test_files(wel_case, wel_file, well_mixed_ipfs, tmp_path):
    """
    Write string to projectfile, and copy ipf files to directory.
    """
    with open(wel_file, "w") as f:
        f.write(wel_case)

    ipf_dir = tmp_path / "ipf"
    ipf_dir.mkdir(exist_ok=True)

    # copy files to test folder
    for p in well_mixed_ipfs:
        copyfile(p, ipf_dir / p.name)


def get_case_name(request):
    id_name = request.node.callspec.id
    # Verify right cases are matched. This can go wrong when case names are not
    # inserted in the right order in Case class.
    cases = id_name.split("-")
    # First entry refers to wel obj, we can skip this.
    assert cases[1] == cases[-1]

    return cases[1]


@parametrize("wel_case, expected", argvalues=list(zip(PRJ_ARGS, READ_ARGS)))
def test_open_projectfile_data_wells(
    wel_case, expected, well_mixed_ipfs, tmp_path, request
):
    # Arrange
    case_name = get_case_name(request)
    wel_file = tmp_path / f"{case_name}.prj"
    setup_test_files(wel_case, wel_file, well_mixed_ipfs, tmp_path)

    # Act
    data, _ = open_projectfile_data(wel_file)
    assert len(set(expected.keys()) ^ set(data.keys())) == 0
    fields = ["time", "layer", "addition", "factor", "has_associated"]
    for wel_name, wel_expected in expected.items():
        actual = data[wel_name]
        for field in fields:
            assert field in actual
            assert actual[field] == wel_expected[field]


@parametrize("wel_case, expected_dict", argvalues=list(zip(PRJ_ARGS, PKG_ARGS)))
@parametrize("wel_cls", argvalues=[LayeredWell, Well])
def test_from_imod5_data_wells(
    wel_cls: Union[LayeredWell, Well],
    wel_case,
    expected_dict,
    well_mixed_ipfs,
    tmp_path,
    request,
):
    # Arrange
    # Replace layer number to zero if non-layered well.
    if wel_cls == Well:
        wel_case = wel_case.replace("1,2, 001", "1,2, 000")
    # Write prj and copy ipfs to right folder.
    case_name = get_case_name(request)
    wel_file = tmp_path / f"{case_name}.prj"
    setup_test_files(wel_case, wel_file, well_mixed_ipfs, tmp_path)

    times = [datetime(1982, i + 1, 1) for i in range(4)]

    # Act
    data, _ = open_projectfile_data(wel_file)
    for wellname in data.keys():
        assert wellname in expected_dict.keys()
        fails, has_time, expected_set_to_zero = expected_dict[wellname]
        if fails:
            with pytest.raises(ValueError):
                wel_cls.from_imod5_data(wellname, data, times=times)
        else:
            well = wel_cls.from_imod5_data(wellname, data, times=times)
            rate = well.dataset["rate"]
            if has_time:
                actual_set_to_zero = [
                    t.values
                    for t in rate.coords["time"]
                    if (rate.sel(time=t) == 0.0).all()
                ]
                expected_set_to_zero = [
                    np.datetime64(t, "ns") for t in expected_set_to_zero
                ]
                diff = set(actual_set_to_zero) ^ set(expected_set_to_zero)
                assert len(diff) == 0
            else:
                assert "time" not in rate.dims
                assert "time" not in rate.coords


@parametrize("wel_case, expected_dict", argvalues=list(zip(PRJ_ARGS, PKG_ARGS)))
@parametrize("wel_cls", argvalues=[LayeredWell, Well])
def test_from_imod5_data_wells__outside_range(
    wel_cls: Union[LayeredWell, Well],
    wel_case,
    expected_dict,
    well_mixed_ipfs,
    tmp_path,
    request,
):
    """
    Test when values are retrieved outside time domain of wells, should be all
    set to zero for unassociated ipfs, and be forward filled with the last entry
    for associated ipfs.
    """
    # Arrange
    # Replace layer number to zero if non-layered well.
    if wel_cls == Well:
        wel_case = wel_case.replace("1,2, 001", "1,2, 000")
    # Write prj and copy ipfs to right folder.
    case_name = get_case_name(request)
    wel_file = tmp_path / f"{case_name}.prj"
    setup_test_files(wel_case, wel_file, well_mixed_ipfs, tmp_path)

    times = [datetime(1985, i + 1, 1) for i in range(4)]

    # Act
    data, _ = open_projectfile_data(wel_file)
    for wellname in data.keys():
        assert wellname in expected_dict.keys()
        fails, has_time, _ = expected_dict[wellname]
        if fails:
            with pytest.raises(ValueError):
                wel_cls.from_imod5_data(wellname, data, times=times)
        else:
            well = wel_cls.from_imod5_data(wellname, data, times=times)
            rate = well.dataset["rate"]
            if has_time:
                actual_set_to_zero = [
                    t.values
                    for t in rate.coords["time"]
                    if (rate.sel(time=t) == 0.0).all()
                ]
                if data[wellname]["has_associated"]:
                    expected_set_to_zero = []
                else:
                    expected_set_to_zero = [np.datetime64(t, "ns") for t in times[:-1]]
                diff = set(actual_set_to_zero) ^ set(expected_set_to_zero)
                assert len(diff) == 0
            else:
                assert "time" not in rate.dims
                assert "time" not in rate.coords
