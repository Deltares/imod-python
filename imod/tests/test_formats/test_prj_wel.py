from datetime import datetime
from shutil import copyfile
from textwrap import dedent

from pytest_cases import (
    get_all_cases,
    get_parametrize_args,
    parametrize,
    parametrize_with_cases,
)

from imod.formats.prj import open_projectfile_data
from imod.mf6 import LayeredWell


class WellPrjCases:
    """Cases for projectfile well records"""

    def case_simple__first(self):
        return dedent(
            """
            0001,(WEL),1
            1982-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__all(self):
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

    def case_simple__all(self):
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
    fails, {wellname: datetime_set_to_zero}
    """

    def case_simple__first(self):
        return False, {
            "wel-simple1": [datetime(1982, 2, 1), datetime(1982, 3, 1)],
        }
    def case_simple__all(self):
        return False, {
            "wel-simple1": [datetime(1982, 2, 1)],
            "wel-simple2": [datetime(1982, 1, 1), datetime(1982, 3, 1)],
            "wel-simple3": [datetime(1982, 1, 1), datetime(1982, 2, 1)],
        }
    def case_associated__first(self):
    def case_associated__all(self):
    def case_associated__all_varying_factors(self):
    def case_associated__multiple_layers_different_factors(self):
    def case_mixed__first(self):
    def case_mixed__all(self):
    def case_mixed__associated_second(self):    

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


@parametrize("wel_case, expected", argvalues=list(zip(PRJ_ARGS, READ_ARGS)))
def test_open_projectfile_data_wells(
    wel_case, expected, well_mixed_ipfs, tmp_path, request
):
    # Arrange
    case_name = request.node.callspec.id
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


@parametrize("wel_case, expected", argvalues=list(zip(PRJ_ARGS, READ_ARGS)))
def test_from_imod5_data_wells(wel_case, expected, well_mixed_ipfs, tmp_path, request):
    # Arrange
    case_name = request.node.callspec.id
    wel_file = tmp_path / f"{case_name}.prj"
    setup_test_files(wel_case, wel_file, well_mixed_ipfs, tmp_path)

    # Act
    data, _ = open_projectfile_data(wel_file)
    times = [datetime(1982, i + 1, 1) for i in range(4)]
    for wellname in data.keys():
        well = LayeredWell.from_imod5_data(wellname, data, times=times)
        pass
