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


class WellPrjCases:
    """Cases for projectfile well records"""
    def case_simple__first(self):
        return dedent(
            """
            0001,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )

    def case_simple__all(self):
        return dedent(
            """
            0003,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            2000-01-02
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple2.ipf"
            2000-01-03
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple3.ipf"
            """
        )

    def case_associated__first(self):
        return dedent(
            """
            0001,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_associated__all(self):
        return dedent(
            """
            0003,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            2000-01-02
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            2000-01-03
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_associated__all_varying_factors(self):
        return dedent(
            """
            0003,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            2000-01-02
            001,001
            1,2, 001, 0.5, 0.0, -999.9900 ,"ipf/associated.ipf"
            2000-01-03
            001,001
            1,2, 001, 0.2, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_associated__multiple_layers_different_factors(self):
        return dedent(
            """
            0001,(WEL),1
            2000-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1,2, 002, 0.75, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_mixed__first(self):
        return dedent(
            """
            0001,(WEL),1
            2000-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_mixed__all(self):
        return dedent(
            """
            0003,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            2000-01-02
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            2000-01-03
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

    def case_mixed__associated_second(self):
        return dedent(
            """
            0002,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            2000-01-02
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )

class WellReadCases:
    """Expected cases as interpreted by ``imod.formats.prj.open_projectfile_data``"""
    def case_simple__first(self):
        return {
            "wel-simple1": {
                "time": [datetime(2000, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }
    def case_simple__all(self):
        return {
            "wel-simple1": {
                "time": [datetime(2000, 1, 1), datetime(2000, 1, 2)],
                "layer": [1, 1],
                "factor": [1.0, 1.0],
                "addition": [0.0, 0.0],
            },
            "wel-simple2": {
                "time": [datetime(2000, 1, 2)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-simple3": {
                "time": [datetime(2000, 1, 3)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }
    def case_associated__first(self):
        return {
            "wel-associated": {
                "time": [datetime(2000, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            }
        }
    def case_associated__all(self):
        return {
            "wel-associated": {
                "time": [
                    datetime(2000, 1, 1),
                    datetime(2000, 1, 2),
                    datetime(2000, 1, 3),
                ],
                "layer": [1, 1, 1],
                "factor": [1.0, 1.0, 1.0],
                "addition": [0.0, 0.0, 0.0],
            },
        }
    def case_associated__all_varying_factors(self):
        return {
            "wel-associated": {
                "time": [
                    datetime(2000, 1, 1),
                    datetime(2000, 1, 2),
                    datetime(2000, 1, 3),
                ],
                "layer": [1, 1, 1],
                "factor": [1.0, 0.5, 0.2],
                "addition": [0.0, 0.0, 0.0],
            },
        }
    def case_associated__multiple_layers_different_factors(self):
        return {
            "wel-associated": {
                "time": [
                    datetime(2000, 1, 1),
                    datetime(2000, 1, 1),
                ],
                "layer": [1, 2],
                "factor": [1.0, 0.75],
                "addition": [0.0, 0.0],
            },
        }
    def case_mixed__first(self):
        return {
            "wel-simple1": {
                "time": [datetime(2000, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-associated": {
                "time": [datetime(2000, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }
    def case_mixed__all(self):
        return {
            "wel-associated": {
                "time": [
                    datetime(2000, 1, 1),
                    datetime(2000, 1, 2),
                    datetime(2000, 1, 3),
                ],
                "layer": [1, 1, 1],
                "factor": [1.0, 1.0, 1.0],
                "addition": [0.0, 0.0, 0.0],
            },
            "wel-simple1": {
                "time": [datetime(2000, 1, 2)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }
    def case_mixed__associated_second(self):    
        return {
            "wel-associated": {
                "time": [datetime(2000, 1, 2)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
            "wel-simple1": {
                "time": [datetime(2000, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }

# pytest_cases doesn't support any "zipped test cases", instead it takes the
# outer product of cases, when providing multiple case sets. To support 
# This we require the following workaround.
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

@parametrize("wel_case, expected",argvalues=list(zip(PRJ_ARGS, READ_ARGS)))
def test_open_projectfile_data_wells(wel_case, expected, well_mixed_ipfs, tmp_path, request):
    # Arrange
    case_name = request.node.callspec.id
    wel_file = tmp_path / f"{case_name}.prj"

    with open(wel_file, "w") as f:
        f.write(wel_case)

    ipf_dir = tmp_path / "ipf"
    ipf_dir.mkdir(exist_ok=True)

    # copy files to test folder
    for p in well_mixed_ipfs:
        copyfile(p, ipf_dir / p.name)

    # Act
    data, _ = open_projectfile_data(wel_file)
    assert len(set(expected.keys()) ^ set(data.keys())) == 0
    for key, wel_expected in expected.items():
        assert "time" in data[key]
        actual = data[key]
        assert actual["time"] == wel_expected["time"]
        assert actual["layer"] == wel_expected["layer"]
        assert actual["addition"] == wel_expected["addition"]
        assert actual["factor"] == wel_expected["factor"]
