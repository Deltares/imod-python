from datetime import datetime
from shutil import copyfile
from textwrap import dedent

from pytest_cases import parametrize_with_cases

from imod.formats.prj import open_projectfile_data


class WellPrjCases:
    def case_simple__first(self):
        prj_string = dedent(
            """
            0001,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            """
        )
        expected = {
            "wel-simple1": {
                "time": [datetime(2000, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            },
        }
        return prj_string, expected

    def case_simple__all(self):
        prj_string = dedent(
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
        expected = {
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
        return prj_string, expected

    def case_associated__first(self):
        prj_string = dedent(
            """
            0001,(WEL),1
            2000-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )
        expected = {
            "wel-associated": {
                "time": [datetime(2000, 1, 1)],
                "layer": [1],
                "factor": [1.0],
                "addition": [0.0],
            }
        }
        return prj_string, expected

    def case_associated__all(self):
        prj_string = dedent(
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
        expected = {
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
        return prj_string, expected

    def case_associated__all_varying_factors(self):
        prj_string = dedent(
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
        expected = {
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
        return prj_string, expected

    def case_associated__multiple_layers_different_factors(self):
        prj_string = dedent(
            """
            0001,(WEL),1
            2000-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            1,2, 002, 0.75, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )
        expected = {
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
        return prj_string, expected

    def case_mixed__first(self):
        prj_string = dedent(
            """
            0001,(WEL),1
            2000-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/simple1.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"ipf/associated.ipf"
            """
        )
        expected = {
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
        return prj_string, expected

    def case_mixed__all(self):
        prj_string = dedent(
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
        expected = {
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
        return prj_string, expected

    def case_mixed__associated_second(self):
        prj_string = dedent(
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
        expected = {
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
        return prj_string, expected


@parametrize_with_cases("wel_case, expected", cases=WellPrjCases)
def test_import_wells(wel_case, expected, well_mixed_ipfs, tmp_path, current_cases):
    # Arrange
    case_name = current_cases["wel_case"].id
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
