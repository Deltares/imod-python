from shutil import copytree
from textwrap import dedent

import numpy as np
from pytest_cases import parametrize_with_cases

from imod.formats.prj import open_projectfile_data
from imod.mf6 import LayeredWell


class WellPrjCases:
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
            001,001
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

    def case_associated__all_fails(self):
       """FAIL: Varying factor over time"""
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


    def case_associated__fail(self):
        """FAIL: Assign same file to multiple layers"""
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

    def case_mixed__fails(self):
        """
        FAIL: Associated ipf defined not in first timestep or all
        timesteps.
        """
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


@parametrize_with_cases("wel_case", cases=WellPrjCases)
def test_import_wells(wel_case, tmp_path, current_cases):
    # Arrange
    case_name = current_cases["wel_case"].id
    wel_file = tmp_path / f"{case_name}.prj"
    
    with open(wel_file, "w") as f:
        f.write(wel_case)
    
    src = r"c:\Users\engelen\projects_wdir\imod-python\imod5_wel\ipf"
    copytree(src, tmp_path/"ipf")

    times = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02"), np.datetime64("2000-01-03")]

    # Act
    data, _ = open_projectfile_data(wel_file)
    well = LayeredWell.from_imod5_data("wel-1", data, times)
