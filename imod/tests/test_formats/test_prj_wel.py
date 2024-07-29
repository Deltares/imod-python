from textwrap import dedent

from pytest_cases import parametrize_with_cases


class WellPrjCases:
    def case_simple__first(self):
        return dedent(
            """
            0001,(WEL),1
            1900-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"simple.ipf"
            """
        )

    def case_simple__all(self):
        return dedent(
            """
            1900-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"simple1.ipf"
            1900-01-02
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"simple2.ipf"
            1900-01-03
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"simple3.ipf"
            """
        )

    def case_associated__first(self):
        return dedent(
            """
            0001,(WEL),1
            1900-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            """
        )
    
    def case_associated__all(self):
       return dedent(
            """
            0001,(WEL),1
            1900-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            1900-01-02
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            1900-01-03
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            """
       )

    def case_associated__all_fails(self):
       """FAIL: Varying factor over time"""
       return dedent(
            """
            0001,(WEL),1
            1900-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            1900-01-02
            001,001
            1,2, 001, 0.5, 0.0, -999.9900 ,"associated.ipf"
            1900-01-03
            001,001
            1,2, 001, 0.2, 0.0, -999.9900 ,"associated.ipf"
            """
       )


    def case_associated__fail(self):
        """FAIL: Assign same file to multiple layers"""
        return dedent(
            """
            0001,(WEL),1
            1900-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            1,2, 002, 0.75, 0.0, -999.9900 ,"associated.ipf"
            """
        )

    def case_mixed__first(self):
        return dedent(
            """
            0001,(WEL),1
            1900-01-01
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"simple.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            """
        )

    def case_mixed__all(self):
        return dedent(
            """
            0001,(WEL),1
            1900-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            1900-01-02
            001,002
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            1,2, 001, 1.0, 0.0, -999.9900 ,"simple.ipf"
            1900-01-03
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            """
        )

    def case_mixed__fails(self):
        """
        FAIL: Associated ipf defined not in first timestep or all
        timesteps.
        """
        return dedent(
            """
            0001,(WEL),1
            1900-01-01
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"simple.ipf"
            1900-01-02
            001,001
            1,2, 001, 1.0, 0.0, -999.9900 ,"associated.ipf"
            """
        )