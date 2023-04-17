import pytest
import xarray as xr
from imod.mf6.lak import Lake
import pathlib
import imod.tests.fixtures.mf6_lake_package_fixture as mf_lake
import textwrap

@pytest.mark.usefixtures("naardermeer", "ijsselmeer")

def test_mf6_write_number_tables(naardermeer, ijsselmeer, tmp_path):
    directory = pathlib.Path("mymodel")
    lake_package_2lakes = Lake.from_lakes_and_outlets([naardermeer(has_lake_table=True), ijsselmeer(has_lake_table=True)])
    actual = mf_lake.write_and_read(lake_package_2lakes, tmp_path, "lake-test")
    assert "ntables 2" in actual


    lake_package_1lakes = Lake.from_lakes_and_outlets([naardermeer(has_lake_table=False), ijsselmeer(has_lake_table=True)])
    actual = mf_lake.write_and_read(lake_package_1lakes, tmp_path, "lake-test")
    assert "ntables 1" in actual

    lake_package_0lakes = Lake.from_lakes_and_outlets([naardermeer(has_lake_table=False), ijsselmeer(has_lake_table=False)])
    actual = mf_lake.write_and_read(lake_package_0lakes, tmp_path, "lake-test")
    assert "ntables 0" in actual



def test_mf6_laketable_reference(naardermeer, ijsselmeer, tmp_path):

    lake_package_2lakes = Lake.from_lakes_and_outlets([naardermeer(has_lake_table=True), ijsselmeer(has_lake_table=True)])
    actual = mf_lake.write_and_read(lake_package_2lakes, tmp_path, "lake-test")
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          nlakes 2
          noutlets 0
          ntables 2
        end dimensions

        begin packagedata
          1 11.0 3 Naardermeer
          2 15.0 3 IJsselmeer
        end packagedata

        begin connectiondata
        1 1 1 2 2 horizontal 0.2 0.4 0.3 0.6 0.5
        1 2 1 2 3 horizontal 0.2 0.4 0.3 0.6 0.5
        1 3 1 3 3 horizontal 0.2 0.4 0.3 0.6 0.5
        2 1 1 5 5 horizontal 0.2 0.4 0.3 0.6 0.5
        2 2 1 5 6 horizontal 0.2 0.4 0.3 0.6 0.5
        2 3 1 6 6 horizontal 0.2 0.4 0.3 0.6 0.5
        end connectiondata

        begin tables
           1  TAB6 FILEIN Naardermeer.ltbl
           2  TAB6 FILEIN IJsselmeer.ltbl
        end tables
        """
    )

    assert actual == expected

    with open(tmp_path / "Naardermeer.ltbl", "r") as f:
        actual_table_naardermeer = f.read()
        expected_table_naardermeer  = textwrap.dedent(
        """\
        BEGIN DIMENSIONS
        NROW 3
        NCOL 3
        END DIMENSIONS
        begin table
        4.0 6.0 5.0
        5.0 7.0 6.0
        6.0 8.0 7.0end table
        """
        )
    assert actual_table_naardermeer == expected_table_naardermeer

    with open(tmp_path / "IJsselmeer.ltbl", "r") as f:
        actual_table_ijsselmeer = f.read()
        expected_table_ijsselmeer  = textwrap.dedent(
        """\
        BEGIN DIMENSIONS
        NROW 6
        NCOL 3
        END DIMENSIONS
        begin table
         8.0 10.0  9.0
         9.0 11.0 10.0
        10.0 12.0 11.0
        11.0 13.0 12.0
        12.0 14.0 13.0
        13.0 15.0 14.0end table
        """
        )
    assert actual_table_ijsselmeer == expected_table_ijsselmeer
