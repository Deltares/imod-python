import os
import textwrap
from pathlib import Path

import pytest

import imod

ipf_associated_header = textwrap.dedent(
    """\
    3
    18
    x
    y
    q_m3
    FilterTopLevel
    FilterBottomLevel
    putcode
    FilterNo
    ALIAS
    StartDateTime
    SurfaceLevel
    WellTopLevel
    WellBottomLevel
    Status
    Type
    Comment
    CommentBy
    Site
    Organisation
    3,txt """
)

ipf_simple_header = textwrap.dedent(
    """\
    13
    9
    xcoord
    ycoord
    q_assigned
    top
    bot
    col
    row
    lay
    whichmodel
    0,txt
    """
)


def projectfile_string(tmp_path):
    return textwrap.dedent(
        f"""\
    0001,(WEL),1, Wells,[WRA]
    2000-01-01 00:00:00
    001,002
    1,2, 000,   1.000000    ,   0.000000    ,  -999.9900    ,'{tmp_path}\ipf1.ipf'
    1,2, 000,   1.000000    ,   0.000000    ,  -999.9900    ,'{tmp_path}\ipf2.ipf'
    """
    )


def ipf1_string_no_duplication():
    return textwrap.dedent(
        f"""\
    {ipf_associated_header}
    191231.52,406381.47,timeseries_wel1,4.11,-1.69,01-PP001,0,B46D0517,"30-11-1981 00:00",13.41,13.41,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    191171.96,406420.89,timeseries_wel1,3.78,-2.02,01-PP002,0,B46D0518,"30-11-1981 00:00",13.18,13.18,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    191112.11,406460.02,timeseries_wel1,3.81,-1.99,01-PP003,0,B46D0519,"30-11-1981 00:00",13.21,13.21,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    """
    )


def ipf1_string_duplication():
    return textwrap.dedent(
        f"""\
    {ipf_associated_header}
    191231.52,406381.47,timeseries_wel1,4.11,-1.69,01-PP001,0,B46D0517,"30-11-1981 00:00",13.41,13.41,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    191171.96,406420.89,timeseries_wel1,3.78,-2.02,01-PP002,0,B46D0518,"30-11-1981 00:00",13.18,13.18,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    191231.52,406381.47,other_timeseries_wel1,4.11,-1.69,01-PP001,0,B46D0517,"30-11-1981 00:00",13.41,13.41,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    """
    )


def ipf2_string():
    return textwrap.dedent(
        f"""\
    {ipf_associated_header}
    191231.52,406381.47,timeseries_wel1,4.11,-1.69,01-PP001,0,B46D0517,"30-11-1981 00:00",13.41,13.41,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    191171.96,406420.89,timeseries_wel1,3.78,-2.02,01-PP002,0,B46D0518,"30-11-1981 00:00",13.18,13.18,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    191112.11,406460.02,timeseries_wel1,3.81,-1.99,01-PP003,0,B46D0519,"30-11-1981 00:00",13.21,13.21,nan,Inactive,Vertical,"Oxmeer","User1 - Zeeland Water","Oxmeer","Zeeland Water"
    """
    )


def ipf_simple_string():
    return textwrap.dedent(
        f"""\
    {ipf_simple_header}
    276875.0,584375.0,-0.2,1.7,1.7,1107,162,1,4.0
    276875.0,584125.0,-0.1,4.7,4.7,1107,163,1,4.0
    276625.0,583875.0,-0.0,0.07,0.07,1106,164,1,4.0
    276875.0,583625.0,-0.1,-0.26,-0.26,1107,165,1,4.0
    276875.0,583375.0,-0.4,-2.65,-2.65,1107,166,1,4.0
    276875.0,583125.0,-0.1,3.1,3.1,1107,167,1,4.0
    277125.0,582875.0,-0.9,-0.45,-0.45,1108,168,1,4.0
    277125.0,582625.0,-0.6,-2.65,-2.65,1108,169,1,4.0
    277125.0,582375.0,-0.3,-2.65,-2.65,1108,170,1,4.0
    277125.0,582125.0,-0.2,-2.65,-2.65,1108,171,1,4.0
    277125.0,581875.0,-0.2,-2.65,-2.65,1108,172,1,4.0
    277125.0,581625.0,-0.3,2.54,2.54,1108,173,1,4.0
    277125.0,581375.0,0.2,1.6,1.6,1108,174,1,4.0
    277125.0,581125.0,0.5,0.6,0.6,1108,175,1,4.0
    """
    )


def timeseries_string():
    return textwrap.dedent(
        """\
    6
    2
    DATE,-9999.0
    MEASUREMENT,-9999.0
    19811130,-676.1507971461288
    19811231,-766.7777419354838
    19820131,-847.6367741935485
    19820228,-927.3857142857142
    19820331,-859.2109677419355
    19820430,-882.7713333333334
    """
    )


def other_timeseries_string():
    return textwrap.dedent(
        """\
    6
    2
    DATE,-9999.0
    MEASUREMENT,-9999.0
    19811130,-174.1507971461288
    19811231,-166.7777419354838
    19820131,-147.6367741935485
    19820228,-127.3857142857142
    19820331,-159.2109677419355
    19820430,-182.7713333333334
    """
    )


def out_of_bounds_timeseries_string():
    return textwrap.dedent(
        """\
    6
    2
    DATE,-9999.0
    MEASUREMENT,-9999.0
    19811130,-174.1507971461288
    19811231,-166.7777419354838
    19820131,-147.6367741935485
    19820228,-127.3857142857142
    19820331,-159.2109677419355
    29991112,-182.7713333333334
    """
    )


def write_ipf_assoc_files(
    projectfile_str,
    ipf1_str,
    ipf2_str,
    timeseries_wel1_str,
    tmp_path,
    other_timeseries_string=None,
):
    with open(Path(tmp_path) / "projectfile.prj", "w") as f:
        f.write(projectfile_str)

    with open(Path(tmp_path) / "ipf1.ipf", "w") as f:
        f.write(ipf1_str)

    with open(Path(tmp_path) / "ipf2.ipf", "w") as f:
        f.write(ipf2_str)

    with open(Path(tmp_path) / "timeseries_wel1.txt", "w") as f:
        f.write(timeseries_wel1_str)

    if other_timeseries_string is not None:
        with open(Path(tmp_path) / "other_timeseries_wel1.txt", "w") as f:
            f.write(other_timeseries_string)

    return Path(tmp_path) / "projectfile.prj"


def write_ipf_mixed_files(
    ipf_assoc_str, ipf_simple_str, timeseries_wel1_str, other_timeseries_str, tmp_path
):
    file_dict = {
        "associated.ipf": ipf_assoc_str,
        "simple1.ipf": ipf_simple_str,
        "simple2.ipf": ipf_simple_str,
        "simple3.ipf": ipf_simple_str,
        "timeseries_wel1.txt": timeseries_wel1_str,
        "other_timeseries_wel1.txt": other_timeseries_str,
    }

    paths = []
    for file, string in file_dict.items():
        path = Path(tmp_path) / file
        paths.append(path)
        with open(path, "w") as f:
            f.write(string)

    return paths


@pytest.fixture(scope="session")
def well_regular_import_prj():
    tmp_path = imod.util.temporary_directory()
    os.makedirs(tmp_path)

    projectfile_str = projectfile_string(tmp_path)
    ipf1_str = ipf1_string_no_duplication()
    ipf2_str = ipf2_string()
    timeseries_well_str = timeseries_string()

    return write_ipf_assoc_files(
        projectfile_str, ipf1_str, ipf2_str, timeseries_well_str, tmp_path
    )


@pytest.fixture(scope="session")
def well_duplication_import_prj():
    tmp_path = imod.util.temporary_directory()
    os.makedirs(tmp_path)

    projectfile_str = projectfile_string(tmp_path)
    ipf1_str = ipf1_string_duplication()
    ipf2_str = ipf2_string()
    timeseries_well_str = timeseries_string()
    other_timeseries_well_str = other_timeseries_string()
    return write_ipf_assoc_files(
        projectfile_str,
        ipf1_str,
        ipf2_str,
        timeseries_well_str,
        tmp_path,
        other_timeseries_well_str,
    )


@pytest.fixture(scope="session")
def well_mixed_ipfs():
    tmp_path = imod.util.temporary_directory()
    os.makedirs(tmp_path)

    ipf_assoc_str = ipf1_string_duplication()
    ipf_simple_str = ipf_simple_string()
    timeseries_well_str = timeseries_string()
    other_timeseries_well_str = other_timeseries_string()

    return write_ipf_mixed_files(
        ipf_assoc_str,
        ipf_simple_str,
        timeseries_well_str,
        other_timeseries_well_str,
        tmp_path,
    )


@pytest.fixture(scope="session")
def well_out_of_bounds_ipfs():
    tmp_path = imod.util.temporary_directory()
    os.makedirs(tmp_path)

    ipf_assoc_str = ipf1_string_duplication()
    ipf_simple_str = ipf_simple_string()
    timeseries_well_str = timeseries_string()
    other_timeseries_well_str = out_of_bounds_timeseries_string()

    return write_ipf_mixed_files(
        ipf_assoc_str,
        ipf_simple_str,
        timeseries_well_str,
        other_timeseries_well_str,
        tmp_path,
    )
