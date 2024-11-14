import sys
from textwrap import dedent
from zipfile import ZipFile

import numpy as np
from numpy.testing import assert_allclose

import imod
from imod.data.sample_data import create_pooch_registry, load_pooch_registry
from imod.formats.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel

registry = create_pooch_registry()
registry = load_pooch_registry(registry)
fname_model = registry.fetch("iMOD5_model.zip")


def snippet_constant_kh(factor: float, addition: float, init: float):
    return dedent(f"""\
        0001,(KHV),1, Horizontal Permeability
        001,2
        1,1,1,{factor},{addition},{init}   >>> (KHV) Horizontal Permeability (IDF) <<<
        1,1,2,{factor},{addition},{init}   >>> (KHV) Horizontal Permeability (IDF) <<<
        """)


def test_import_constants(tmp_path):
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file1 = open(projects_file, "w")
    file1.write(snippet_constant_kh(factor=1.0, addition=0.0, init=1.1))
    file1.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_constant_kh(factor=1.0, addition=1.2, init=1.1))
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_constant_kh(factor=4.0, addition=3.0, init=1.1))
    file1.close()
    result_snippet_2 = open_projectfile_data(projects_file)

    assert_allclose(
        result_snippet_1[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] + 1.2
    )
    assert_allclose(
        result_snippet_2[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] * 4 + 3
    )


def snippet_idf_import_kh(factor: float, addition: float):
    return dedent(f"""\
        0001,(KHV),1, Horizontal Permeability
        001,2
        1,1,1,{factor},{addition},-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L1.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
        1,1,2,{factor},{addition},-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L2.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
        """)


def test_import_idf(tmp_path):
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file0 = open(projects_file, "w")
    file0.write(snippet_idf_import_kh(factor=1.0, addition=0.0))
    file0.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_import_kh(factor=1.0, addition=1.2))
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    file2 = open(projects_file, "w")
    file2.write(snippet_idf_import_kh(factor=4.0, addition=3.0))
    file2.close()
    result_snippet_2 = open_projectfile_data(projects_file)

    assert_allclose(
        result_snippet_1[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] + 1.2
    )
    assert_allclose(
        result_snippet_2[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] * 4 + 3
    )


def snippet_idf_import_transient(
    factor1: float, addition1: float, factor2: float, addition2: float
):
    return dedent(f"""\
        0002, (chd), 1, ConstantHead, ['head']
        2018-01-01 00:00:00                  
        001, 001
        1, 2, 001,{factor1},{addition1}, -9999., '.\Database\KHV\VERSION_1\IPEST_KHV_L1.IDF'
        2018-02-01 00:00:00
        001, 001
        1, 2, 001, {factor2},{addition2}, -9999., '.\Database\KHV\VERSION_1\IPEST_KHV_L1.IDF'
        """)


def test_import_idf_transient(tmp_path):
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file0 = open(projects_file, "w")
    file0.write(
        snippet_idf_import_transient(
            factor1=1.0, addition1=0.0, factor2=3.0, addition2=2.0
        )
    )
    file0.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    values_at_time_0 = result_snippet_0[0]["chd"]["head"].isel(time=0)
    values_at_time_1 = result_snippet_0[0]["chd"]["head"].isel(time=1)

    assert_allclose(values_at_time_0 * 3 + 2.0, values_at_time_1, rtol=1e-5, atol=1e-5)


def snippet_gen_import_hfb(factor: float, addition: float):
    return dedent(f"""\
        0001,(HFB),1, Horizontal Flow Barrier
        001,2
        1,2, 003,{factor},{addition},  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_BX.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
        1,2, 005,{factor},{addition},  -999.9900    ,'.\Database\HFB\VERSION_1\IBV2_HOOFDBREUKEN_SY.GEN' >>> (HFB) Horizontal Barrier Flow (GEN) <<<
        """)


def test_import_gen(tmp_path):
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file1 = open(projects_file, "w")
    file1.write(snippet_gen_import_hfb(factor=2.0, addition=1.3))
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    # the final multiplier is the product of the factor and the addition
    assert np.all(result_snippet_1[0]["hfb-1"]["geodataframe"]["multiplier"] == 2 * 1.3)


def snippet_gen_import_ipf(
    factor1: float, addition1: float, factor2: float, addition2: float
):
    return dedent(f"""\
        0001,(WEL),1, Wells
        STEADY-STATE
        001,003
        1,2,5,{factor1},{addition1},-999.99,                                       '.\Database\WEL\VERSION_1\WELLS_L3.IPF' >>> (WRA) Well Rate (IPF) <<<
        1,2,7,{factor2},{addition2},-999.99,                                       '.\Database\WEL\VERSION_1\WELLS_L4.IPF' >>> (WRA) Well Rate (IPF) <<<
        1,2,9,{factor1},{addition1},-999.99,                                       '.\Database\WEL\VERSION_1\WELLS_L5.IPF' >>> (WRA) Well Rate (IPF) <<<
        """)


def test_import_ipf(tmp_path):
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file1 = open(projects_file, "w")
    file1.write(
        snippet_gen_import_ipf(factor1=1.0, addition1=0.0, factor2=1.0, addition2=0.0)
    )
    file1.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(
        snippet_gen_import_ipf(factor1=2.0, addition1=1.3, factor2=-1.0, addition2=0.0)
    )
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    assert np.all(
        result_snippet_1[0]["wel-WELLS_L3"]["dataframe"][0]["rate"]
        == 2 * result_snippet_0[0]["wel-WELLS_L3"]["dataframe"][0]["rate"] + 1.3
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L4"]["dataframe"][0]["rate"]
        == -1 * result_snippet_0[0]["wel-WELLS_L4"]["dataframe"][0]["rate"] + 0
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L5"]["dataframe"][0]["rate"]
        == 2 * result_snippet_0[0]["wel-WELLS_L4"]["dataframe"][0]["rate"] + 1.3
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L3"]["dataframe"][0]["filt_top"] == 11.0
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L3"]["dataframe"][0]["filt_bot"] == 6.0
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L4"]["dataframe"][0]["filt_top"] == 11.0
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L4"]["dataframe"][0]["filt_bot"] == 6.0
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L5"]["dataframe"][0]["filt_top"] == 11.0
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L5"]["dataframe"][0]["filt_bot"] == 6.0
    )


def test_import_ipf_unique_id_and_logging(tmp_path):
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    logfile_path = tmp_path / "logfile.txt"

    try:
        with open(logfile_path, "w") as sys.stdout:
            # start logging
            imod.logging.configure(
                LoggerType.PYTHON,
                log_level=LogLevel.WARNING,
                add_default_file_handler=False,
                add_default_stream_handler=True,
            )
            projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

            file1 = open(projects_file, "w")
            file1.write(
                snippet_gen_import_ipf(
                    factor1=2.0, addition1=1.3, factor2=-1.0, addition2=0.0
                )
            )
            file1.close()

            # Act
            result_snippet_1 = open_projectfile_data(projects_file)
    finally:
        # turn the logger off again
        imod.logging.configure(
            LoggerType.NULL,
            log_level=LogLevel.WARNING,
            add_default_file_handler=False,
            add_default_stream_handler=False,
        )

    # test that id's were made unique
    # Assert
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L3"]["dataframe"][0]["id"] == "extractions"
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L4"]["dataframe"][0]["id"] == "extractions_1"
    )
    assert np.all(
        result_snippet_1[0]["wel-WELLS_L5"]["dataframe"][0]["id"] == "extractions_2"
    )

    with open(logfile_path, "r") as log_file:
        log = log_file.read()
        assert "This happened at x =\n197910, y = 362860, id = extractions" in log
        assert "appended with the suffix _1" in log
        assert "appended with the suffix _2" in log


def snippet_boundary_condition(factor: float, addition: float):
    return dedent(f"""\
        0001,(CHD),1, Constant Head
        STEADY-STATE
        001,2
        1,2,1,{factor},{addition},-999.99, './Database/SHD/VERSION_1/STATIONAIR/25/HEAD_STEADY-STATE_L1.IDF' >>> (SHD) Starting Heads (IDF) <<<
        1,2,2,{factor},{addition},-999.99, './Database/SHD/VERSION_1/STATIONAIR/25/HEAD_STEADY-STATE_L2.IDF' >>> (SHD) Starting Heads (IDF) <<<
        """)


def test_import_idf_boundary_condition(tmp_path):
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file1 = open(projects_file, "w")
    file1.write(snippet_boundary_condition(factor=1.0, addition=0.0))
    file1.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_boundary_condition(factor=2.0, addition=3.3))
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    assert np.all(
        result_snippet_1[0]["chd-1"]["head"]
        == 2 * result_snippet_0[0]["chd-1"]["head"] + 3.3
    )
    assert np.all(
        result_snippet_1[0]["chd-2"]["head"]
        == 2 * result_snippet_0[0]["chd-2"]["head"] + 3.3
    )


def snippet_idf_without_layer_dim(factor: float, addition: float):
    return dedent(f"""\
        0001,(RIV),1, Rivers
        STEADY-STATE
        004,001
        1,2,0,{factor},{addition},-999.99,'.\Database\RIV\VERSION_1\RIVER_PRIMAIR\IPEST_RIVER_PRIMAIR_COND_GEMIDDELD.IDF' >>> (CON) Conductance (IDF) <<<
        1,2,0,{factor},{addition},-999.99,     '.\Database\RIV\VERSION_1\RIVER_PRIMAIR\RIVER_PRIMAIR_STAGE_GEMIDDELD.IDF' >>> (RST) River Stage (IDF) <<<
        1,2,0,{factor},{addition},-999.99,                             '.\Database\RIV\VERSION_1\MAAS\BOTTOM19912011.IDF' >>> (RBT) River Bottom (IDF) <<<        
        1,2,0,{factor},{addition},-999.99,    '.\Database\RIV\VERSION_1\RIVER_PRIMAIR\RIVER_PRIMAIR_INFFCT_GEMIDDELD.IDF' >>> (RIF) Infiltration Factor (IDF) <<<        
        """)


def test_import_no_layer(tmp_path):
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_without_layer_dim(factor=1.0, addition=0.0))
    file1.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_without_layer_dim(factor=2.0, addition=3.3))
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    for key in ("stage", "conductance", "bottom_elevation", "infiltration_factor"):
        actual = result_snippet_1[0]["riv"][key].compute().values
        actual = actual[~np.isnan(actual)]
        expected = 2 * result_snippet_0[0]["riv"][key].compute().values + 3.3
        expected = expected[~np.isnan(expected)]
        assert abs((actual - expected).min()) < 1e-6
        assert abs((actual - expected).max()) < 1e-6
