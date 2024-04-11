import importlib
from textwrap import dedent
from zipfile import ZipFile

import numpy as np
import pooch
from numpy.testing import assert_allclose

from imod.formats.prj import open_projectfile_data

REGISTRY = pooch.create(
    path=pooch.os_cache("imod"),
    base_url="https://github.com/Deltares/imod-artifacts/raw/main/",
    version=None,
    version_dev="main",
    env="IMOD_DATA_DIR",
)

with importlib.resources.files("imod.data") as pkg_dir:
    REGISTRY.load_registry(pkg_dir / "registry.txt")
fname_model = REGISTRY.fetch("iMOD5_model.zip")


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

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_import_kh(factor=1.0, addition=0.0))
    file1.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_import_kh(factor=1.0, addition=1.2))
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_import_kh(factor=4.0, addition=3.0))
    file1.close()
    result_snippet_2 = open_projectfile_data(projects_file)

    assert_allclose(
        result_snippet_1[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] + 1.2
    )
    assert_allclose(
        result_snippet_2[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] * 4 + 3
    )


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
        result_snippet_1[0]["wel-1"]["dataframe"]["rate"]
        == 2 * result_snippet_0[0]["wel-1"]["dataframe"]["rate"] + 1.3
    )
    assert np.all(
        result_snippet_1[0]["wel-2"]["dataframe"]["rate"]
        == -1 * result_snippet_0[0]["wel-2"]["dataframe"]["rate"] + 0
    )
    assert np.all(
        result_snippet_1[0]["wel-3"]["dataframe"]["rate"]
        == 2 * result_snippet_0[0]["wel-2"]["dataframe"]["rate"] + 1.3
    )
