import importlib
from zipfile import ZipFile

import pooch

from imod.formats.prj import open_projectfile_data
from numpy.testing import assert_allclose

REGISTRY = pooch.create(
    path=pooch.os_cache("imod"),
    base_url="https://github.com/Deltares/imod-artifacts/raw/main/",
    version=None,
    version_dev="main",
    env="IMOD_DATA_DIR",
)
def snippet_constant(factor: float, addition: float, init: float):
    return f"""
0001,(KHV),1, Horizontal Permeability
001,1
1,1,1,{factor},{addition},{init}   >>> (KHV) Horizontal Permeability (IDF) <<<
"""

def snippet_idf_import(factor: float, addition: float):
    return f"""
0001,(KHV),1, Horizontal Permeability
001,1
1,1,1,{factor},{addition},-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L1.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
"""



def test_import_constants(tmp_path):
    with importlib.resources.files("imod.data") as pkg_dir:
        REGISTRY.load_registry(pkg_dir / "registry.txt")
    fname_model = REGISTRY.fetch("iMOD5_model.zip")
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file1 = open(projects_file, "w")
    file1.write(snippet_constant(factor=1.0, addition= 0.0, init= 1.1))
    file1.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_constant(factor=1.0, addition= 1.2, init= 1.1))
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_constant(factor=4.0, addition= 3.0, init= 1.1))
    file1.close()
    result_snippet_2 = open_projectfile_data(projects_file)

    assert_allclose(result_snippet_1[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] + 1.2 )
    assert_allclose(result_snippet_2[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] *4 + 3 ) 

def test_import_idf(tmp_path):
    with importlib.resources.files("imod.data") as pkg_dir:
        REGISTRY.load_registry(pkg_dir / "registry.txt")
    fname_model = REGISTRY.fetch("iMOD5_model.zip")
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path / "iMOD5_model_pooch" / "iMOD5_model.prj"

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_import(factor=1.0, addition= 0.0))
    file1.close()
    result_snippet_0 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_import(factor=1.0, addition=1.2))
    file1.close()
    result_snippet_1 = open_projectfile_data(projects_file)

    file1 = open(projects_file, "w")
    file1.write(snippet_idf_import(factor=4.0, addition= 3.0))
    file1.close()
    result_snippet_2 = open_projectfile_data(projects_file)

    assert_allclose(result_snippet_1[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] + 1.2 )
    assert_allclose(result_snippet_2[0]["khv"]["kh"], result_snippet_0[0]["khv"]["kh"] *4 + 3 ) 