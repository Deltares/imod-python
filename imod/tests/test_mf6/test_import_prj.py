import imod
from zipfile import ZipFile
import pooch
from imod.formats.prj import open_projectfile_data
import importlib

REGISTRY = pooch.create(
    path=pooch.os_cache("imod"),
    base_url="https://github.com/Deltares/imod-artifacts/raw/main/",
    version=None,
    version_dev="main",
    env="IMOD_DATA_DIR",
)    
snippet = ( \
"""
0001,(KHV),1, Horizontal Permeability
001,2
1,1,1,1.0,0.0,1.0   >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,2,1.0,0.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L2.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
""",
"""
0001,(KHV),1, Horizontal Permeability
001,2
1,1,1,1.0,1.2,1.0   >>> (KHV) Horizontal Permeability (IDF) <<<
1,2,2,1.0,1.0,-999.99, '.\Database\KHV\VERSION_1\IPEST_KHV_L2.IDF' >>> (KHV) Horizontal Permeability (IDF) <<<
""",
)
def test_import(tmp_path):

    with importlib.resources.files("imod.data") as pkg_dir:
        REGISTRY.load_registry(pkg_dir / "registry.txt")    
    fname_model = REGISTRY.fetch("iMOD5_model.zip")    
    with ZipFile(fname_model) as archive:
        archive.extractall(tmp_path)

    projects_file = tmp_path/"iMOD5_model_pooch"/"iMOD5_model.prj"

    file1 = open(projects_file, "w")
    file1.write(snippet[0])
    file1.close()
    idict_0 =  open_projectfile_data(projects_file)    

    file1 = open(projects_file, "w")
    file1.write(snippet[1])
    file1.close()
    idict_1 =  open_projectfile_data(projects_file)   

    pass