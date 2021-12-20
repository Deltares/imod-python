"""
Functions to load sample data.
"""
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import pkg_resources
import pooch

REGISTRY = pooch.create(
    path=pooch.os_cache("imod"),
    base_url="https://gitlab.com/deltares/imod/imod-artifacts/-/raw/main/",
    version=None,
    version_dev="main",
    env="IMOD_DATA_DIR",
)
with pkg_resources.resource_stream("imod.data", "registry.txt") as registry_file:
    REGISTRY.load_registry(registry_file)


def twri_output(path: Union[str, Path]) -> None:
    fname_twri = REGISTRY.fetch("ex01-twri-output.zip")
    with ZipFile(fname_twri) as archive:
        archive.extractall(path)
