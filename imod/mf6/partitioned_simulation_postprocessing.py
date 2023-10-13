from pathlib import Path
from typing import List

import imod
from imod.typing.grid import merge


def merge_heads(simulation_dir: Path, model_names: List[str]):
    heads = []
    for modelname in model_names:
        modelDirectory = simulation_dir / modelname
        grb_path = get_grb_file_path(modelDirectory)
        head = imod.mf6.open_hds(
            modelDirectory / f"{modelname}.hds",
            grb_path,
        )
        heads.append(head)

    head = merge(*heads)
    return head


def get_grb_file_path(model_directory: Path) -> Path:
    """
    Returns the path of the grb file.
    """
    grb_files = list(model_directory.glob("**/*.grb"))
    if len(grb_files) != 1:
        raise RuntimeError(
            f"there should be exactly one grb file in directory {model_directory}."
        )
    return grb_files[0]
