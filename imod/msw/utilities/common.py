from pathlib import Path

from imod.typing import GridDataArray
from imod.typing.grid import concat


def concat_imod5(arg1: GridDataArray, arg2: GridDataArray) -> GridDataArray:
    return concat([arg1, arg2], dim="subunit").assign_coords(subunit=[0, 1])


def find_in_file_list(filename: str, paths: list[str]) -> str:
    for file in paths:
        if filename == Path(file[0]).name.lower():
            return file[0]
    raise ValueError(f"could not find {filename} in list of paths: {paths}")
