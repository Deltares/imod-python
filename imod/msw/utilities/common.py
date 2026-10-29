from pathlib import Path

from xarray.core.utils import is_scalar

from imod.typing import GridDataArray
from imod.typing.grid import concat


def _is_equal_scalar_value(da: GridDataArray, value: float) -> bool:
    """
    Helper function to guarantee that the check in ``is_scalar_and_equal`` can
    shortcut after is_scalar returns False.
    """
    return bool(da.to_numpy()[()] == value)


def is_scalar_and_equal(da: GridDataArray, value: float) -> bool:
    """
    Check if a GridDataArray is equal to a scalar value. Shortcuts if the array
    is not scalar to preserve performance.
    """
    return is_scalar(da) and _is_equal_scalar_value(da, value)


def concat_imod5(arg1: GridDataArray, arg2: GridDataArray) -> GridDataArray:
    return concat([arg1, arg2], dim="subunit").assign_coords(subunit=[0, 1])


def find_in_file_list(filename: str, paths: list[str]) -> str:
    for file in paths:
        if filename == Path(file[0]).name.lower():
            return file[0]
    raise ValueError(f"could not find {filename} in list of paths: {paths}")
