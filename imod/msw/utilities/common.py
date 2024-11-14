from imod.typing import GridDataArray
from imod.typing.grid import concat


def concat_imod5(arg1: GridDataArray, arg2: GridDataArray) -> GridDataArray:
    return concat([arg1, arg2], dim="subunit").assign_coords(subunit=[0, 1])
