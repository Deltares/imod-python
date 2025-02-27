from imod.typing import GridDataArray
from imod.typing.grid import zeros_like

def create_layered_top(bottom: GridDataArray, top: GridDataArray) -> GridDataArray:
    """
    Create a top array with a layer dimension, from a top array with no layer
    dimension and a bottom array with a layer dimension. The (output) top of
    layer n is assigned the bottom of layer n-1.

    Parameters
    ----------
    bottom: {DataArray, UgridDataArray}
        Bottoms with layer dimension
    top: {DataArray, UgridDataArray}
        Top, without layer dimension

    Returns
    -------
    new_top: {DataArray, UgridDataArray}
        Top with layer dimension.
    """
    new_top = zeros_like(bottom)
    new_top[0] = top
    new_top[1:] = bottom[0:-1].values

    return new_top
