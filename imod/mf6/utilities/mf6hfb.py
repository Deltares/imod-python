from typing import List

import xarray as xr

from imod.mf6.hfb import (
    HorizontalFlowBarrierBase,
    _prepare_barrier_dataset_for_mf6_adapter,
)
from imod.mf6.mf6_hfb_adapter import Mf6HorizontalFlowBarrier
from imod.typing import GridDataArray


def create_empty_da(dims: tuple[str, str, str]) -> xr.DataArray:
    return xr.DataArray([[[], []], [[], []]], dims=dims)


def inverse_sum(a: xr.Dataset) -> xr.Dataset:
    """Sum of the inverse"""
    return (1 / a).sum()


def merge_hfb_packages(
    hfb_ls: List[HorizontalFlowBarrierBase],
    idomain: GridDataArray,
    top: GridDataArray,
    bottom: GridDataArray,
    k: GridDataArray,
    strict_hfb_validation: bool = True,
) -> Mf6HorizontalFlowBarrier:
    """
    Merges HorizontalFlowBarrier packages into single package as MODFLOW 6
    doesn't support multiple HFB packages.

    Parameters
    ----------
    hfb_ls: list
        List of HorizontalFlowBarrier packages. These will be merged into one.
        Function takes settings like "print_input" from the first object in the
        list.
    idomain: GridDataArray
            Grid with active cells.
    top: GridDataArray
        Grid with top of model layers.
    bottom: GridDataArray
        Grid with bottom of model layers.
    k: GridDataArray
        Grid with hydraulic conductivities.
    strict_hfb_validation: bool
        Turn on strict horizontal flow barrier validation.
    """

    barrier_ls = [
        hfb._to_connected_cells_dataset(idomain, top, bottom, k, strict_hfb_validation)
        for hfb in hfb_ls
    ]
    barrier_dataset = xr.concat(barrier_ls, dim="cell_id")
    # Catch entirely empty case
    mf6_hfb_dims = ("cell_dims1", "cell_dims2", "cell_id")
    if len(barrier_dataset.coords["cell_id"]) == 0:
        empty_da = create_empty_da(mf6_hfb_dims)
        empty_data_vars = 4 * [empty_da]
        return Mf6HorizontalFlowBarrier(*empty_data_vars)

    # xarray GroupbyDataset doesn't allow reducing with different methods per variable.
    # Therefore groupby twice: once for cell_id, once for hydraulic_characteristic.
    cell_id_merged = (
        barrier_dataset[["cell_id1", "cell_id2"]].groupby("cell_id").first()
    )
    hc_merged = 1 / barrier_dataset[["hydraulic_characteristic"]].groupby(
        "cell_id"
    ).map(inverse_sum)
    # Force correct dim order
    cell_id_merged = cell_id_merged.transpose(*mf6_hfb_dims)
    # Merge datasets into one
    barrier_dataset_merged = xr.merge([cell_id_merged, hc_merged], join="exact")
    # Set leftover options
    barrier_dataset_merged["print_input"] = hfb_ls[0].dataset["print_input"]
    barrier_dataset_merged = _prepare_barrier_dataset_for_mf6_adapter(
        barrier_dataset_merged
    )
    return Mf6HorizontalFlowBarrier(**barrier_dataset_merged.data_vars)
