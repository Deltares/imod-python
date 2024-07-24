from typing import Optional

import xarray as xr

from imod.mf6.chd import ConstantHead


def concat_layered_chd_packages(
    name: str,
    dict_packages: dict[str, ConstantHead],
    remove_merged_packages: bool = True,
) -> Optional[ConstantHead]:
    """

    Parameters
    ----------
    name: str
        The name of the package that was split over layers.
        If they are called "chd-1" and so on then set name to "chd
    dict_packages: dict[str, ConstantHead]
        dictionary  with package names as key and the packages as values
    remove_merged_packages: bool = True
        set to True to remove merged packages from dict_packages

    This function merges chd-packages whose name starts with "name" into a a
    single chd package. This is aimed at chd packages that are split over
    layers- so we would have chd-1, chd-2 and so on and these packages would
    define a chd package for layer 1, 2 and so on. This function merges them
    into a single chd package. If remove_merged_packages is True, then the
    packages that are concatenated are removed from the input dictionary, so
    that this on output only contains the packages that were not merged.
    """

    candidate_keys = [k for k in dict_packages.keys() if name in k]
    if len(candidate_keys) == 0:
        return None

    dataset_list = []
    for key in candidate_keys:
        pack = dict_packages[key]
        dataset_list.append(pack.dataset)
        if remove_merged_packages:
            dict_packages.pop(key)

    concat_dataset = xr.concat(
        dataset_list, dim="layer", compat="equals", data_vars="different"
    )
    return ConstantHead._from_dataset(concat_dataset)
