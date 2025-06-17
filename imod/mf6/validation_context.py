from dataclasses import dataclass

from imod.typing import GridDataset


@dataclass
class ValidationContext:
    validate: bool = True
    strict_well_validation: bool = True
    strict_hfb_validation: bool = True
    ignore_time_no_data: bool = False


def trim_time_dimension(ds: GridDataset, **kwargs) -> GridDataset:
    """
    Prepare object for validation, drop time dimension if
    ignore_time_no_data is set in validation context.
    """
    if "validation_context" in kwargs:
        validation_context = kwargs["validation_context"]
        if validation_context.ignore_time_no_data:
            # If ignore_time_no_data is set, we can ignore time dimension
            if "time" in ds.dims:
                return ds.isel(time=0)
    return ds
