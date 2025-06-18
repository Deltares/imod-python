from dataclasses import dataclass
from typing import cast

from imod.typing import GridDataset


@dataclass
class ValidationSettings:
    validate: bool = True
    strict_well_validation: bool = True
    strict_hfb_validation: bool = True
    ignore_time: bool = False


def trim_time_dimension(ds: GridDataset, **kwargs) -> GridDataset:
    """
    Prepare object for validation, drop time dimension if
    ignore_time_no_data is set in validation context.
    """
    if "validation_context" in kwargs:
        validation_context = cast(ValidationSettings, kwargs["validation_context"])
        if validation_context.ignore_time:
            # If ignore_time_no_data is set, we can ignore time dimension
            return ds.isel(time=0, missing_dims="ignore")
    return ds
