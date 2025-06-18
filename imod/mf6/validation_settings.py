from dataclasses import dataclass
from typing import cast

from imod.typing import GridDataset


@dataclass
class ValidationSettings:
    """
    Validation settings for MF6 model validation.

    Parameters
    ----------
    validate: bool
        Whether to perform validation.
    strict_well_validation: bool
        Whether to enforce strict validation for wells.
    strict_hfb_validation: bool
        Whether to enforce strict validation for HFBs.
    ignore_time: bool
        If True, ignore time dimension in validation. Instead, select first
        timestep of dataset. This can save a lot of time during writing when the
        time dimension is not relevant for the validation process. Especially
        when boundary conditions do not have cell activity changes over time.
    
    Examples
    --------
    
    >>> import imod
    >>> settings = imod.mf6.ValidationSettings(validate=True, strict_well_validation=False)

    You can also set attributes directly after instantiation:
    
    >>> settings.ignore_time = True

    You can provide the settings to :class:`imod.mf6.Modflow6Simulations`:

    >>> sim = imod.mf6.Modflow6Simulation(validation_context=settings)
    """
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
