from dataclasses import dataclass
from typing import cast, overload

from imod.typing import GridDataArray, GridDataset


@dataclass
class ValidationSettings:
    """
    Validation settings for MF6 model validation. Configuring
    :class:`imod.mf6.ValidationSettings` can help performance or reduce the
    strictness of validation for some packages, namely the Well and HFB package.

    Parameters
    ----------
    validate: bool, default=True
        Whether to perform validation.
    strict_well_validation: bool, default=True
        Whether to enforce strict validation for wells. If set to False, faulty
        wells are automatically removed during the writing process.
    strict_hfb_validation: bool, default=True
        Whether to enforce strict validation for HFBs. If set to False, faulty
        HFBs are automatically removed during the writing process.
    ignore_time: bool, default=False
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

    or provide it to :meth:`imod.mf6.Modflow6Simulation.set_validation_settings`:

    >>> sim = imod.mf6.Modflow6Simulation()
    >>> sim.set_validation_settings(settings)
    """

    validate: bool = True
    strict_well_validation: bool = True
    strict_hfb_validation: bool = True
    ignore_time: bool = False


@overload
def trim_time_dimension(ds: GridDataset, **kwargs) -> GridDataset: ...


@overload
def trim_time_dimension(ds: GridDataArray, **kwargs) -> GridDataArray: ...


def trim_time_dimension(
    ds: GridDataArray | GridDataset, **kwargs
) -> GridDataArray | GridDataset:
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
