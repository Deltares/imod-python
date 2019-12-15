import imod
import pandas as pd
import numpy as np
import xarray as xr


def stability_constraint_wel(wel, top_bot, porosity=0.3, R=1.0):
    """
    Computes sink/source stability constraint as applied in MT3D for adaptive
    timestepping (Zheng & Wang, 1999 p54). For the WEL package, a flux is known
    beforehand, so we can evaluate beforehand if a flux assigned to a cell
    will necessitate a small timestap, thereby slowing down calculations.

    Returns a ipf DataFrame that includes a column for the specific discharge and
    resulting minimum timestep.

    Parameters
    ----------
    wel: pd.DataFrame
        pd.DataFrame that defines a WEL package. Minimally includes
        x, y, layer and Q column.
    top_bot: xr.Dataset of floats, containing 'top', 'bot' and optionally
        'dz' of layers.
        Dimensions must be exactly ``("layer", "y", "x")``.
    porosity: float or xr.DataArray of floats, optional (default 0.3)
        If xr.DataArray, dimensions must be exactly ``("layer", "y", "x")``.
    R: Retardation factor, optional (default)
        Only when sorption is a factor.

    Returns
    -------
    wel: pd.DataFrame containing addition qs (specific discharge) and 
        dt (minimum timestep) columns
    """

    b = imod.select.points_in_bounds(
        top_bot, x=wel["x"], y=wel["y"], layer=wel["layer"]
    )
    indices = imod.select.points_indices(
        top_bot, x=wel.loc[b, "x"], y=wel.loc[b, "y"], layer=wel.loc[b, "layer"]
    )

    if "dz" not in top_bot:  # create if not present
        top_bot["dz"] = top_bot["top"] - top_bot["bot"]
    top_bot["volume"] = top_bot["dz"] * top_bot.dx * -top_bot.dy

    wel.loc[b, "qs"] = wel.loc[b, "Q"].abs() / top_bot["volume"].isel(indices).values
    wel.loc[b, "dt"] = R * porosity / wel.loc[b, "qs"]

    return wel


def stability_constraint_advection(front, lower, right, top_bot, porosity=0.3, R=1.0):
    """
    Computes advection stability constraint as applied in MT3D for adaptive
    timestepping (Zheng & Wang, 1999 p54). This function can be used to select
    which cells necessitate a small timestap, thereby slowing down calculations.

    Front, lower, and right arguments refer to iMOD face flow budgets, in cubic
    meters per day. In terms of flow direction these are defined as:

    * ``front``: positive with ``y`` (negative with row index)
    * ``lower``: positive with ``layer`` (positive with layer index)
    * ``right``: negative with ``x`` (negative with column index)

    Returns the minimum timestep that is required to satisfy this constraint.
    The resulting dt xr.DataArray is the minimum timestep over all three directions,
    dt_xyz is an xr.Dataset containing minimum timesteps for the three directions 
    separately.

    Parameters
    ----------
    front: xr.DataArray of floats, optional
        Dimensions must be exactly ``("layer", "y", "x")``.
    lower: xr.DataArray of floats, optional
        Dimensions must be exactly ``("layer", "y", "x")``.
    right: xr.DataArray of floats, optional
        Dimensions must be exactly ``("layer", "y", "x")``.
    top_bot: xr.Dataset of floats, containing 'top', 'bot' and optionally
        'dz' of layers.
        Dimensions must be exactly ``("layer", "y", "x")``.
    porosity: float or xr.DataArray of floats, optional (default 0.3)
        If xr.DataArray, dimensions must be exactly ``("layer", "y", "x")``.
    R: Retardation factor, optional (default)
        Only when sorption is a factor.

    Returns
    -------
    dt: xr.DataArray of floats
    dt_xyz: xr.Dataset of floats

    Examples
    --------
    Load the face flows, and select the last time using index selection:
    """

    # top_bot reselect to bdg bounds
    top_bot = top_bot.sel(x=right.x, y=right.y, layer=right.layer)

    if "dz" not in top_bot:
        top_bot["dz"] = top_bot["top"] - top_bot["bot"]
    top_bot["volume"] = top_bot["dz"] * top_bot.dx * -top_bot.dy

    # average to cell centre
    right = right.rolling(x=2, min_periods=2).mean().shift(x=-1)
    front = front.rolling(y=2, min_periods=2).mean().shift(y=-1)
    lower = lower.rolling(layer=2, min_periods=2).mean().shift(layer=-1)

    volume = top_bot["volume"] * porosity

    # dt of constituents (d), make absolute
    dt_x = np.abs(volume / right)
    dt_y = np.abs(volume / front)
    dt_z = np.abs(volume / lower)

    dt_xyz = xr.concat(
        (dt_x, dt_y, dt_z), dim=pd.Index(["x", "y", "z"], name="direction")
    )

    # overall dt due to advection criterion (d)
    dt = dt_xyz.min(dim="direction")

    return dt, dt_xyz
