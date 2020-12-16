import numpy as np
import pandas as pd
import xarray as xr

import imod


def stability_constraint_wel(wel, top_bot, porosity=0.3, R=1.0):
    r"""
    Computes sink/source stability constraint as applied in MT3D for adaptive
    timestepping (Zheng & Wang, 1999 p54).

    .. math:: \Delta t \leq \frac{R\theta }{\left | q_{s} \right |}

    For the WEL package, a flux is known
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
    top_bot["volume"] = np.abs(top_bot["dz"] * top_bot.dx * top_bot.dy)

    wel.loc[b, "qs"] = wel.loc[b, "Q"].abs() / top_bot["volume"].isel(indices).values
    wel.loc[b, "dt"] = R * porosity / wel.loc[b, "qs"]

    return wel


def stability_constraint_advection(front, lower, right, top_bot, porosity=0.3, R=1.0):
    r"""
    Computes advection stability constraint as applied in MT3D for adaptive
    timestepping (Zheng & Wang, 1999 p54):

    .. math:: \Delta t \leq \frac{R}{\frac{\left | v_{x} \right |}{\Delta x}+\frac{\left | v_{y} \right |}{\Delta y}+\frac{\left | v_{z} \right |}{\Delta z}}

    This function can be used to select
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
    """

    # top_bot reselect to bdg bounds
    top_bot = top_bot.sel(x=right.x, y=right.y, layer=right.layer)

    # Compute flow velocities
    qs_x, qs_y, qs_z = imod.evaluate.flow_velocity(
        front, lower, right, top_bot, porosity
    )

    if "dz" not in top_bot:
        top_bot["dz"] = top_bot["top"] - top_bot["bot"]
    # dz between layers is 0.5*dz_up + 0.5*dz_down
    dz_m = top_bot.dz.rolling(layer=2, min_periods=2).mean()
    dz_m = dz_m.shift(layer=-1)

    # assert all dz positive - Issue #140
    assert np.all(dz_m >= 0)

    # absolute velocities (m/d)
    abs_v_x = np.abs(qs_x / porosity)
    abs_v_y = np.abs(qs_y / porosity)
    abs_v_z = np.abs(qs_z / porosity)

    # dt of constituents (d)
    dt_x = R / (abs_v_x / top_bot.dx)
    dt_y = R / (abs_v_y / np.abs(top_bot.dy))
    dt_z = R / (abs_v_z / dz_m)

    # overall dt due to advection criterion (d)
    dt = 1.0 / (1.0 / dt_x + 1.0 / dt_y + 1.0 / dt_z)

    dt_xyz = xr.concat(
        (dt_x, dt_y, dt_z), dim=pd.Index(["x", "y", "z"], name="direction")
    )
    return dt, dt_xyz


def _calculate_intra_cell_dt(
    source_stage, source_cond, sink_stage, sink_cond, eff_volume
):
    """Calculate intra-cell dt by assuming a flux from a higher source_stage to a lower sink_stage,
    ignoring other head influences. Use limiting (lowest) conductance. eff_volume is the effective
    volume per cell (cell volume * effective porosity)"""
    cond = xr.ufuncs.minimum(source_cond, sink_cond)
    Q = cond * (source_stage - sink_stage)
    Q = Q.where(source_stage > sink_stage)

    return eff_volume / Q


def intra_cell_boundary_conditions(
    top_bot, porosity=0.3, riv=None, ghb=None, drn=None, drop_allnan=True
):
    """Function to pre-check boundary-conditions against one another for large intra-cell fluxes.
    ghb and river can function as source and sink, drn only as sink.

    Parameters
    ----------
    top_bot : xr.Dataset of floats
        'top_bot' should at least contain `top` and `bot` data_vars
    porosity : float or xr.DataArray of floats, optional
        Effective porosity of model cells
    riv : (dict or list of) imod.RiverPackage, optional
    ghb : (dict or list of) imod.GeneralHeadBoundaryPackage, optional
    drn : (dict or list of) imod.DrainagePackage, optional
    drop_allnan : boolean, optional
        Whether source-sink combinations without overlap should be dropped from result (default True)

    Returns
    -------
    dt_min: xr.DataArray of floats
        `dt_min` is the minimum calculated timestep over all combinations of boundary conditions
    dt_all: xr.DataArray of floats
        `dt_all` is the calculated timestep for all combinations of boundary conditions
    """
    if riv is None and ghb is None:
        raise ValueError(
            "At least one source boundary condition must be supplied through riv or ghb."
        )

    # convert all inputs to dictionaries of packages
    if riv is None:
        riv = {}
    elif isinstance(riv, dict):
        pass
    elif isinstance(riv, (list, tuple)):
        riv = {f"riv_{i}": l for i, l in enumerate(riv)}
    else:
        riv = {"riv": riv}

    if ghb is None:
        ghb = {}
    elif isinstance(ghb, dict):
        pass
    elif isinstance(ghb, (list, tuple)):
        ghb = {f"ghb_{i}": l for i, l in enumerate(ghb)}
    else:
        ghb = {"ghb": ghb}

    if drn is None:
        drn = {}
    elif isinstance(drn, dict):
        pass
    elif isinstance(drn, (list, tuple)):
        drn = {f"drn_{i}": l for i, l in enumerate(drn)}
    else:
        drn = {"drn": drn}

    # get sources and sinks:
    sources = {}
    sources.update(ghb)
    sources.update(riv)
    sinks = {}
    sinks.update(ghb)
    sinks.update(riv)
    sinks.update(drn)

    # determine effective volume
    if "dz" not in top_bot:
        top_bot["dz"] = top_bot["top"] - top_bot["bot"]
    eff_volume = top_bot["dz"] * top_bot.dx * np.abs(top_bot.dy) * porosity

    def _get_stage_name(sid):
        if sid in riv:
            return "stage"
        elif sid in ghb:
            return "head"
        elif sid in drn:
            return "elevation"

    # for all possible combinations: determine dt
    resultids = []
    results = []
    for sourceid, source in sources.items():
        for sinkid, sink in sinks.items():
            if sourceid == sinkid:
                continue
            comb = f"{sourceid}-{sinkid}"

            if comb not in resultids:
                # source in riv: only where stage > bottom elev
                if sourceid in riv:
                    source = source.where(source["stage"] > source["bottom_elevation"])

                dt = _calculate_intra_cell_dt(
                    source_stage=source[_get_stage_name(sourceid)],
                    source_cond=source["conductance"],
                    sink_stage=sink[_get_stage_name(sinkid)],
                    sink_cond=sink["conductance"],
                    eff_volume=eff_volume,
                )

                if not drop_allnan or not dt.isnull().all():
                    results.append(dt)
                    resultids.append(comb)
    dt_all = xr.concat(
        results, pd.Index(resultids, name="combination"), coords="minimal"
    )

    # overall dt
    dt_min = dt_all.min(dim="combination")

    return dt_min, dt_all
