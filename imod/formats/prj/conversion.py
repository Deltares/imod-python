from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import pygeos
import xarray as xr
import xugrid as xu

import imod


def regrid(
    source,
    target,
    method,
    original2d=None,
):
    if source.dims[-2:] != ("y", "x"):  # So it's a constant
        if original2d is None:
            raise ValueError("original2d must be provided for constant values")
        source = source * xr.ones_like(original2d)
    ugrid_source = xu.UgridDataArray.from_structured(source)
    regridder = xu.OverlapRegridder(ugrid_source, target=target, method=method)
    return regridder.regrid(ugrid_source)


def create_idomain(active):
    """
    Find cells that should get a passthrough value: IDOMAIN = -1.

    We may find them by forward and back-filling: if they are filled in both, they
    contain active cells in both directions, and the value should be set to -1.
    """
    ones = xu.ones_like(active, dtype=float).where(active)
    passthrough = ones.ffill("layer").notnull() & ones.bfill("layer").notnull()
    idomain = ones.combine_first(
        xu.full_like(active, -1.0, dtype=float).where(passthrough)
    )
    return idomain.fillna(0).astype(int)


def create_disv(
    top,
    bottom,
    target,
):
    disv_top = regrid(top, target, method="mean")
    disv_bottom = regrid(bottom, target, method="mean")
    thickness = disv_top - disv_bottom
    active = thickness > 0  # TODO larger than a specific value for round off?
    idomain = create_idomain(active)
    disv = imod.mf6.VerticesDiscretization(
        top=disv_top.sel(layer=1),
        bottom=disv_bottom,
        idomain=idomain,
    )
    return disv, disv_top, disv_bottom, active


def create_npf(
    k,
    k33,
    target,
    active,
    original2d,
):
    disv_k = regrid(k, target, method="geometric_mean", original2d=original2d).where(
        active
    )
    disv_k33 = regrid(k33, target, method="harmonic_mean", original2d=original2d).where(
        active
    )
    return imod.mf6.NodePropertyFlow(
        icelltype=0,
        k=disv_k,
        k33=disv_k33,
    )


def create_chd(
    model,
    key,
    value,
    target,
    ibound,
    active,
    original2d,
    **kwargs,
):
    is_constant_head = regrid(source=ibound, target=target, method="minimum") < 0
    head = value["head"]
    head = xu.UgridDataArray.from_structured(head)
    regridder = xu.BarycentricInterpolator(source=head, target=target)
    constant_head = regridder.regrid(head).where(active & is_constant_head)
    model[key] = imod.mf6.ConstantHead(head=constant_head)
    return


def create_drn(
    model,
    key,
    value,
    target,
    active,
    top,
    bottom,
    original2d,
    **kwargs,
):
    conductance = value["conductance"]
    elevation = value["elevation"]

    disv_cond = regrid(conductance, target, method="conductance", original2d=original2d)
    disv_elev = regrid(elevation, target, method="mean", original2d=original2d)
    location = xu.ones_like(active, dtype=float)
    location = location.where((disv_elev >= bottom) & (disv_elev < top)).where(active)
    disv_cond = (location * disv_cond).dropna("layer", how="all")
    disv_elev = (location * disv_elev).dropna("layer", how="all")
    model[key] = imod.mf6.Drainage(
        elevation=disv_elev,
        conductance=disv_cond,
    )
    return


def create_riv(
    model,
    key,
    value,
    target,
    active,
    original2d,
    top,
    bottom,
    **kwargs,
):
    conductance = value["conductance"]
    stage = value["stage"]
    bottom_elevation = value["bottom_elevation"]
    infiltration_factor = value["infiltration_factor"]

    disv_cond = regrid(conductance, target, method="conductance", original2d=original2d)
    disv_elev = regrid(bottom_elevation, target, method="mean", original2d=original2d)
    disv_stage = regrid(stage, target, method="mean", original2d=original2d)
    # River may contain values where the stage < bottom.
    disv_elev = disv_elev.where(disv_stage >= disv_elev, other=disv_stage)
    disv_inff = regrid(
        infiltration_factor, target, method="mean", original2d=original2d
    )

    location = xu.ones_like(active, dtype=float)
    location = location.where((disv_elev >= bottom) & (disv_elev < top)).where(active)
    disv_cond = (location * disv_cond).dropna("layer", how="all")
    disv_elev = (location * disv_elev).dropna("layer", how="all")
    disv_stage = (location * disv_stage).dropna("layer", how="all")
    disv_inff = (location * disv_inff).dropna("layer", how="all")

    riv = imod.mf6.River(
        stage=disv_stage,
        conductance=disv_cond * disv_inff,
        bottom_elevation=disv_elev,
    )
    drn = imod.mf6.Drainage(
        conductance=(1.0 - disv_inff) * disv_cond,
        elevation=disv_stage,
    )
    model[key] = riv
    model[f"{key}-drn"] = drn
    return


def create_rch(
    model,
    key,
    value,
    target,
    active,
    original2d,
    **kwargs,
):
    rate = value["rate"] * 0.001
    disv_rate = regrid(rate, target, method="mean", original2d=original2d).where(active)
    model[key] = imod.mf6.Recharge(rate=disv_rate)
    return


def create_wel(
    model,
    key,
    value,
    target,
    **kwargs,
):
    dataframe = value["dataframe"]
    layer = value["layer"]

    columns = dataframe.columns
    x, y, rate = columns[:3]
    xy = np.column_stack([dataframe[x], dataframe[y]])
    cell2d = target.locate_points(xy)
    model[key] = imod.mf6.WellDisVertices(
        layer=np.full_like(cell2d, layer),
        cell2d=cell2d,
        rate=dataframe[rate],
    )
    return


def create_ic(model, key, value, target, active, **kwargs):
    start = value["head"]
    start = xu.UgridDataArray.from_structured(start)
    regridder = xu.BarycentricInterpolator(source=start, target=target)
    disv_start = regridder.regrid(start).where(active)
    model[key] = imod.mf6.InitialConditions(start=disv_start)
    return


def create_hfb(model, key, value, active, target, **kwargs):
    data = value["geodataframe"]
    layer = value["layer"]

    polygons = pygeos.from_shapely(data.geometry)
    coordinates, index = pygeos.get_coordinates(polygons, return_index=True)
    df = pd.DataFrame({"index": index, "x": coordinates[:, 0], "y": coordinates[:, 1]})
    df = df.drop_duplicates().reset_index(drop=True)
    indices = np.repeat(np.arange(len(df) // 2), 2)
    linestrings = pygeos.linestrings(df["x"], y=df["y"], indices=indices)
    lines = gpd.GeoDataFrame(geometry=linestrings)

    if "resistance" in data:
        lines["resistance"] = data["resistance"]
    elif "multiplier" in data:
        lines["resistance"] = -1.0 * data["resistance"]
    else:
        raise ValueError(
            "Expected resistance or multiplier in HFB data, "
            f"received instead: {list(data.keys())}"
        )
    snapped, _ = xu.snap_to_grid(lines, grid=target, max_snap_distance=0.5)

    resistance = snapped["resistance"]
    if layer != 0:
        resistance = resistance.assign_coords(layer=layer)

    model[key] = imod.mf6.HorizontalFlowBarrier(
        resistance=resistance,
        idomain=active.astype(int),
    )
    return


def merge_hfbs(hfbs):
    c_per_layer = defaultdict(list)
    idomain = hfbs[0]["idomain"]
    for hfb in hfbs:
        resistance = hfb["resistance"].expand_dims("layer")
        for layer, da in resistance.groupby("layer"):
            c_per_layer[layer].append(da)

    merged_c = xr.concat([xr.merge(v) for v in c_per_layer.values()], dim="layer")
    merged_c = xu.UgridDataArray(
        merged_c["resistance"], grid=hfbs[0].dataset.ugrid.grid
    )
    return imod.mf6.HorizontalFlowBarrier(resistance=merged_c, idomain=idomain)


PKG_CONVERSION = {
    "chd": create_chd,
    "drn": create_drn,
    "hfb": create_hfb,
    "shd": create_ic,
    "rch": create_rch,
    "riv": create_riv,
    "wel": create_wel,
}


def convert_to_disv(projectfile_data, target):
    data = projectfile_data.copy()
    model = imod.mf6.GroundwaterFlowModel()

    # Mandatory packages first.
    disv, top, bottom, active = create_disv(
        top=data["top"]["top"],
        bottom=data["bot"]["bottom"],
        target=target,
    )
    original2d = data["bot"]["bottom"].sel(layer=1)
    npf = create_npf(
        k=data["khv"]["kh"],
        k33=data["kvv"]["kv"],
        target=target,
        active=active,
        original2d=original2d,
    )
    model["npf"] = npf
    model["disv"] = disv

    ibound = data["bnd"]["ibound"]
    # Boundary conditions, one by one.
    for key, value in data.items():
        print(key)
        pkg = key.split("-")[0]
        try:
            # conversion will update model
            conversion = PKG_CONVERSION[pkg]

            if pkg == "chd":
                continue

            try:
                conversion(
                    model=model,
                    key=key,
                    value=value,
                    target=target,
                    ibound=ibound,
                    active=active,
                    original2d=original2d,
                    top=top,
                    bottom=bottom,
                )
            except Exception as e:
                raise type(e)(f"{e}\nduring conversion of {key}")

        except KeyError:
            pass

    # Treat hfb's separately: they must be merged into one,
    # as MODFLOW6 only supports a single HFB.
    hfb_keys = [key for key in model.keys() if key.split("-")[0] == "hfb"]
    hfbs = [model.pop(key) for key in hfb_keys]
    if hfbs:
        model["hfb"] = merge_hfbs(hfbs)

    return model
