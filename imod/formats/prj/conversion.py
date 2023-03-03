import pickle
from collections import Counter, defaultdict
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xugrid as xu

import imod


def xy_hash(da: xr.DataArray) -> Tuple[int]:
    """
    Create a unique identifier based on the x and y coordinates of a DataArray.
    """
    x = hash(pickle.dumps(da["x"].values))
    y = hash(pickle.dumps(da["y"].values))
    return x, y


class SingularTargetRegridderWeightsCache:
    """
    Create a mapping of (source_coords, regridder_cls) => regridding weights.

    Allows re-use of the regridding weights, as computing the weights is the
    most costly step.

    The regridder only processes x and y coordinates: we can hash these,
    and get a unique identifier. The target is assumed to be constant.
    """

    def __init__(self, projectfile_data, target, cache_size: int):
        # Collect the x-y coordinates of all x-y dimensioned DataArrays.
        # Determine which regridding method to use.
        # Count occurrences of both.
        # Create and cache weights of the most common ones.
        keys = []
        sources = {}
        methods = {}
        for pkgdict in projectfile_data.values():
            for variable, da in pkgdict.items():
                xydims = set(("x", "y"))

                if isinstance(da, xr.DataArray) and xydims.issubset(da.dims):
                    if variable == "head":
                        cls = xu.BarycentricInterpolator
                        method = None
                    elif variable == "conductance":
                        cls = xu.RelativeOverlapRegridder
                        method = "conductance"
                    else:
                        cls = xu.OverlapRegridder
                        method = "mean"

                    x, y = xy_hash(da)
                    key = (x, y, cls)
                    keys.append(key)
                    sources[key] = da
                    methods[key] = method

        counter = Counter(keys)
        self.target = target
        self.weights = {}
        for key, _ in counter.most_common(cache_size):
            cls = key[2]
            ugrid_source = xu.UgridDataArray.from_structured(sources[key])
            kwargs = {"source": ugrid_source, "target": target}
            method = methods[key]
            if method is not None:
                kwargs["method"] = method
            regridder = cls(**kwargs)
            self.weights[key] = regridder.regrid_weights

    def regrid(
        self,
        source: xr.DataArray,
        method: str = "mean",
        original2d: Optional[xr.DataArray] = None,
    ):
        if source.dims[-2:] != ("y", "x"):  # So it's a constant
            if original2d is None:
                raise ValueError("original2d must be provided for constant values")
            source = source * xr.ones_like(original2d)

        ugrid_source = xu.UgridDataArray.from_structured(source)
        kwargs = {"source": ugrid_source, "target": self.target}
        if method == "barycentric":
            cls = xu.BarycentricInterpolator
        elif method == "conductance":
            cls = xu.RelativeOverlapRegridder
            kwargs["method"] = method
        else:
            cls = xu.OverlapRegridder
            kwargs["method"] = method

        x, y = xy_hash(source)
        key = (x, y, cls)
        if key in self.weights:
            weights = self.weights[key]
            kwargs["weights"] = weights
            regridder = cls(**kwargs)
        else:
            regridder = cls(**kwargs)
            return regridder.regrid(source)

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
    cache,
    top,
    bottom,
):
    top = top.compute()
    disv_top = cache.regrid(top)
    disv_bottom = cache.regrid(bottom)
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
    cache,
    k,
    vertical_anisotropy,
    active,
    original2d,
):
    disv_k = cache.regrid(k, method="geometric_mean", original2d=original2d).where(
        active
    )
    k33 = k * vertical_anisotropy
    disv_k33 = cache.regrid(k33, method="harmonic_mean", original2d=original2d).where(
        active
    )
    return imod.mf6.NodePropertyFlow(
        icelltype=0,
        k=disv_k,
        k33=disv_k33,
    )


def create_chd(
    cache,
    model,
    key,
    value,
    ibound,
    active,
    original2d,
    **kwargs,
):
    head = value["head"]

    if "layer" in head.coords:
        layer = head.layer
        ibound = ibound.sel(layer=layer)
        active = active.sel(layer=layer)

    is_constant_head = ibound < 0
    constant_head = cache.regrid(
        head,
        method="barycentric",
        original2d=original2d,
    ).where(active & is_constant_head)
    model[key] = imod.mf6.ConstantHead(head=constant_head)
    return


def create_drn(
    cache,
    model,
    key,
    value,
    active,
    top,
    bottom,
    original2d,
    **kwargs,
):
    conductance = value["conductance"]
    elevation = value["elevation"]

    disv_cond = cache.regrid(
        conductance,
        method="conductance",
        original2d=original2d,
    )
    disv_elev = cache.regrid(elevation, original2d=original2d)
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
    cache,
    model,
    key,
    value,
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

    disv_cond = cache.regrid(
        conductance,
        method="conductance",
        original2d=original2d,
    )
    disv_elev = cache.regrid(bottom_elevation, original2d=original2d)
    disv_stage = cache.regrid(stage, original2d=original2d)
    # River may contain values where the stage < bottom.
    disv_elev = disv_elev.where(disv_stage >= disv_elev, other=disv_stage)
    disv_inff = cache.regrid(infiltration_factor, original2d=original2d)

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
    cache,
    model,
    key,
    value,
    active,
    original2d,
    **kwargs,
):
    rate = value["rate"] * 0.001
    disv_rate = cache.regrid(rate, original2d=original2d).where(active)
    model[key] = imod.mf6.Recharge(rate=disv_rate)
    return


def create_wel(
    cache,
    model,
    key,
    value,
    **kwargs,
):
    target = cache.target
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


def create_ic(cache, model, key, value, active, **kwargs):
    start = value["head"]
    disv_start = cache.regrid(source=start, method="barycentric").where(active)
    model[key] = imod.mf6.InitialConditions(start=disv_start)
    return


def create_hfb(cache, model, key, value, active, **kwargs):
    target = cache.target
    data = value["geodataframe"]
    layer = value["layer"]

    coordinates, index = shapely.get_coordinates(data.geometry, return_index=True)
    df = pd.DataFrame({"index": index, "x": coordinates[:, 0], "y": coordinates[:, 1]})
    df = df.drop_duplicates().reset_index(drop=True)
    indices = np.repeat(np.arange(len(df) // 2), 2)
    linestrings = shapely.linestrings(df["x"], y=df["y"], indices=indices)
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

    # Setup the regridding weights cache.
    weights_cache = SingularTargetRegridderWeightsCache(data, target, cache_size=5)

    # Mandatory packages first.
    disv, top, bottom, active = create_disv(
        cache=weights_cache,
        top=data["top"]["top"],
        bottom=data["bot"]["bottom"],
    )
    original2d = data["bot"]["bottom"].sel(layer=1)

    npf = create_npf(
        cache=weights_cache,
        k=data["khv"]["kh"],
        vertical_anisotropy=data["kva"]["vertical_anisotropy"],
        active=active,
        original2d=original2d,
    )
    model["npf"] = npf
    model["disv"] = disv

    ibound = data["bnd"]["ibound"]
    new_ibound = weights_cache.regrid(source=ibound, method="minimum")

    # Boundary conditions, one by one.
    for key, value in data.items():
        pkg = key.split("-")[0]
        try:
            # conversion will update model
            conversion = PKG_CONVERSION[pkg]

            if pkg == "hfb":
                continue

            try:
                conversion(
                    cache=weights_cache,
                    model=model,
                    key=key,
                    value=value,
                    ibound=new_ibound,
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
