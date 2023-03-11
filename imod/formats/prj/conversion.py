import itertools
import pickle
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xugrid as xu

import imod


def vectorized_overlap(bounds_a, bounds_b):
    """
    Vectorized overlap computation.

    Compare with:

    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    """
    return np.maximum(
        0.0,
        np.minimum(bounds_a[..., 1], bounds_b[..., 1])
        - np.maximum(bounds_a[..., 0], bounds_b[..., 0]),
    )


def hash_xy(da: xr.DataArray) -> Tuple[int]:
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
                    # for initial condition, constant head, general head boundary
                    if variable == "head":
                        cls = xu.BarycentricInterpolator
                        method = None
                    elif variable == "conductance":
                        cls = xu.RelativeOverlapRegridder
                        method = "conductance"
                    else:
                        cls = xu.OverlapRegridder
                        method = "mean"

                    x, y = hash_xy(da)
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
            self.weights[key] = regridder.weights

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

        kwargs = {"target": self.target}
        if method == "barycentric":
            cls = xu.BarycentricInterpolator
        elif method == "conductance":
            cls = xu.RelativeOverlapRegridder
            kwargs["method"] = method
        else:
            cls = xu.OverlapRegridder
            kwargs["method"] = method

        x, y = hash_xy(source)
        key = (x, y, cls)
        if key in self.weights:
            kwargs["weights"] = self.weights[key]
            regridder = cls.from_weights(**kwargs)
            # Avoid creation of a UgridDataArray here
            dims = source.dims[:-2]
            coords = {k: source.coords[k] for k in dims}
            facedim = "mesh2d_nFace"
            face_source = xr.DataArray(
                source.data.reshape(*source.shape[:-2], -1),
                coords=coords,
                dims=[*dims, facedim],
                name=source.name,
            )
            return xu.UgridDataArray(
                regridder.regrid_dataarray(face_source, (facedim,)),
                regridder._target.ugrid_topology,
            )
        else:
            ugrid_source = xu.UgridDataArray.from_structured(source)
            kwargs["source"] = ugrid_source
            regridder = cls(**kwargs)
            return regridder.regrid(ugrid_source)


def create_idomain(thickness):
    """
    Find cells that should get a passthrough value: IDOMAIN = -1.

    We may find them by forward and back-filling: if they are filled in both, they
    contain active cells in both directions, and the value should be set to -1.
    """
    active = (
        thickness > 0
    ).compute()  # TODO larger than a specific value for round off?
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
    disv_top = cache.regrid(top).compute()
    disv_bottom = cache.regrid(bottom).compute()
    thickness = disv_top - disv_bottom
    idomain = create_idomain(thickness)
    disv = imod.mf6.VerticesDiscretization(
        top=disv_top.sel(layer=1),
        bottom=disv_bottom,
        idomain=idomain,
    )
    active = idomain > 0
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
    repeat,
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

    chd = imod.mf6.ConstantHead(head=constant_head)
    if repeat is not None:
        chd.set_repeat_stress(repeat)
    model[key] = chd
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
    repeat,
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

    drn = imod.mf6.Drainage(
        elevation=disv_elev,
        conductance=disv_cond,
    )
    if repeat is not None:
        model[key] = drn.set_repeat_stress(repeat)
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
    repeat,
    **kwargs,
):
    def finish(uda):
        """
        Set dimension order, and drop empty layers.
        """
        facedim = uda.ugrid.grid.face_dimension
        if "time" in uda.dims:
            dims = ("time", "layer", facedim)
        else:
            dims = ("layer", facedim)
        return uda.transpose(*dims).dropna("layer", how="all")

    def assign_to_layer(
        conductance, stage, elevation, infiltration_factor, top, bottom, active
    ):
        """
        Assign river boundary to multiple layers. Distribute the conductance based
        on the vertical degree of overlap.

        Parameters
        ----------
        conductance
        stage:
            water stage
        elevation:
            bottom elevation of the river
        infiltration_factor
            factor (generally <1) to reduce infiltration conductance compared
            to drainage conductance.
        top:
            layer model top elevation
        bottom:
            layer model bottom elevation
        active:
            active or inactive cells (idomain > 0)
        """
        valid = conductance > 0.0
        conductance = conductance.where(valid)
        stage = stage.where(valid)
        elevation = elevation.where(valid)
        elevation = elevation.where(elevation <= stage, other=stage)

        water_top = stage.where(stage <= top)
        water_bottom = elevation.where(elevation > bottom)
        layer_height = top - bottom
        layer_height = layer_height.where(active)  # avoid 0 thickness layers
        fraction = (water_top - water_bottom) / layer_height
        # Set values of 0.0 to 1.0, but do not change NaN values:
        fraction = fraction.where(~(fraction == 0.0), other=1.0)
        location = xu.ones_like(fraction).where(fraction.notnull() & active)

        layered_conductance = finish(conductance * fraction)
        layered_stage = finish(stage * location)
        layered_elevation = finish(elevation * location)
        infiltration_factor = finish(infiltration_factor * location)

        return (
            layered_conductance,
            layered_stage,
            layered_elevation,
            infiltration_factor,
        )

    conductance = value["conductance"]
    stage = value["stage"]
    bottom_elevation = value["bottom_elevation"]
    infiltration_factor = value["infiltration_factor"]

    disv_cond_2d = cache.regrid(
        conductance,
        method="conductance",
        original2d=original2d,
    )
    disv_elev_2d = cache.regrid(bottom_elevation, original2d=original2d)
    disv_stage_2d = cache.regrid(stage, original2d=original2d)
    disv_inff_2d = cache.regrid(infiltration_factor, original2d=original2d)

    disv_cond, disv_stage, disv_elev, disv_inff = assign_to_layer(
        conductance=disv_cond_2d,
        stage=disv_stage_2d,
        elevation=disv_elev_2d,
        infiltration_factor=disv_inff_2d,
        top=top,
        bottom=bottom,
        active=active,
    )

    riv = imod.mf6.River(
        stage=disv_stage,
        conductance=disv_cond * disv_inff,
        bottom_elevation=disv_elev,
    )
    drn = imod.mf6.Drainage(
        conductance=(1.0 - disv_inff) * disv_cond,
        elevation=disv_stage,
    )

    if repeat is not None:
        riv.set_repeat_stress(repeat)
        drn.set_repeat_stress(repeat)

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
    repeat,
    **kwargs,
):
    rate = value["rate"] * 0.001
    disv_rate = cache.regrid(rate, original2d=original2d).where(active)
    rch = imod.mf6.Recharge(rate=disv_rate)
    if repeat is not None:
        rch.set_repeat_stress(repeat)
    model[key] = rch
    return


def create_wel(
    cache,
    model,
    key,
    value,
    repeat,
    **kwargs,
):
    target = cache.target
    dataframe = value["dataframe"]
    layer = value["layer"]

    columns = dataframe.columns
    x, y, rate = columns[:3]
    xy = np.column_stack([dataframe[x], dataframe[y]])
    cell2d = target.locate_points(xy)

    wel = imod.mf6.WellDisVertices(
        layer=np.full_like(cell2d, layer),
        cell2d=cell2d,
        rate=dataframe[rate],
    )
    if repeat is not None:
        wel.set_repeat_stress(repeat)
    model[key] = wel
    return


def create_ic(cache, model, key, value, active, **kwargs):
    start = value["head"]
    disv_start = cache.regrid(source=start, method="barycentric").where(active)
    model[key] = imod.mf6.InitialConditions(start=disv_start)
    return


def multi_layer_hfb(
    snapped: xu.UgridDataset,
    dataframe: gpd.GeoDataFrame,
    top: xu.UgridDataArray,
    bottom: xu.UgridDataArray,
    k: xu.UgridDataArray,
) -> xu.UgridDataArray:
    """
    Assign horizontal flow barriers to layers.

    Reduce the effective resistance by fraction of overlap with the layer
    thickness.
    """

    def mean_left_and_right(uda, left, right):
        facedim = uda.ugrid.grid.face_dimension
        uda_left = uda.ugrid.obj.isel({facedim: left}).drop_vars(facedim)
        uda_right = uda.ugrid.obj.isel({facedim: right}).drop_vars(facedim)
        return xr.concat((uda_left, uda_right), dim="two").mean("two")

    def extract_dataframe_data(snapped, dataframe):
        line_index = snapped["line_index"].values
        line_index = line_index[~np.isnan(line_index)].astype(int)
        sample = dataframe.iloc[line_index]
        coordinates, index = shapely.get_coordinates(
            sample.geometry, include_z=True, return_index=True
        )
        grouped = pd.DataFrame({"index": index, "z": coordinates[:, 2]}).groupby(
            "index"
        )
        zmin = grouped["z"].min().values
        zmax = grouped["z"].max().values
        return zmin, zmax, sample["resistance"].values

    def effective_resistance(snapped, dataframe, top, bottom, k):
        left, right = snapped.ugrid.grid.edge_face_connectivity[edge_index].T
        top_mean = mean_left_and_right(top, left, right)
        bottom_mean = mean_left_and_right(bottom, left, right)
        k_mean = mean_left_and_right(k, left, right)

        n_layer, n_edge = top_mean.shape
        layer_bounds = np.empty((n_edge, n_layer, 2), dtype=float)
        layer_bounds[..., 0] = bottom_mean.values.T
        layer_bounds[..., 1] = top_mean.values.T

        zmin, zmax, resistance = extract_dataframe_data(snapped, dataframe)
        hfb_bounds = np.empty((n_edge, n_layer, 2), dtype=float)
        hfb_bounds[..., 0] = zmin[:, np.newaxis]
        hfb_bounds[..., 1] = zmax[:, np.newaxis]

        overlap = vectorized_overlap(hfb_bounds, layer_bounds)
        height = layer_bounds[..., 1] - layer_bounds[..., 0]
        # Avoid runtime warnings when diving by 0:
        height[height <= 0] = np.nan
        fraction = (overlap / height).T

        resistance = resistance[np.newaxis, :]
        c_aquifer = 1.0 / k_mean
        inverse_c = (fraction / resistance) + ((1.0 - fraction) / c_aquifer)
        c_total = 1.0 / inverse_c
        return c_total

    edge_index = np.argwhere(snapped["resistance"].notnull().values).ravel()
    c_total = effective_resistance(snapped, dataframe, top, bottom, k)

    notnull = c_total.notnull().values
    layer, edge_subset = np.nonzero(notnull)
    edge_subset_index = edge_index[edge_subset]
    resistance_layered = (
        xr.ones_like(top["layer"]) * xu.full_like(snapped["resistance"], np.nan)
    ).compute()
    resistance_layered.values[layer, edge_subset_index] = c_total.values[notnull]
    resistance_layered = resistance_layered.dropna("layer", how="all")

    return resistance_layered


def create_hfb(cache, model, key, value, active, top, bottom, k, **kwargs):
    target = cache.target
    dataframe = value["geodataframe"]
    layer = value["layer"]

    coordinates, index = shapely.get_coordinates(dataframe.geometry, return_index=True)
    df = pd.DataFrame({"index": index, "x": coordinates[:, 0], "y": coordinates[:, 1]})
    df = df.drop_duplicates().reset_index(drop=True)
    indices = np.repeat(np.arange(len(df) // 2), 2)
    linestrings = shapely.linestrings(df["x"], y=df["y"], indices=indices)
    lines = gpd.GeoDataFrame(geometry=linestrings)

    if "resistance" in dataframe:
        lines["resistance"] = dataframe["resistance"]
    elif "multiplier" in dataframe:
        lines["resistance"] = -1.0 * dataframe["resistance"]
    else:
        raise ValueError(
            "Expected resistance or multiplier in HFB dataframe, "
            f"received instead: {list(dataframe.keys())}"
        )
    snapped, _ = xu.snap_to_grid(lines, grid=target, max_snap_distance=0.5)

    resistance = snapped["resistance"]
    if layer != 0:
        resistance = resistance.assign_coords(layer=layer)
    else:
        resistance = multi_layer_hfb(snapped, dataframe, top, bottom, k)

    model[key] = imod.mf6.HorizontalFlowBarrierResistance(
        resistance=resistance,
        idomain=active.astype(int),
    )
    return


def merge_hfbs(hfbs, idomain):
    first = hfbs[0]
    grid = first.dataset.ugrid.grid
    layer = idomain["layer"].values
    n_layer = layer.size
    n_edge = grid.n_edge
    c_merged = xr.DataArray(
        data=np.zeros((n_layer, n_edge), dtype=float),
        dims=("layer", grid.edge_dimension),
        coords={"layer": layer},
    )

    for hfb in hfbs:
        resistance = hfb["resistance"].fillna(0.0)
        if "layer" not in resistance.dims:
            resistance = resistance.expand_dims("layer")
        for layer, da in resistance.groupby("layer"):
            c_merged.values[layer - 1, :] += da.values

    final_c = xu.UgridDataArray(
        c_merged.where(c_merged > 0).dropna("layer", how="all"),
        grid,
    )
    return imod.mf6.HorizontalFlowBarrierResistance(resistance=final_c, idomain=idomain)


PKG_CONVERSION = {
    "chd": create_chd,
    "drn": create_drn,
    "hfb": create_hfb,
    "shd": create_ic,
    "rch": create_rch,
    "riv": create_riv,
    "wel": create_wel,
}


def expand_repetitions(
    repeat_stress: List[datetime], times: List[datetime]
) -> Dict[datetime, datetime]:
    first = times[0]
    last = times[-1]
    expanded = {}
    for date, year in itertools.product(
        range(first.year, last.year + 1), repeat_stress
    ):
        newdate = date.replace(year=year)
        if newdate < last:
            expanded[newdate] = date
    return expanded


def convert_to_disv(projectfile_data, target, repeat_stress=None, times=None):
    if times is None:
        if repeat_stress is not None:
            raise ValueError("times is required when repeat_stress is given")
    else:
        times = sorted(times)

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
    idomain = disv["idomain"].compute()
    k = npf["k"].compute()
    model["npf"] = npf
    model["disv"] = disv

    ibound = data["bnd"]["ibound"]
    new_ibound = weights_cache.regrid(source=ibound, method="minimum")

    # Boundary conditions, one by one.
    for key, value in data.items():
        pkg = key.split("-")[0]
        try:
            # conversion will update model
            convert = PKG_CONVERSION[pkg]
            if repeat_stress is None:
                repeat = None
            else:
                repeat = repeat_stress.get(key)
                if repeat is not None:
                    repeat = expand_repetitions(repeat, times)

            try:
                convert(
                    cache=weights_cache,
                    model=model,
                    key=key,
                    value=value,
                    ibound=new_ibound,
                    active=active,
                    original2d=original2d,
                    top=top,
                    bottom=bottom,
                    k=k,
                    repeat=repeat,
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
        model["hfb"] = merge_hfbs(hfbs, idomain)

    return model
