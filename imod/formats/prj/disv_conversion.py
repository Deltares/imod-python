"""
Most of the functionality here attempts to replicate what iMOD does with
project files.
"""
import itertools
import pickle
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu

import imod

try:
    import geopandas as gpd
except ImportError:
    gpd = imod.util.MissingOptionalModule("geopandas")

try:
    import shapely
except ImportError:
    shapely = imod.util.MissingOptionalModule("shapely")


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
            facedim = self.target.face_dimension
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


def raise_on_layer(value, variable: str):
    da = value[variable]
    if "layer" in da.dims:
        raise ValueError(f"{variable} should not be assigned a layer")
    return da


def finish(uda):
    """
    Set dimension order, and drop empty layers.
    """
    facedim = uda.ugrid.grid.face_dimension
    dims = ("time", "layer", facedim)
    return uda.transpose(*dims, missing_dims="ignore").dropna("layer", how="all")


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
    ibound,
):
    if top.dims == ("layer",):
        if ibound.dims != ("layer", "y", "x"):
            raise ValueError(
                "Either ibound or top must have dimensions (layer, y, x) to "
                "derive model extent. Both may not be provided as constants."
            )
        top = top * xr.ones_like(ibound)
        original2d = ibound.isel(layer=0, drop=True)
    else:
        original2d = top.isel(layer=0, drop=True)

    if bottom.dims == ("layer",):
        bottom = bottom * xr.ones_like(ibound)

    top = top.compute()
    bottom = bottom.compute()
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
    return disv, disv_top, disv_bottom, active, original2d


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

    disv_head = cache.regrid(
        head,
        method="barycentric",
        original2d=original2d,
    )
    valid = (ibound < 0) & active

    if not valid.any():
        return

    chd = imod.mf6.ConstantHead(head=disv_head.where(valid))
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
    conductance = raise_on_layer(value, "conductance")
    elevation = raise_on_layer(value, "elevation")

    disv_cond = cache.regrid(
        conductance,
        method="conductance",
        original2d=original2d,
    )
    disv_elev = cache.regrid(elevation, original2d=original2d)
    valid = (disv_cond > 0) & disv_elev.notnull() & active
    location = xu.ones_like(active, dtype=float)
    location = location.where((disv_elev > bottom) & (disv_elev <= top)).where(valid)
    disv_cond = finish(location * disv_cond)
    disv_elev = finish(location * disv_elev)

    if disv_cond.isnull().all():
        return

    drn = imod.mf6.Drainage(
        elevation=disv_elev,
        conductance=disv_cond,
    )
    if repeat is not None:
        drn.set_repeat_stress(repeat)
    model[key] = drn
    return


def create_ghb(
    cache,
    model,
    key,
    value,
    active,
    original2d,
    repeat,
    **kwargs,
):
    conductance = value["conductance"]
    head = value["head"]

    disv_cond = cache.regrid(
        conductance,
        method="conductance",
        original2d=original2d,
    )
    disv_head = cache.regrid(
        head,
        method="barycentric",
        original2d=original2d,
    )
    valid = (disv_cond > 0.0) & disv_head.notnull() & active

    ghb = imod.mf6.GeneralHeadBoundary(
        conductance=disv_cond.where(valid),
        head=disv_head.where(valid),
    )
    if repeat is not None:
        ghb.set_repeat_stress(repeat)
    model[key] = ghb
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

        # TODO: this removes too much when the stage is higher than the top...
        # Instead: just cut through all layers until the bottom elevation.
        # Then, assign a transmissivity weighted conductance.
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

    conductance = raise_on_layer(value, "conductance")
    stage = raise_on_layer(value, "stage")
    bottom_elevation = raise_on_layer(value, "bottom_elevation")
    infiltration_factor = raise_on_layer(value, "infiltration_factor")

    disv_cond_2d = cache.regrid(
        conductance,
        method="conductance",
        original2d=original2d,
    ).compute()
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

    if disv_cond.isnull().all():
        return

    # The infiltration factor may be 0. In that case, we need only create a DRN
    # package.
    drn = imod.mf6.Drainage(
        conductance=(1.0 - disv_inff) * disv_cond,
        elevation=disv_stage,
    )
    if repeat is not None:
        drn.set_repeat_stress(repeat)
    model[f"{key}-drn"] = drn

    riv_cond = disv_cond * disv_inff
    riv_valid = riv_cond > 0.0
    if not riv_valid.any():
        return

    riv = imod.mf6.River(
        stage=disv_stage.where(riv_valid),
        conductance=riv_cond.where(riv_valid),
        bottom_elevation=disv_elev.where(riv_valid),
    )
    if repeat is not None:
        riv.set_repeat_stress(repeat)
    model[key] = riv

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
    rate = raise_on_layer(value, "rate") * 0.001
    disv_rate = cache.regrid(rate, original2d=original2d).where(active)
    # Find highest active layer
    highest = active["layer"] == active["layer"].where(active).min()
    location = highest.where(highest)
    disv_rate = finish(disv_rate * location)

    # Skip if there's no data
    if disv_rate.isnull().all():
        return

    rch = imod.mf6.Recharge(rate=disv_rate)
    if repeat is not None:
        rch.set_repeat_stress(repeat)
    model[key] = rch
    return


def create_evt(
    cache,
    model,
    key,
    value,
    active,
    original2d,
    repeat,
    **kwargs,
):
    surface = raise_on_layer(value, "surface")
    rate = raise_on_layer(value, "rate") * 0.001
    depth = raise_on_layer(value, "depth")

    # Find highest active layer
    highest = active["layer"] == active["layer"].where(active).min()
    location = highest.where(highest)

    disv_surface = cache.regrid(surface, original2d=original2d).where(active)
    disv_surface = finish(disv_surface * location)

    disv_rate = cache.regrid(rate, original2d=original2d).where(active)
    disv_rate = finish(disv_rate * location)

    disv_depth = cache.regrid(depth, original2d=original2d).where(active)
    disv_depth = finish(disv_depth * location)

    # At depth 1.0, the rate is 0.0.
    proportion_depth = xu.ones_like(disv_surface).where(disv_surface.notnull())
    proportion_rate = xu.zeros_like(disv_surface).where(disv_surface.notnull())

    evt = imod.mf6.Evapotranspiration(
        surface=disv_surface,
        rate=disv_rate,
        depth=disv_depth,
        proportion_rate=proportion_rate,
        proportion_depth=proportion_depth,
    )
    if repeat is not None:
        evt.set_repeat_stress(repeat)
    model[key] = evt
    return


def create_sto(
    cache,
    storage_coefficient,
    active,
    original2d,
    transient,
):
    if storage_coefficient is None:
        disv_coef = 0.0
    else:
        disv_coef = cache.regrid(storage_coefficient, original2d=original2d).where(
            active
        )

    sto = imod.mf6.StorageCoefficient(
        storage_coefficient=disv_coef,
        specific_yield=0.0,
        transient=transient,
        convertible=0,
    )
    return sto


def create_wel(
    cache,
    model,
    key,
    value,
    active,
    top,
    bottom,
    k,
    repeat,
    **kwargs,
):
    target = cache.target
    dataframe = value["dataframe"]
    layer = value["layer"]

    if layer <= 0:
        dataframe = imod.prepare.assign_wells(
            wells=dataframe,
            top=top,
            bottom=bottom,
            k=k,
            minimum_thickness=0.01,
            minimum_k=1.0,
        )
    else:
        dataframe["index"] = np.arange(len(dataframe))
        dataframe["layer"] = layer

    first = dataframe.groupby("index").first()
    well_layer = first["layer"].values
    xy = np.column_stack([first["x"], first["y"]])
    cell2d = target.locate_points(xy)
    valid = (cell2d >= 0) & active.values[well_layer - 1, cell2d]

    cell2d = cell2d[valid] + 1
    # Skip if no wells are located inside cells
    if not valid.any():
        return

    if "time" in dataframe.columns:
        # Ensure the well data is rectangular.
        time = np.unique(dataframe["time"].values)
        dataframe = dataframe.set_index("time")
        # First ffill, then bfill!
        dfs = [df.reindex(time).ffill().bfill() for _, df in dataframe.groupby("index")]
        rate = (
            pd.concat(dfs)
            .reset_index()
            .set_index(["time", "index"])["rate"]
            .to_xarray()
        )
    else:
        rate = xr.DataArray(
            dataframe["rate"], coords={"index": dataframe["index"]}, dims=["index"]
        )

    # Don't forget to remove the out-of-bounds points.
    rate = rate.where(xr.DataArray(valid, dims=["index"]), drop=True)

    wel = imod.mf6.WellDisVertices(
        layer=well_layer,
        cell2d=cell2d,
        rate=rate,
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
    dataframe: "gpd.GeoDataFrame",
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


def __start_end_pairs_segments(arr):
    """
    Returns 2d array with starts and ends of each segment.

    Input:
    [1, 2, 3, 4]
    Returns:
    [
        [1 ,2],
        [2, 3],
        [3, 4],
    ]
    """
    shape = (len(arr) - 1, 2)
    points = np.empty(shape)
    points[:, 0] = arr[:-1]
    points[:, 1] = arr[1:]
    return points


def __split_linestring_into_segments(dataframe):
    coordinates, index = shapely.get_coordinates(dataframe.geometry, return_index=True)
    # Create DataFrame to drop duplicates
    df = pd.DataFrame({"index": index, "x": coordinates[:, 0], "y": coordinates[:, 1]})
    df = df.drop_duplicates().reset_index(drop=True)

    # Work around issue with xu.snap_to_grid, where provided linestrings only
    # work if individual segments are provided.
    linestring_indices = __start_end_pairs_segments(df["index"])
    # Only preserve connections between two points where start and end share
    # the same hfb index.
    to_preserve = linestring_indices[:, 0] == linestring_indices[:, 1]
    linestring_indices = linestring_indices[to_preserve].ravel()

    points_x = __start_end_pairs_segments(df["x"])
    points_x = points_x[to_preserve].ravel()
    points_y = __start_end_pairs_segments(df["y"])
    points_y = points_y[to_preserve].ravel()

    segment_indices = np.repeat(np.arange(len(points_x) // 2), 2)

    segments = shapely.linestrings(points_x, y=points_y, indices=segment_indices)
    return gpd.GeoDataFrame(geometry=segments), linestring_indices[::2]


def create_hfb(cache, model, key, value, active, top, bottom, k, **kwargs):
    target = cache.target
    dataframe = value["geodataframe"]
    layer = value["layer"]

    # Split line into segments, which is required to run xu.snap_to_grid
    # FUTURE: Remove segmentation when xu.snap_to_grid updates.
    lines, hfb_indices = __split_linestring_into_segments(dataframe)

    if "resistance" in dataframe:
        lines["resistance"] = dataframe["resistance"][hfb_indices].values
    elif "multiplier" in dataframe:
        lines["resistance"] = -1.0 * dataframe["resistance"][hfb_indices].values
    else:
        raise ValueError(
            "Expected resistance or multiplier in HFB dataframe, "
            f"received instead: {list(dataframe.keys())}"
        )
    snapped, _ = xu.snap_to_grid(lines, grid=target, max_snap_distance=0.5)

    resistance = snapped["resistance"]
    if resistance.isnull().all():
        return

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
    "evt": create_evt,
    "ghb": create_ghb,
    "hfb": create_hfb,
    "shd": create_ic,
    "rch": create_rch,
    "riv": create_riv,
    "wel": create_wel,
}


def expand_repetitions(
    repeat_stress: List[datetime], time_min: datetime, time_max: datetime
) -> Dict[datetime, datetime]:
    expanded = {}
    for year, date in itertools.product(
        range(time_min.year, time_max.year + 1),
        repeat_stress,
    ):
        newdate = date.replace(year=year)
        if newdate < time_max:
            expanded[newdate] = date
    return expanded


def convert_to_disv(
    projectfile_data, target, time_min=None, time_max=None, repeat_stress=None
):
    """
    Convert the contents of a project file to a MODFLOW6 DISV model.

    The ``time_min`` and ``time_max`` are **both** required when
    ``repeat_stress`` is given. The entries in the Periods section of the
    project file will be expanded to yearly repeats between ``time_min`` and
    ``time_max``.

    Additionally, ``time_min`` and ``time_max`` may be used to slice the input
    to a specific time domain.

    The returned model is steady-state if none of the packages contain a time
    dimension. The model is transient if any of the packages contain a time
    dimension. This can be changed by setting the "transient" value in the
    storage package of the returned model. Storage coefficient input is
    required for a transient model.

    Parameters
    ----------
    projectfile_data: dict
        Dictionary with the projectfile topics as keys, and the data
        as xarray.DataArray, pandas.DataFrame, or geopandas.GeoDataFrame.
    target: xu.Ugrid2d
        The unstructured target topology. All data is transformed to match this
        topology.
    time_min: datetime, optional
        Minimum starting time of a stress.
        Required when ``repeat_stress`` is provided.
    time_max: datetime, optional
        Maximum starting time of a stress.
        Required when ``repeat_stress`` is provided.
    repeat_stress: dict of dict of string to datetime, optional
        This dict contains contains, per topic, the period alias (a string) to
        its datetime.

    Returns
    -------
    disv_model: imod.mf6.GroundwaterFlowModel

    """
    if repeat_stress is not None:
        if time_min is None or time_max is None:
            raise ValueError(
                "time_min and time_max are required when repeat_stress is given"
            )

    for arg in (time_min, time_max):
        if arg is not None and not isinstance(arg, datetime):
            raise TypeError(
                "time_min and time_max must be datetime.datetime. "
                f"Received: {type(arg).__name__}"
            )

    data = projectfile_data.copy()
    model = imod.mf6.GroundwaterFlowModel()

    # Setup the regridding weights cache.
    weights_cache = SingularTargetRegridderWeightsCache(data, target, cache_size=5)

    # Mandatory packages first.
    ibound = data["bnd"]["ibound"].compute()
    disv, top, bottom, active, original2d = create_disv(
        cache=weights_cache,
        top=data["top"]["top"],
        bottom=data["bot"]["bottom"],
        ibound=ibound,
    )

    npf = create_npf(
        cache=weights_cache,
        k=data["khv"]["kh"],
        vertical_anisotropy=data["kva"]["vertical_anisotropy"],
        active=active,
        original2d=original2d,
    )

    model["npf"] = npf
    model["disv"] = disv
    model["oc"] = imod.mf6.OutputControl(save_head="all")

    # Used in other package construction:
    idomain = disv["idomain"].compute()
    k = npf["k"].compute()
    new_ibound = weights_cache.regrid(source=ibound, method="minimum").compute()

    # Boundary conditions, one by one.
    for key, value in data.items():
        pkg = key.split("-")[0]
        convert = PKG_CONVERSION.get(pkg)
        # Skip unsupported packages
        if convert is None:
            continue

        if repeat_stress is None:
            repeat = None
        else:
            repeat = repeat_stress.get(key)
            if repeat is not None:
                repeat = expand_repetitions(repeat, time_min, time_max)

        try:
            # conversion will update model instance
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

    # Treat hfb's separately: they must be merged into one,
    # as MODFLOW6 only supports a single HFB.
    hfb_keys = [key for key in model.keys() if key.split("-")[0] == "hfb"]
    hfbs = [model.pop(key) for key in hfb_keys]
    if hfbs:
        model["hfb"] = merge_hfbs(hfbs, idomain)

    transient = any("time" in pkg.dataset.dims for pkg in model.values())
    if transient and (time_min is not None or time_max is not None):
        model = model.clip_box(time_min=time_min, time_max=time_max)

    sto_entry = data.get("sto")
    if sto_entry is None:
        if transient:
            raise ValueError("storage input is required for a transient run")
        storage_coefficient = None
    else:
        storage_coefficient = sto_entry["storage_coefficient"]

    model["sto"] = create_sto(
        cache=weights_cache,
        storage_coefficient=storage_coefficient,
        active=active,
        original2d=original2d,
        transient=transient,
    )

    return model
