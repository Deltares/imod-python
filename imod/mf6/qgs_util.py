from itertools import product
from xml.sax.saxutils import unescape

import declxml as xml
import numpy as np
import pyproj

# For creating colormap
from matplotlib import cm, colors

import imod
import imod.qgs as qgs


def _get_color_hexes_cmap(n, cmap_name="magma"):
    cmap = cm.get_cmap(cmap_name, n)
    return [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]


def _create_colorramp(data_min, data_max, n, cmap_name="magma"):
    hexes = _get_color_hexes_cmap(n, cmap_name)
    values = np.linspace(data_min, data_max, num=n)
    values_str = ["{:.2f}".format(value) for value in values]
    items = [
        qgs.Item(label=value_str, value=value_str, color=hex)
        for value_str, hex in zip(values_str, hexes)
    ]
    return qgs.ColorRampShader(colorramp=qgs.ColorRamp(), item=items)


def _generate_layer_ids(pkgnames):
    """Generate unique layer ids for the qgis projectfile

    The format of these layer ids is very important, shorter strings are not accepted by Qgis.
    """
    rng = np.random.default_rng()
    random_ints = rng.integers(32000, size=len(pkgnames), dtype=np.int16)
    return [
        "{}_00000000_0000_0000_0000_0000000{:05d}".format(pkgname, integer)
        for pkgname, integer in zip(pkgnames, random_ints)
    ]


def _create_groups(pkg, data_vars, aggregate_layers):
    """Create unique groups for each data variable - layer combination"""
    if ("layer" in pkg.dataset.dims) and not aggregate_layers:
        layers = pkg.dataset["layer"].values
    else:
        layers = [1]  # Exception for recharge package
    return list(product(data_vars, layers))


def _data_range_per_data_var(pkg, data_vars):
    data_min = dict([(var, float(pkg.dataset[var].min())) for var in data_vars])
    data_max = dict([(var, float(pkg.dataset[var].max())) for var in data_vars])
    return data_min, data_max


def _get_time_range(model):
    modeltimes = model._yield_times()
    start_time = min([min(pkgtimes) for pkgtimes in modeltimes])
    end_time = max([max(pkgtimes) for pkgtimes in modeltimes])
    return start_time, end_time


def _create_mesh_dataset_group(pkgname, groups):
    provider_names = [r"{}_layer:{}".format(var, layer) for var, layer in groups]
    mesh_dataset_group_tree_items = [
        qgs.Mesh_Dataset_Group_Tree_Item(dataset_index=str(i), provider_name=name)
        for i, name in enumerate(provider_names)
    ]
    # For some reason this list starts with an item with index -1
    mesh_dataset_group_tree_items.insert(
        0, qgs.Mesh_Dataset_Group_Tree_Item(dataset_index=str(-1), provider_name="")
    )
    return qgs.Dataset_Groups_Tree(
        mesh_dataset_group_tree_item=mesh_dataset_group_tree_items
    )


def _create_maplayers(
    model,
    pkgnames,
    layer_ids,
    data_paths,
    data_vars_ls,
    extent,
    spatial_ref_sys,
    aggregate_layers,
):
    time_format = "%Y-%m-%dT%H:%M:%SZ"
    start_time_string, end_time_string = [
        imod.util._compose_timestring(t, time_format=time_format)
        for t in _get_time_range(model)
    ]
    temporal = qgs.Temporal(
        reference_time=start_time_string,
        start_time_extent=start_time_string,
        end_time_extent=end_time_string,
    )

    maplayers = []
    for pkgname, layer_id, data_path, data_vars in zip(
        pkgnames, layer_ids, data_paths, data_vars_ls
    ):
        pkg = model[pkgname]
        groups = _create_groups(pkg, data_vars, aggregate_layers)

        dataset_group = _create_mesh_dataset_group(pkgname, groups)

        data_min, data_max = _data_range_per_data_var(pkg, data_vars)

        scalar_settings = [
            qgs.Scalar_Settings(
                group=group_nr,
                colorrampshader=_create_colorramp(
                    data_min[data_var], data_max[data_var], 20
                ),
                min_val=data_min[data_var],
                max_val=data_max[data_var],
            )
            for group_nr, (data_var, _) in enumerate(groups)
        ]

        srs = qgs.Srs(spatialrefsys=spatial_ref_sys)

        maplayer_kwargs = dict(
            srs=srs,
            extent=extent,
            id=layer_id,
            layername=pkgname,
            dataset_groups_tree=dataset_group,
            datasource=data_path,
            mesh_renderer_settings=qgs.Mesh_Renderer_Settings(
                scalar_settings=scalar_settings
            ),
        )

        if "time" in pkg.dataset.dims:
            maplayer_kwargs["provider"] = qgs.Provider(time_unit="3")
            maplayer_kwargs["temporal"] = temporal
        else:
            maplayer_kwargs["provider"] = qgs.Provider()

        maplayers.append(qgs.MapLayer(**maplayer_kwargs))
    return maplayers


def _create_qgis_tree(
    model, pkgnames, data_paths, data_vars_ls, crs, aggregate_layers=False
):
    """Create tree of qgis objects.


    Parameters
    ----------
    model : imod.wq.Model, imod.mf6.Model
        model objects, to which the packages are assigned
    pkgnames : list of str
        list with names of packages that contain an x and y dimension
    data_paths : list of str
        relative path to the respective netcdf a package is saved in
    data_vars_ls : nested list of str
        list with a list of str per variable that contains an x and y dimension
    crs : TYPE
        project crs

    Returns
    -------
    qgs_tree : qgs.QGIS
        a tree-like object with all qgis settings, which can be saved to xml

    """
    # Find if "dis" should be taken (mf6) or "bas" (wq)
    for n in ["dis", "bas"]:
        if n in model.keys():
            spatial_ref_var = n

    spatial_ref_sys = qgs.SpatialRefSys(
        wkt=pyproj.crs.CRS(crs).to_wkt(), geographicflag=True
    )

    # Create map canvas
    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(model[spatial_ref_var])
    extent = qgs.Extent(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    mapcanvas = qgs.MapCanvas(extent=extent)

    # Create legend
    layer_ids = _generate_layer_ids(pkgnames)
    legendlayers = [
        qgs.LegendLayer(
            name=pkgname, filegroup=qgs.FileGroup(qgs.LegendLayerFile(layerid=layer_id))
        )
        for pkgname, layer_id in zip(pkgnames, layer_ids)
    ]
    legend = qgs.Legend(legendlayer=legendlayers)

    # Create project layers
    maplayers = _create_maplayers(
        model,
        pkgnames,
        layer_ids,
        data_paths,
        data_vars_ls,
        extent,
        spatial_ref_sys,
        aggregate_layers,
    )
    projectlayers = qgs.ProjectLayers(maplayer=maplayers)

    # Create layer tree
    custom_order = qgs.Custom_Order(item=layer_ids)
    layer_tree_layers = []
    for pkgname, layer_id, data_path in zip(pkgnames, layer_ids, data_paths):
        layer_tree_layers.append(
            qgs.Layer_Tree_Layer(name=pkgname, id=layer_id, source=data_path)
        )
    layer_tree_group = qgs.Layer_Tree_Group_Leaf(
        layer_tree_layer=layer_tree_layers, custom_order=custom_order
    )

    layerorder = qgs.LayerOrder(layer=[qgs.Layer(id=i) for i in layer_ids])

    qgs_tree = qgs.Qgis(
        projectcrs=qgs.ProjectCrs(spatialrefsys=spatial_ref_sys),
        layer_tree_group_leaf=layer_tree_group,
        layerorder=layerorder,
        mapcanvas=mapcanvas,
        legend=legend,
        projectlayers=projectlayers,
    )
    return qgs_tree


def _write_qgis_projectfile(qgs_tree, qgs_path):
    processor = qgs.make_processor(qgs.Qgis)
    text = xml.serialize_to_string(processor, qgs_tree, indent="  ")
    # Double quote characters get automatically changed to '&quot;' by this function
    # Because when using indents, minidom.toprettyxml() is used, which automatically escapes double quotes
    # Double quotes are unfortunately included in WKT strings, so we need to fix this by unescaping
    text = unescape(text, {"&quot;": '"'})

    lines_out = text.splitlines(keepends=True)[
        1:
    ]  # remove the first line as it is just some xml info we do not want.

    with open(qgs_path, "w") as f:
        f.writelines(lines_out)
