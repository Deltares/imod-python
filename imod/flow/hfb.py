import geopandas as gpd
import imod
import jinja2
import numpy as np
from imod.flow.pkgbase import Package, Vividict
from imod.wq import timeutil


class HorizontalFlowBarrier(Package):
    """
    Horizontal barriers obstructing flow such as semi- or impermeable fault
    zone or a sheet pile wall are defined for each model layer by a \*.GEN line
    file.

    Parameters
    ----------
    id_name: str or list of str
        name of the barrier
    geometry: object array of shapely LineStrings
        geometry of barriers, should be lines
    layer: "None" or int
        layer where barrier is located
    resistance: float or list of floats
        resistance of the barrier (d).
    """

    _template_projectfile = jinja2.Template(
        "0001, ({{pkg_id}}), 1, {{name}}, {{variable_order}}\n"
        '{{"{:03d}".format(variable_order|length)}}, {{"{:03d}".format(n_entry)}}\n'
        "{%- for variable in variable_order%}\n"  # Preserve variable order
        "{%-    for layer, value in package_data[variable].items()%}\n"
        # 1 indicates the layer is activated
        # 2 indicates the second element of the final two elements should be read
        # 1.000 is the multiplication factor
        # 0.000 is the addition factor
        # -9999 indicates there is no data, following iMOD usual practice
        '1, 2, {{"{:03d}".format(layer)}}, {{resistance[loop.index]}}, 0.000, -9999., {{value}}\n'
        "{%-    endfor %}\n"
        "{%- endfor %}\n"
    )

    _pkg_id = "hfb"
    _variable_order = ["resistance"]

    def __init__(
        self,
        id_name=None,
        geometry=None,
        layer=None,
        resistance=None,
    ):
        super(__class__, self).__init__()
        variables = {
            "id_name": id_name,
            "geometry": geometry,
            "layer": layer,
            "resistance": resistance,
        }
        variables = {k: np.atleast_1d(v) for k, v in variables.items() if v is not None}
        length = max(map(len, variables.values()))
        index = np.arange(1, length + 1)
        self.dataset["index"] = index

        for k, v in variables.items():
            if v.size == index.size:
                self.dataset[k] = ("index", v)
            elif v.size == 1:
                self.dataset[k] = ("index", np.full(length, v))
            else:
                raise ValueError(f"Length of {k} does not match other arguments")

    def _compose_values_layer(self, varname, directory, nlayer, time=None):
        values = {}
        d = {"directory": directory, "name": directory.stem, "extension": ".gen"}

        if "layer" in self.dataset:
            for layer in np.unique(self.dataset["layer"]):
                layer = int(layer)
                d["layer"] = layer
                values[layer] = self._compose_path(d)
        else:
            for layer in range(1, nlayer + 1):  # 1-based indexing
                values[layer] = self._compose_path(d)

        return values

    def _save_layers(self, gdf, directory):
        d = {"directory": directory, "name": directory.stem, "extension": ".gen"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        gdf["Id"] = gdf.index

        if "layer" in gdf:
            for layer, layerdf in gdf.groupby("layer"):
                d["layer"] = layer
                # Ensure right order
                outdf = layerdf[["Id", "geometry"]]
                path = self._compose_path(d)
                imod.gen.write(path, outdf)
        else:
            outdf = gdf[["Id", "geometry"]]
            path = self._compose_path(d)
            imod.gen.write(path, outdf)

    def save(self, directory):
        gdf = gpd.GeoDataFrame(self.dataset.to_dataframe())
        # Use _save_layers to keep the code consistent with the Wel implementation.
        self._save_layers(gdf, directory)

    def _render_projectfile(self, **kwargs):
        kwargs["resistance"] = self.dataset["resistance"].values
        return super()._render_projectfile(**kwargs)
