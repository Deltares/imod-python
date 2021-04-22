from imod.flow.pkgbase import Package, Vividict
import imod
import numpy as np
from imod.wq import timeutil


class HorizontalFlowBoundary(Package):
    """
    Horizontal barriers obstructing flow such as semi- or impermeable fault zone or a sheet pile wall are
    defined for each model layer by a *.GEN line file.

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

    _pkg_id = ["hfb"]
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

    def _save_layers(self, df, directory):
        d = {"directory": directory, "name": directory.stem, "extension": "gen"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        df["Id"] = df["index"]
        df["feature_type"] = "line"

        if "layer" in df:
            for layer, layerdf in df.groupby("layer"):
                d["layer"] = layer
                # Ensure right order
                outdf = layerdf[["Id", "feature_type", "geometry"]]
                path = self._compose_path(d)
                imod.gen.write(path, outdf)
        else:
            outdf = df[["Id", "feature_type", "geometry"]]
            path = self._compose_path(d)
            imod.gen.write(path, outdf)

    def save(self, directory):
        # I used _save_layers to keep the code similar to the Wel implementation.
        self._save_layers(self.dataset.to_dataframe(), directory)

    # TODO 1. Write save function
    # TODO 2. Compose package
    # TODO 3. Create template (Note that the multiplication factor is used as the resistance)
    # TODO 4. Render
