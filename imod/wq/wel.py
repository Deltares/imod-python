from pathlib import Path

import jinja2
import numpy as np

import imod
from imod.wq.pkgbase import BoundaryCondition


def _column_order(df, variable):
    """
    Return ordered columns, and associated timeseries columns.
    """
    if "time" in df:
        assoc_columns = ["time", variable]
        if "layer" in df:
            columns = ["x", "y", "id", "time", "layer", variable]
        else:
            columns = ["x", "y", "id", "time", variable]
    else:
        assoc_columns = None
        if "layer" in df:
            columns = ["x", "y", variable, "layer", "id"]
        else:
            columns = ["x", "y", variable, "id"]
    return columns, assoc_columns


class Well(BoundaryCondition):
    """
    The Well package is used to simulate a specified flux to individual cells
    and specified in units of length3/time.

    Parameters
    ----------
    id_name: str or list of str
        name of the well(s).
    x: float or list of floats
        x coordinate of the well(s).
    y: float or list of floats
        y coordinate of the well(s).
    rate: float or list of floats.
        pumping rate in the well(s).
    layer: "None" or int, optional
        layer from which the pumping takes place.
    time: "None" or listlike of np.datetime64, datetime.datetime, pd.Timestamp,
    cftime.datetime
        time during which the pumping takes place. Only need to specify if model
        is transient.
    save_budget: bool, optional
        is a flag indicating if the budget should be saved (IRIVCB).
        Default is False.
    """

    _pkg_id = "wel"

    _template = jinja2.Template(
        "    {%- for time, timedict in wels.items() -%}"
        "        {%- for layer, value in timedict.items() %}\n"
        "    wel_p{{time}}_s{{system_index}}_l{{layer}} = {{value}}"
        "        {%- endfor -%}\n"
        "    {%- endfor -%}"
    )

    # TODO: implement well to concentration IDF and use ssm_template
    # Ignored for now, since wells are nearly always extracting

    def __init__(
        self,
        id_name,
        x,
        y,
        rate,
        layer=None,
        time=None,
        concentration=None,
        save_budget=False,
    ):
        super().__init__()
        variables = {
            "id_name": id_name,
            "x": x,
            "y": y,
            "rate": rate,
            "layer": layer,
            "time": time,
            "concentration": concentration,
        }
        variables = {k: np.atleast_1d(v) for k, v in variables.items() if v is not None}
        length = max(map(len, variables.values()))
        index = np.arange(1, length + 1)
        self["index"] = index

        for k, v in variables.items():
            if v.size == index.size:
                self[k] = ("index", v)
            elif v.size == 1:
                self[k] = ("index", np.full(length, v))
            else:
                raise ValueError(f"Length of {k} does not match other arguments")

        self["save_budget"] = save_budget

    def _max_active_n(self, varname, nlayer, nrow, ncol):
        """
        Determine the maximum active number of cells that are active
        during a stress period.

        Parameters
        ----------
        varname : str
            name of the variable to use to calculate the maximum number of
            active cells. Not used for well, here for polymorphism.
        nlayer, nrow, ncol : int
        """
        nmax = np.unique(self["id_name"]).size
        if "layer" not in self.dataset.coords:  # Then it applies to every layer
            nmax *= nlayer
        self._cellcount = nmax
        self._ssm_cellcount = nmax
        return nmax

    def _compose_values_layer(self, directory, nlayer, name, time=None, compress=True):
        values = {}
        d = {"directory": directory, "name": name, "extension": ".ipf"}

        if time is None:
            if "layer" in self.dataset:
                for layer in np.unique(self.dataset["layer"]):
                    layer = int(layer)
                    d["layer"] = layer
                    values[layer] = self._compose_path(d)
            else:
                values["$"] = self._compose_path(d)

        else:
            d["time"] = time
            if "layer" in self.dataset:
                # Since the well data is in long table format, it's the only
                # input that has to be inspected.
                select = np.argwhere((self.dataset["time"] == time).values)
                for layer in np.unique(self.dataset["layer"].values[select]):
                    d["layer"] = layer
                    values[layer] = self._compose_path(d)
            else:
                values["?"] = self._compose_path(d)

        if "layer" in self.dataset:
            # Compose does not accept non-integers, so use 0, then replace
            d["layer"] = 0
            if np.unique(self.dataset["layer"].values).size == nlayer:
                token_path = imod.util.compose(d).as_posix()
                token_path = token_path.replace("_l0", "_l$")
                values = {"$": token_path}
            elif compress:
                range_path = imod.util.compose(d).as_posix()
                range_path = range_path.replace("_l0", "_l:")
                # TODO: temporarily disable until imod-wq is fixed
                values = self._compress_idflayers(values, range_path)

        return values

    def _compose_values_time(self, directory, name, globaltimes, nlayer):
        # TODO: rename to _compose_values_timelayer?
        values = {"?": self._compose_values_layer(directory, nlayer=nlayer, name=name)}
        return values

    def _render(self, directory, globaltimes, system_index, nlayer):
        d = {"system_index": system_index}
        d["wels"] = self._compose_values_time(directory, "rate", globaltimes, nlayer)
        return self._template.render(d)

    def _render_ssm(self, directory, globaltimes, nlayer):
        if "concentration" in self.dataset.data_vars:
            d = {"pkg_id": self._pkg_id}
            name = "concentration"
            if "species" in self.dataset["concentration"].coords:
                concentration = {}
                for species in self.dataset["concentration"]["species"].values:
                    concentration[species] = self._compose_values_time(
                        directory, f"{name}_c{species}", globaltimes, nlayer=nlayer
                    )
            else:
                concentration = {
                    1: self._compose_values_time(
                        directory, name, globaltimes, nlayer=nlayer
                    )
                }
            d["concentration"] = concentration
            return self._ssm_template.render(d)
        else:
            return ""

    def _save_layers(self, df, directory, name, variable):
        d = {"directory": directory, "name": name, "extension": ".ipf"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        if "time" in df:
            itype = "timeseries"
        else:
            itype = None

        columns, assoc_columns = _column_order(df, variable)
        path = self._compose_path(d)
        df = df[columns]
        if "layer" in df:
            for layer, layerdf in df.groupby("layer"):
                # Ensure different IDs per layer are not overwritten.
                layerdf["id"] = f"{name}_l{layer}/" + layerdf["id"].astype(str)
                imod.ipf.save(
                    path=path, df=layerdf, itype=itype, assoc_columns=assoc_columns
                )
        else:
            imod.ipf.save(path=path, df=df, itype=itype, assoc_columns=assoc_columns)

        return

    def save(self, directory):
        directory = Path(directory)

        all_species = [None]
        if "concentration" in self.dataset.data_vars:
            if "species" in self.dataset["concentration"].coords:
                all_species = self.dataset["concentration"]["species"].values

        df = self.dataset.to_dataframe().rename(columns={"id_name": "id"})
        self._save_layers(df, directory, "rate", "rate")

        # Loop over species if applicable
        if "concentration" in self.dataset:
            for species in all_species:
                if species is not None:
                    ds = self.dataset.sel(species=species)
                else:
                    ds = self.dataset

                df = ds.to_dataframe().rename(columns={"id_name": "id"})
                name = "concentration"
                if species is not None:
                    name = f"{name}_c{species}"
                self._save_layers(df, directory, name, "concentration")

        return

    def _pkgcheck(self, ibound=None):
        # TODO: implement
        pass

    def repeat_stress(self, stress_repeats, use_cftime=False):
        raise NotImplementedError(
            "Well does not support repeated stresses: time-varying data is "
            "saved into associated IPF files. Set explicit timeseries intead."
        )
