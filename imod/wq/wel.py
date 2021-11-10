import jinja2
import numpy as np

import imod
from imod import util
from imod.wq import timeutil
from imod.wq.pkgbase import BoundaryCondition


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

    __slots__ = ("save_budget",)
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
        if "layer" not in self.coords:  # Then it applies to every layer
            nmax *= nlayer
        self._cellcount = nmax
        self._ssm_cellcount = nmax
        return nmax

    def _compose_values_layer(self, directory, nlayer, name, time=None, compress=True):
        values = {}
        d = {"directory": directory, "name": name, "extension": ".ipf"}

        if time is None:
            if "layer" in self:
                for layer in np.unique(self["layer"]):
                    layer = int(layer)
                    d["layer"] = layer
                    values[layer] = self._compose(d)
            else:
                values["?"] = self._compose(d)

        else:
            d["time"] = time
            if "layer" in self:
                # Since the well data is in long table format, it's the only
                # input that has to be inspected.
                select = np.argwhere((self["time"] == time).values)
                for layer in np.unique(self["layer"].values[select]):
                    d["layer"] = layer
                    values[layer] = self._compose(d)
            else:
                values["?"] = self._compose(d)

        if "layer" in self:
            # Compose does not accept non-integers, so use 0, then replace
            d["layer"] = 0
            if np.unique(self["layer"].values).size == nlayer:
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
        values = {}
        if "time" in self:
            self_times = np.unique(self["time"].values)
            if "stress_repeats" in self.attrs:
                stress_repeats_keys = np.array(
                    list(self.attrs["stress_repeats"].keys())
                )
                stress_repeats_values = np.array(
                    list(self.attrs["stress_repeats"].values())
                )
                package_times, inds = np.unique(
                    np.concatenate([self_times, stress_repeats_keys]), return_index=True
                )
                # Times to write in the runfile
                runfile_times = np.concatenate([self_times, stress_repeats_values])[
                    inds
                ]
            else:
                runfile_times = package_times = self_times

            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)

            for time, start_end in zip(runfile_times, starts_ends):
                # Check whether any range occurs in the input.
                # If does does, compress should be False
                compress = not (":" in start_end)
                values[start_end] = self._compose_values_layer(
                    directory, nlayer=nlayer, name=name, time=time, compress=compress
                )
        else:  # for all periods
            values["?"] = self._compose_values_layer(
                directory, nlayer=nlayer, name=name
            )
        return values

    def _render(self, directory, globaltimes, system_index, nlayer):
        d = {"system_index": system_index}
        name = directory.stem
        d["wels"] = self._compose_values_time(directory, name, globaltimes, nlayer)
        return self._template.render(d)

    def _render_ssm(self, directory, globaltimes, nlayer):
        if "concentration" in self.data_vars:
            d = {"pkg_id": self._pkg_id}
            name = f"{directory.stem}-concentration"
            if "species" in self["concentration"].coords:
                concentration = {}
                for species in self["concentration"]["species"].values:
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

    def _save_layers(self, df, directory, time=None):
        d = {"directory": directory, "name": directory.stem, "extension": ".ipf"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        if time is not None:
            d["time"] = time

        if "layer" in df:
            for layer, layerdf in df.groupby("layer"):
                d["layer"] = layer
                # Ensure right order
                outdf = layerdf[["x", "y", "rate", "id_name"]]
                path = self._compose(d)
                imod.ipf.write(path, outdf)
        else:
            outdf = df[["x", "y", "rate", "id_name"]]
            path = self._compose(d)
            imod.ipf.write(path, outdf)

    @staticmethod
    def _save_layers_concentration(df, directory, name, time=None):
        d = {"directory": directory, "name": name, "extension": ".ipf"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        if time is not None:
            d["time"] = time

        if "layer" in df:
            for layer, layerdf in df.groupby("layer"):
                d["layer"] = layer
                # Ensure right order
                outdf = layerdf[["x", "y", "concentration", "id_name"]]
                path = util.compose(d)
                imod.ipf.write(path, outdf)
        else:
            outdf = df[["x", "y", "concentration", "id_name"]]
            path = util.compose(d)
            imod.ipf.write(path, outdf)

    def save(self, directory):
        all_species = [None]
        if "concentration" in self.data_vars:
            if "species" in self["concentration"].coords:
                all_species = self["concentration"]["species"].values

        # Loop over species if applicable
        for species in all_species:
            if species is not None:
                ds = self.sel(species=species)
            else:
                ds = self

            if "time" in ds:
                for time, timeda in ds.groupby("time"):
                    timedf = timeda.to_dataframe()
                    ds._save_layers(timedf, directory, time=time)
                    if "concentration" in ds.data_vars:
                        name = f"{directory.stem}-concentration"
                        if species is not None:
                            name = f"{name}_c{species}"
                        ds._save_layers_concentration(
                            timedf, directory, name, time=time
                        )
            else:
                ds._save_layers(ds.to_dataframe(), directory)
                if "concentration" in ds.data_vars:
                    name = f"{directory.stem}-concentration"
                    if species is not None:
                        name = f"{name}_c{species}"
                    ds._save_layers_concentration(timedf, directory, name)

    def _pkgcheck(self, ibound=None):
        # TODO: implement
        pass

    def repeat_stress(self, stress_repeats, use_cftime=False):
        # To ensure consistency, it isn't possible to use differing stress_repeatss
        # between rate and concentration: the number of points might change
        # between stress periods, and isn't especially easy to check.
        if "time" not in self:
            raise ValueError(
                "This Wel package does not have time, cannot add stress_repeats."
            )
        # Replace both key and value by the right datetime type
        d = {
            timeutil.to_datetime(k, use_cftime): timeutil.to_datetime(v, use_cftime)
            for k, v in stress_repeats.items()
        }
        self.attrs["stress_repeats"] = d
