from pathlib import Path
from typing import Dict, List, Tuple, Union

import jinja2
import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.package import Package


def convert_tuples_to_columns(tuple_array: np.ndarray) -> xr.DataArray:
    # TODO: task gitlab 550
    # this conversion will not be necessary anymore if we give the gwfgwf package 2d arrays for structured grids and 1d arrays for unstructured grids.
    if not isinstance(tuple_array[0], (int, np.integer)):
        array2d: np.ndarray = np.ndarray((len(tuple_array), 2), dtype=int)

        for index in range(len(tuple_array)):
            array2d[index, 0] = tuple_array[index][0]
            array2d[index, 1] = tuple_array[index][1]
        return xr.DataArray(array2d)
    else:
        return tuple_array


class GWFGWF(Package):
    """
    This package is for writing an exchange file, used for splitting up a model
    into different submodels (that can be solved in parallel). It (usually)
    is not instantiated by users, but created by the "split" method of the
    simulation class."""

    _keyword_map: Dict[str, str] = {}
    _pkg_id = "gwfgwf"

    def __init__(
        self,
        model_id1: str,
        model_id2: str,
        cell_id1: np.ndarray,
        cell_id2: np.ndarray,
        layer: np.ndarray,
        cl1: np.ndarray,
        cl2: np.ndarray,
        hwva: np.ndarray,
    ):
        self.dataset = xr.Dataset()
        self.dataset["cell_id1"] = convert_tuples_to_columns(cell_id1)
        self.dataset["cell_id2"] = convert_tuples_to_columns(cell_id2)
        self.dataset["layer"] = layer
        self.dataset["model_name_1"] = model_id1
        self.dataset["model_name_2"] = model_id2
        self.dataset["ihc"] = np.ones_like(cl1, dtype=int)
        self.dataset["cl1"] = cl1
        self.dataset["cl2"] = cl2
        self.dataset["hwva"] = hwva

    def set_options(
        self,
        print_input: bool,
        print_flows: bool,
        save_flows: bool,
        cell_averaging: bool,
        variablecv: bool,
        newton: bool,
    ):
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["cell_averaging"] = cell_averaging
        self.dataset["variable_cv"] = variablecv
        self.dataset["newton"] = newton

    def filename(self) -> str:
        return f"{self.packagename() }.{self._pkg_id}"

    def packagename(self) -> str:
        return f"{self.dataset['model_name_1'].values[()]}_{self.dataset['model_name_2'].values[()] }"

    def render(
        self,
        directory: Path,
        pkgname: str,
        globaltimes: Union[List, np.ndarray],
        binary: bool,
    ):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        _template = env.get_template("exg-gwfgwf.j2")

        cellid_shape_dimension = len(self.dataset["cell_id1"].shape)

        d = {}
        d["nexg"] = len(self.dataset["layer"].values)
        for varname in self.dataset.data_vars:
            key = self._keyword_map.get(varname, varname)

            value = self[varname].values[()]
            if self._valid(value):  # skip False or None
                d[key] = value
        # check dimensionality of cell_1d - it has 2 columns for structured grids and 1 column for unstructured grids
        if cellid_shape_dimension == 1:
            exchangeblock = {
                "layer1": self.dataset["layer"].values,
                "cellid": self.dataset["cell_id1"].values[:],
                "layer2": self.dataset["layer"].values,
                "cellid2": self.dataset["cell_id2"].values[:],
                "ihc": self.dataset["ihc"],
                "cl1": self.dataset["cl1"],
                "cl2": self.dataset["cl2"],
                "hwva": self.dataset["hwva"],
            }
        elif cellid_shape_dimension == 2:
            exchangeblock = {
                "layer1": self.dataset["layer"].values,
                "cellid1_d1": self.dataset["cell_id1"].values[:, 0],
                "cellid1_d2": self.dataset["cell_id1"].values[:, 1],
                "layer2": self.dataset["layer"].values,
                "cellid2_d1": self.dataset["cell_id2"].values[:, 0],
                "cellid2_d2": self.dataset["cell_id2"].values[:, 1],
                "ihc": self.dataset["ihc"],
                "cl1": self.dataset["cl1"],
                "cl2": self.dataset["cl2"],
                "hwva": self.dataset["hwva"],
            }
        else:
            raise ValueError("unexpected dimension in cell_id1")
        exchangeblock_str = (
            pd.DataFrame(exchangeblock)
            .astype("O")
            .to_csv(sep=" ", header=False, index=False, line_terminator="\n")
        )
        d["exchangesblock"] = exchangeblock_str

        return _template.render(d)

    def get_specification(self) -> Tuple[str, str, str, str]:
        """
        Returns a tuple containing the exchange type, the exchange file name, and the model names. This can be used
        to write the exchange information in the simulation .nam input file
        """
        return (
            "GWF6-GWF6",
            self.filename(),
            self.dataset["model_name_1"].values[()],
            self.dataset["model_name_2"].values[()],
        )

    def clip_box(
        self,
        time_min=None,
        time_max=None,
        layer_min=None,
        layer_max=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        state_for_boundary=None,
    ) -> Package:
        raise NotImplementedError("this package cannot be clipped")
