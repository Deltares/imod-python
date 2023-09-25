from typing import Dict, Tuple

import jinja2
import numpy as np
import pandas as pd
import xarray as xr

from imod.mf6.package import Package
from imod.mf6.regridding_utils import RegridderType
from imod.typing.grid import GridDataArray


class GWFGWF(Package):
    """
    This package is for writing an exchange file, used for splitting up a model
    into different submodels (that can be solved in parallel). It (usually)
    is not instantiated by users, but created by the "split" method of the
    simulation class."""

    _keyword_map = {}
    _pkg_id = "gwfgwf"

    def __init__(
        self,
        model_id1: str,
        model_id2: str,
        cell_id1: np.ndarray,
        cell_id2: np.ndarray,
        layer: np.ndarray,
    ):
        self.dataset = xr.Dataset()
        self.dataset["cell_id1"] = cell_id1
        self.dataset["cell_id2"] = cell_id2
        self.dataset["layer"] = layer
        self.dataset["model_name_1"] = model_id1
        self.dataset["model_name_2"] = model_id2

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

    def render(self, directory, pkgname, globaltimes, binary):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        _template = env.get_template("exg-gwfgwf.j2")

        d = {}
        d["nexg"] = len(self.dataset["layer"].values)
        for varname in self.dataset.data_vars:
            key = self._keyword_map.get(varname, varname)

            value = self[varname].values[()]
            if self._valid(value):  # skip False or None
                d[key] = value
        # check dimensionality of cell_1d - it has 2 columns for structured grids and 1 column for unstructured grids

        exchangeblock = {
            "layer1": self.dataset["layer"].values,
            "cellid": self.dataset["cell_id1"].values,
            "layer2": self.dataset["layer"].values,
            "cellid2": self.dataset["cell_id2"].values,
        }

        exchangeblock_str = (
            pd.DataFrame(exchangeblock)
            .astype("O")
            .to_csv(sep=" ", header=False, index=False, line_terminator="\n")
        )
        exchangeblock_str = exchangeblock_str.replace('"', "")
        exchangeblock_str = exchangeblock_str.replace("(", "")
        exchangeblock_str = exchangeblock_str.replace(")", "")
        exchangeblock_str = exchangeblock_str.replace(",", "")
        d["exchangesblock"] = exchangeblock_str

        return _template.render(d)

    def get_specification(self) -> Tuple[str, str, str, str]:
        return (
            "GWF6-GWF6",
            self.filename(),
            self.dataset["model_name_1"].values[()],
            self.dataset["model_name_2"].values[()],
        )

    def regrid_like(
        self,
        target_grid: GridDataArray,
        regridder_types: Dict[str, Tuple[RegridderType, str]] = None,
    ) -> Package:
        raise Exception("this package cannot be regridded")

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
        raise Exception("this package cannot be clipped")
