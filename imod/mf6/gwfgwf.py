from typing import Dict, Tuple

import numpy as np
import xarray as xr

from imod.mf6.package import Package


class GWFGWF(Package):
    """
    This package is for writing an exchange file, used for splitting up a model
    into different submodels (that can be solved in parallel). It (usually)
    is not instantiated by users, but created by the "split" method of the
    simulation class."""

    _keyword_map: Dict[str, str] = {}
    _pkg_id = "gwfgwf"
    _template = Package._initialize_template(_pkg_id)

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
        **kwargs,
    ):
        super().__init__(locals())
        self.dataset["cell_id1"] = cell_id1
        self.dataset["cell_id2"] = cell_id2
        self.dataset["layer"] = layer
        self.dataset["model_name_1"] = model_id1
        self.dataset["model_name_2"] = model_id2
        self.dataset["ihc"] = xr.DataArray(np.ones_like(cl1, dtype=int))
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

    def get_specification(self) -> Tuple[str, str, str, str]:
        """
        Returns a tuple containing the exchange type, the exchange file name, and the model names. This can be used
        to write the exchange information in the simulation .nam input file
        """
        return (
            "GWF6-GWF6",
            self.filename(),
            self.dataset["model_name_1"].values[()].take(0),
            self.dataset["model_name_2"].values[()].take(0),
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
