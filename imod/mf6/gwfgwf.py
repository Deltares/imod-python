from typing import Dict, Optional, Tuple, Union

import numpy as np
import xarray as xr

from imod.mf6.auxiliary_variables import add_periodic_auxiliary_variable
from imod.mf6.package import Package


class GWFGWF(Package):
    """
    This package is for writing an exchange file, used for splitting up a model
    into different submodels (that can be solved in parallel). It (usually)
    is not instantiated by users, but created by the "split" method of the
    simulation class."""

    _keyword_map: Dict[str, str] = {}
    _auxiliary_data = {"auxiliary_data": "variable"}
    _pkg_id = "gwfgwf"
    _template = Package._initialize_template(_pkg_id)

    def __init__(
        self,
        model_id1: str,
        model_id2: str,
        cell_id1: xr.DataArray,
        cell_id2: xr.DataArray,
        layer: xr.DataArray,
        cl1: xr.DataArray,
        cl2: xr.DataArray,
        hwva: xr.DataArray,
        angldegx: Optional[xr.DataArray] = None,
        cdist: Optional[xr.DataArray] = None,
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

        auxiliary_variables = [var for var in [angldegx, cdist] if var is not None]
        if auxiliary_variables:
            self.dataset["auxiliary_data"] = xr.merge(auxiliary_variables).to_array(
                name="auxiliary_data"
            )
            add_periodic_auxiliary_variable(self)

    def set_options(
        self,
        print_input: Optional[bool] = None,
        print_flows: Optional[bool] = None,
        save_flows: Optional[bool] = None,
        cell_averaging: Optional[str] = None,
        dewatered: Optional[bool] = None,
        variablecv: Optional[bool] = None,
        xt3d: Optional[bool] = None,
        newton: Optional[bool] = None,
    ):
        self._toggle_options("print_input", print_input)
        self._toggle_options("print_flows", print_flows)
        self._toggle_options("save_flows", save_flows)
        self._toggle_options("cell_averaging", cell_averaging)
        self._toggle_options("dewatered", dewatered)
        self._toggle_options("variablecv", variablecv)
        self._toggle_options("xt3d", xt3d)
        self._toggle_options("newton", newton)

    def _toggle_options(
        self, option_name: str, option_value: Optional[Union[bool, str]]
    ):
        if option_value:
            self.dataset[option_name] = option_value
        else:
            self.dataset.drop_vars(option_name, errors="ignore")

    def filename(self) -> str:
        return f"{self.packagename()}.{self._pkg_id}"

    def packagename(self) -> str:
        return f"{self.dataset['model_name_1'].values[()]}_{self.dataset['model_name_2'].values[()]}"

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
