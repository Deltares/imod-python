from typing import Optional

import cftime
import numpy as np
import xarray as xr

from imod.mf6.auxiliary_variables import expand_transient_auxiliary_variables
from imod.mf6.exchangebase import ExchangeBase
from imod.mf6.package import Package
from imod.typing import GridDataArray


class GWFGWF(ExchangeBase):
    """
    This package is for writing an exchange file, used for splitting up a model
    into different submodels (that can be solved in parallel). It (usually)
    is not instantiated by users, but created by the "split" method of the
    simulation class."""

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
        dict_dataset = {
            "cell_id1": cell_id1,
            "cell_id2": cell_id2,
            "layer": layer,
            "model_name_1": model_id1,
            "model_name_2": model_id2,
            "ihc": xr.DataArray(np.ones_like(cl1, dtype=int)),
            "cl1": cl1,
            "cl2": cl2,
            "hwva": hwva,
        }
        super().__init__(dict_dataset)

        auxiliary_variables = [var for var in [angldegx, cdist] if var is not None]
        if auxiliary_variables:
            self.dataset["auxiliary_data"] = xr.merge(auxiliary_variables).to_array(
                name="auxiliary_data"
            )
            expand_transient_auxiliary_variables(self)

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
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["cell_averaging"] = cell_averaging
        self.dataset["dewatered"] = dewatered
        self.dataset["variablecv"] = variablecv
        self.dataset["xt3d"] = xt3d
        self.dataset["newton"] = newton

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        top: Optional[GridDataArray] = None,
        bottom: Optional[GridDataArray] = None,
        state_for_boundary: Optional[GridDataArray] = None,
    ) -> Package:
        raise NotImplementedError("this package cannot be clipped")
