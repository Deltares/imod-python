from pathlib import Path
from typing import Optional, Union

import cftime
import numpy as np
import xarray as xr

from imod.logging.logging_decorators import init_log_decorator
from imod.mf6.auxiliary_variables import expand_transient_auxiliary_variables
from imod.mf6.exchangebase import ExchangeBase
from imod.mf6.package import Package
from imod.typing import GridDataArray


class GWTGWT(ExchangeBase):
    """
    This package is for writing an exchange file, used for splitting up a model
    into different submodels (that can be solved in parallel). It (usually)
    is not instantiated by users, but created by the "split" method of the
    simulation class.
    """

    _auxiliary_data = {"auxiliary_data": "variable"}
    _pkg_id = "gwtgwt"
    _template = Package._initialize_template(_pkg_id)

    @init_log_decorator()
    def __init__(
        self,
        transport_model_id1: str,
        transport_model_id2: str,
        flow_model_id1: str,
        flow_model_id2: str,
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
            "model_name_1": transport_model_id1,
            "model_name_2": transport_model_id2,
            "flow_model_name_1": flow_model_id1,
            "flow_model_name_2": flow_model_id2,
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
        save_flows: Optional[bool] = None,
        adv_scheme: Optional[str] = None,
        dsp_xt3d_off: Optional[bool] = None,
        dsp_xt3d_rhs: Optional[bool] = None,
    ):
        self.dataset["save_flows"] = save_flows
        self.dataset["adv_scheme"] = adv_scheme
        self.dataset["dsp_xt3d_off"] = dsp_xt3d_off
        self.dataset["dsp_xt3d_rhs"] = dsp_xt3d_rhs

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


    def render(self, directory: Path, pkgname: str, globaltimes: Union[list[np.datetime64], np.ndarray], binary: bool) -> str:
        return self.render_with_geometric_constants(directory, pkgname, globaltimes, binary)