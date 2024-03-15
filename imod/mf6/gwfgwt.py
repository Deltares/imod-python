from copy import deepcopy
from typing import Optional

import cftime
import numpy as np

from imod.mf6.exchangebase import ExchangeBase
from imod.mf6.package import Package
from imod.mf6.utilities.logging_decorators import init_log_decorator
from imod.typing import GridDataArray


class GWFGWT(ExchangeBase):
    _pkg_id = "gwfgwt"
    
    _template = Package._initialize_template(_pkg_id)
    @init_log_decorator()
    def __init__(self, model_id1: str, model_id2: str):
        dict_dataset = {
            "model_name_1": model_id1,
            "model_name_2": model_id2,
        }

        super().__init__(dict_dataset)

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
        """
        The GWF-GWT exchange does not have any spatial coordinates that can be clipped.
        """
        return deepcopy(self)
