from typing import Optional

import cftime
import numpy as np

from imod.mf6.exchangebase import ExchangeBase, ExchangeType
from imod.mf6.package import Package
from imod.typing import GridDataArray


class GWFGWT(ExchangeBase):
    _exchange_type = ExchangeType.GWFGWT
    _pkg_id = "gwfgwt"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, model_id1: str, model_id2: str):
        super().__init__(locals())
        self.dataset["model_name_1"] = model_id1
        self.dataset["model_name_2"] = model_id2

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
        pass
