from copy import deepcopy
from typing import Self

import cftime
import numpy as np

from imod.logging import init_log_decorator
from imod.mf6.exchangebase import ExchangeBase
from imod.mf6.package import Package
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
        time_min: cftime.datetime | np.datetime64 | str | None = None,
        time_max: cftime.datetime | np.datetime64 | str | None = None,
        layer_min: int | None = None,
        layer_max: int | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        top: GridDataArray | None = None,
        bottom: GridDataArray | None = None,
    ) -> Self:
        """
        The GWF-GWT exchange does not have any spatial coordinates that can be
        clipped. Package is deepcopied instead.
        """
        return deepcopy(self)
