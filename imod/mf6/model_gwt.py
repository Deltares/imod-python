from __future__ import annotations

from imod.mf6.model import Modflow6Model, initialize_template
from imod.typing import GridDataArray


class GroundwaterTransportModel(Modflow6Model):
    """
    The GroundwaterTransportModel (GWT) simulates transport of a single solute
    species flowing in groundwater.
    """

    _mandatory_packages = ("mst", "dsp", "oc", "ic")
    _model_id = "gwt6"

    def __init__(
        self,
        listing_file: str = None,
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
    ):
        super().__init__()
        self._options = {
            "listing_file": listing_file,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
        }

        self._template = initialize_template("gwt-nam.j2")

    def clip_box(
        self,
        time_min: str = None,
        time_max: str = None,
        layer_min: int = None,
        layer_max: int = None,
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
        state_for_boundary: GridDataArray = None,
    ):
        clipped = super()._clip_box_packages(
            time_min, time_max, layer_min, layer_max, x_min, x_max, y_min, y_max
        )

        return clipped
