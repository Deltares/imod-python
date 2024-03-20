from __future__ import annotations

from typing import Optional

from imod.logging.logging_decorators import init_log_decorator
from imod.mf6.model import Modflow6Model


class GroundwaterTransportModel(Modflow6Model):
    """
    The GroundwaterTransportModel (GWT) simulates transport of a single solute
    species flowing in groundwater.
    """

    _mandatory_packages = ("mst", "dsp", "oc", "ic")
    _model_id = "gwt6"
    _template = Modflow6Model._initialize_template("gwt-nam.j2")

    @init_log_decorator()
    def __init__(
        self,
        listing_file: Optional[str] = None,
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

