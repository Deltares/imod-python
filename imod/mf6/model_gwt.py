from __future__ import annotations

from typing import Optional

from imod.logging import init_log_decorator
from imod.mf6.model import Modflow6Model
from imod.schemata import TypeSchema


class GroundwaterTransportModel(Modflow6Model):
    """
    The GroundwaterTransportModel (GWT) simulates transport of a single solute
    species flowing in groundwater.
    More information can be found here:
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.4.2.pdf#page=172

    Parameters
    ----------

    listing_file: Optional[str] = None
        name of the listing file to create for this GWT model. If not specified,
        then the name of the list file will be the basename of the GWT model
        name file and the 'lst' extension.
    print_input: bool = False
        if True, indicates that the list of exchange entries will be echoed to
        the listing file immediately after it is read.
    print_flows: bool = False
        if True, indicates that the list of exchange flow rates will be printed
        to the listing file for every stress period in which "SAVE BUDGET" is
        specified in Output Control
    save_flows: bool = False,
        if True, indicates that all model package flow terms will be written to
        the file specified with "BUDGET FILEOUT" in Output Control.
    """

    _mandatory_packages = ("mst", "dsp", "oc", "ic")
    _model_id = "gwt6"
    _template = Modflow6Model._initialize_template("gwt-nam.j2")

    _init_schemata = {
        "listing_file": [TypeSchema(str)],
        "print_input": [TypeSchema(bool)],
        "print_flows": [TypeSchema(bool)],
        "save_flows": [TypeSchema(bool)],
    }

    @init_log_decorator()
    def __init__(
        self,
        listing_file: Optional[str] = None,
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
        validate: bool = True,
    ):
        super().__init__()
        self._options = {
            "listing_file": listing_file,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
        }
        self.validate_init_schemata_options(validate)
