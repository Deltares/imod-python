import numpy as np

from imod.mf6.package import Package
from imod.schemata import (
    DTypeSchema,
)

_init_schemata = {
    "maxbound": [DTypeSchema(np.integer)],
}

class ApiPackage(Package):
    """
    Parameters
    ----------    
    maxbound: int
        The number of cells for which information will be queried or set with api calls. 
    print_input: ({True, False}, optional)
        keyword to indicate that the list of constant head information will
        be written to the listing file immediately after it is read. Default is
        False.    
    print_flows: ({True, False}, optional)
        Indicates that the list of constant head flow rates will be printed to
        the listing file for every stress period time step in which "BUDGET
        PRINT" is specified in Output Control. If there is no Output Control
        option and PRINT FLOWS is specified, then flow rates are printed for the
        last time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that constant head flow terms will be written to the file
        specified with "BUDGET FILEOUT" in Output Control. Default is False.
    
    .. note::
       This package can be added to both flow and transport models.
    """

    _pkg_id = "api"
    _template = Package._initialize_template(_pkg_id)


    def __init__(self, maxbound: int, print_input: bool= False, print_flows: bool = False, save_flows: bool= False, validate:bool=True):
        dict_dataset = {
            "maxbound": maxbound,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
        }     
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    