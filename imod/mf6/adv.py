import numpy as np
import pandas as pd
from imod.mf6.pkgbase import Package
from enum import Enum

class AdvectionSchemes(Enum):
    '''
    Enumerator of available numerical schemes for advection.
    '''
    upstream = 1
    central = 2
    TVD = 3


class Advection(Package):
    """
    Advection (Adv)

    Parameters
    ----------
    scheme: enumerator of type AdvectionSchemes
    """
    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)
    _defaultScheme = AdvectionSchemes.upstream    
 

    def __init__(self, scheme=_defaultScheme):     
        super().__init__()
        schemeToString={}
        schemeToString[AdvectionSchemes.upstream] = 'upstream'
        schemeToString[AdvectionSchemes.central] = 'central'
        schemeToString[AdvectionSchemes.TVD] = 'TVD'   
        self.dataset["scheme"] = schemeToString[scheme] 
    