import numpy as np
import pandas as pd
from imod.mf6.pkgbase import Package
from enum import Enum


class AdvectionUpstream(Package):
    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self):
        pass

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        d["scheme"] = "upstream"
        return self._template.render(d)



class AdvectionCentral(Package):
    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self):
        pass

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        d["scheme"] = "central"
        return self._template.render(d)


class AdvectionTVD(Package):

    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)
 

    def __init__(self):     
       pass

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        d["scheme"] = "TVD"
        return self._template.render(d) 
    