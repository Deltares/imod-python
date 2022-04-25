import numpy as np
import pandas as pd
from imod.mf6.pkgbase import Package

class DifDisp(Package):
    _pkg_id = "dsp"
    _template = Package._initialize_template(_pkg_id)

    _xt3d_off = False
    _xt3d_rhs = False

    def __init__(self, diffc, alh,  ath1, alv = [], ath2= [], atv= [] ):     
        super().__init__()
        self.dataset["diffc"] = diffc
        self.dataset["alh"] = alh
        self.dataset["ath1"] = ath1
        if alv != []:
            self.dataset["alv"] = alv
        if ath2 != []:                    
            self.dataset["ath2"] = ath2
        if atv != []:   
            self.dataset["atv"] = atv

    def SetXT3DOff(self, trueOrFalse):
        self._xt3d_off = trueOrFalse

    def SetXT3DRhs(self, trueOrFalse):
        self._xt3d_rhs = trueOrFalse

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        dspdirectory = directory / "dsp"

        if (self._xt3d_off ):
            self.dataset["XT3D_OFF"]=self._xt3d_off
        if (self._xt3d_rhs ):
            self.dataset["XT3D_RHS"]=self._xt3d_rhs

        for varname in ["diffc", "alh", "ath1", "alv", "ath2", "atv"]:
            if varname in self.dataset.keys():
                layered, value = self._compose_values(
                    self[varname], dspdirectory, varname, binary=binary
                )
                if self._valid(value):  # skip False or None
                    d[f"{varname}_layered"], d[varname] = layered, value

        return self._template.render(d)
    

            