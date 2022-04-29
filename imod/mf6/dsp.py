import numpy as np
import pandas as pd
from imod.mf6.pkgbase import Package

class Dispersion(Package):
    _pkg_id = "dsp"
    _template = Package._initialize_template(_pkg_id)
    _grid_data = {"diffc": np.float64, "alh": np.float64, "ath1": np.float64, "alv": np.float64, "ath2": np.float64, "atv": np.float64}

    def __init__(self, xt3dOff , xt3dRHS, diffusion_coefficient, longitudinal_horizontal_dispersivity,  
    transversal_horizontal1_dispersivity, longitudinal_vertical_dispersivity = None,  transversal_horizontal2_dispersivity= None, 
    transversal_vertical_dispersivity=None  ):     
        super().__init__(locals())
        self.dataset["XT3D_OFF"] = xt3dOff
        self.dataset["XT3D_RHS"] = xt3dRHS        
        self.dataset["diffc"] = diffusion_coefficient
        self.dataset["alh"] = longitudinal_horizontal_dispersivity
        self.dataset["ath1"] = transversal_horizontal1_dispersivity
        if longitudinal_vertical_dispersivity != None:
            self.dataset["alv"] = longitudinal_vertical_dispersivity
        if transversal_horizontal2_dispersivity != None:                    
            self.dataset["ath2"] = transversal_horizontal2_dispersivity
        if transversal_vertical_dispersivity != None:   
            self.dataset["atv"] = transversal_vertical_dispersivity

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        dspdirectory = directory / "dsp"
        if self.dataset["XT3D_OFF"]:
            d["XT3D_OFF"]= self.dataset["XT3D_OFF"]
        if self.dataset["XT3D_RHS"]:            
            d["XT3D_RHS"]= self.dataset["XT3D_RHS"]

        for varname in ["diffc", "alh", "ath1", "alv", "ath2", "atv"]:
            if varname in self.dataset.keys():
                layered, value = self._compose_values(
                    self[varname], dspdirectory, varname, binary=binary
                )
                if self._valid(value):  # skip False or None
                    d[f"{varname}_layered"], d[varname] = layered, value

        return self._template.render(d)
        
    

            