import numpy as np
import pandas as pd
from imod.mf6.pkgbase import Package

class Dispersion(Package):
    _pkg_id = "dsp"
    _template = Package._initialize_template(_pkg_id)
    _grid_data = {"diffc": np.float64, "alh": np.float64, "ath1": np.float64, "alv": np.float64, "ath2": np.float64, "atv": np.float64}

    def __init__(self, xt3dOff , xt3dRHS, diffusion_coefficient, longitudinal_horizontal_dispersivity,  
    transversal_horizontal1_dispersivity, longitudinal_vertical_dispersivity = [],  transversal_horizontal2_dispersivity= [], 
    transversal_vertical_dispersivity= []  ):     
        super().__init__()
        self.dataset["XT3D_OFF"] = xt3dOff
        self.dataset["XT3D_RHS"] = xt3dRHS        
        self.dataset["diffc"] = diffusion_coefficient
        self.dataset["alh"] = longitudinal_horizontal_dispersivity
        self.dataset["ath1"] = transversal_horizontal1_dispersivity
        if longitudinal_vertical_dispersivity != []:
            self.dataset["alv"] = longitudinal_vertical_dispersivity
        if transversal_horizontal2_dispersivity != []:                    
            self.dataset["ath2"] = transversal_horizontal2_dispersivity
        if transversal_vertical_dispersivity != []:   
            self.dataset["atv"] = transversal_vertical_dispersivity
    

            