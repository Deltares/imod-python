import numpy as np
import pandas as pd
from imod.mf6.pkgbase import Package

class Dispersion(Package):
    """
    Molecular Diffusion and Dispersion.

    Parameters
    ----------
    xt3dOff: deactivate the xt3d method and use the faster and less accurate approximation. (XT3D_OFF) (Bool)
    xt3dRHS:  add xt3d terms to right-hand side, when possible. This option uses less memory, 
        but may require more iterations.  (XT3D_RHS) (Bool)
    diffusion_coefficient: effective molecular diffusion coefficient (DIFFC) (xu.UgridDataArray)
    longitudinal_horizontal: longitudinal dispersivity in horizontal direction. If flow is strictly horizontal,
         then this is the longitudinal dispersivity that will be used. If flow is not strictly horizontal or strictly
         vertical, then the longitudinal dispersivity is a function of both ALH and ALV. If mechanical dispersion is 
         represented (by specifying any dispersivity values) then this array is required. (ALH) (xu.UgridDataArray)
    longitudinal_vertical: longitudinal dispersivity in vertical direction. If flow is strictly vertical, then this is the longitudinal 
        dispsersivity value that will be used. If flow is not strictly horizontal or strictly vertical, then the 
        longitudinal dispersivity is a function of both ALH and ALV. If this value is not specified and mechanical 
        dispersion is represented, then this array is set equal to ALH. (ALV) (xu.UgridDataArray)
    transverse_horizontal1: transverse dispersivity in horizontal direction. This is the transverse dispersivity value
         for the second ellipsoid axis. If flow is strictly horizontal and directed in the x direction (along a row 
         for a regular grid), then this value controls spreading in the y direction. If mechanical dispersion is 
         represented (by specifying any dispersivity values) then this array is required. (ATH1) (xu.UgridDataArray)
    transverse_horizontal2: transverse dispersivity in horizontal direction. This is the transverse dispersivity value 
        for the third ellipsoid axis. If flow is strictly horizontal and directed in the x direction (along a 
        row for a regular grid), then this value controls spreading in the z direction. If this value is not specified 
        and mechanical dispersion is represented, then this array is set equal to ATH1. (ATH2) (xu.UgridDataArray)
    tranverse_vertical:  transverse dispersivity when flow is in vertical direction. If flow is strictly vertical and 
        directed in the z direction, then this value controls spreading in the x and y directions. If this value is 
        not specified and mechanical dispersion is represented, then this array is set equal to ATH2. (ATV) (xu.UgridDataArray)
    """    


    _pkg_id = "dsp"
    _template = Package._initialize_template(_pkg_id)
    _grid_data = {"diffc": np.float64, "alh": np.float64, "ath1": np.float64, "alv": np.float64, "ath2": np.float64, "atv": np.float64}

    _keyword_map = {"diffusion_coefficient": "diffc",
        "longitudinal_horizontal": "alh", 
        "transversal_horizontal1": "ath1",
        "longitudinal_vertical": "alv",
        "transversal_horizontal2": "ath2",
        "transversal_vertical": "atv" }

    def __init__(self, xt3dOff , xt3dRHS, diffusion_coefficient, longitudinal_horizontal,  
    transversal_horizontal1, longitudinal_vertical = None,  transversal_horizontal2= None, 
    transversal_vertical=None  ):     
        super().__init__(locals())
        self.dataset["XT3D_OFF"] = xt3dOff
        self.dataset["XT3D_RHS"] = xt3dRHS        
        self.dataset["diffusion_coefficient"] = diffusion_coefficient
        self.dataset["longitudinal_horizontal"] = longitudinal_horizontal
        self.dataset["transversal_horizontal1"] = transversal_horizontal1
        if longitudinal_vertical != None:
            self.dataset["longitudinal_vertical"] = longitudinal_vertical
        if transversal_horizontal2 != None:                    
            self.dataset["transversal_horizontal2"] = transversal_horizontal2
        if transversal_vertical != None:   
            self.dataset["transversal_vertical"] = transversal_vertical

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        dspdirectory = directory / "dsp"
        if self.dataset["XT3D_OFF"]:
            d["XT3D_OFF"]= self.dataset["XT3D_OFF"]
        if self.dataset["XT3D_RHS"]:            
            d["XT3D_RHS"]= self.dataset["XT3D_RHS"]

        for varname in ["diffusion_coefficient", "longitudinal_horizontal", "transversal_horizontal1", "longitudinal_vertical", "transversal_horizontal2", "transversal_vertical"]:
            if varname in self.dataset.keys():
                trueName =self. _keyword_map[varname]
                layered, value = self._compose_values(
                    self[varname], dspdirectory, trueName, binary=binary
                )
                if self._valid(value):  # skip False or None
                    d[f"{trueName}_layered"], d[trueName] = layered, value

        return self._template.render(d)
        
    

            