import numpy as np
import pandas as pd
from imod.mf6.pkgbase import Package

class Dispersion(Package):
    """
    Molecular Diffusion and Dispersion.

    Parameters
    ----------
    diffusion_coefficient: xr.DataArray
        effective molecular diffusion coefficient (DIFFC) 
    longitudinal_horizontal: xu.UgridDataArray
        longitudinal dispersivity in horizontal direction. If flow is strictly
        horizontal, then this is the longitudinal dispersivity that will be
        used. If flow is not strictly horizontal or strictly vertical, then the
        longitudinal dispersivity is a function of both ALH and ALV. If
        mechanical dispersion is represented (by specifying any dispersivity
        values) then this array is required. (ALH) 
    transverse_horizontal1: xr.DataArray
        transverse dispersivity in horizontal direction. This is the transverse
        dispersivity value
         for the second ellipsoid axis. If flow is strictly horizontal and
         directed in the x direction (along a row for a regular grid), then this
         value controls spreading in the y direction. If mechanical dispersion
         is represented (by specifying any dispersivity values) then this array
         is required. (ATH1) 
    longitudinal_vertical: xr.DataArray, optional
        longitudinal dispersivity in vertical direction. If flow is strictly
        vertical, then this is the longitudinal dispsersivity value that will be
        used. If flow is not strictly horizontal or strictly vertical, then the
        longitudinal dispersivity is a function of both ALH and ALV. If this
        value is not specified and mechanical dispersion is represented, then
        this array is set equal to ALH. (ALV)          
    transverse_horizontal2: xr.DataArray, optional
        transverse dispersivity in horizontal direction. This is the transverse
        dispersivity value for the third ellipsoid axis. If flow is strictly
        horizontal and directed in the x direction (along a row for a regular
        grid), then this value controls spreading in the z direction. If this
        value is not specified and mechanical dispersion is represented, then
        this array is set equal to ATH1. (ATH2) 
    tranverse_vertical: xr.DataArray, optional
        transverse dispersivity when flow is in vertical direction. If flow is
        strictly vertical and directed in the z direction, then this value
        controls spreading in the x and y directions. If this value is not
        specified and mechanical dispersion is represented, then this array is
        set equal to ATH2. (ATV) 
    xt3d_off: bool, optional
        deactivate the xt3d method and use the faster and less accurate
        approximation. (XT3D_OFF) 
    xt3d_rhs: bool, optional 
        add xt3d terms to right-hand side, when possible. This option uses less
        memory, but may require more iterations.  (XT3D_RHS)         
    """    


    _pkg_id = "dsp"
    _template = Package._initialize_template(_pkg_id)
    _grid_data = {"diffusion_coefficient": np.float64, "longitudinal_horizontal": np.float64, "transversal_horizontal1": np.float64, "longitudinal_vertical": np.float64, "transversal_horizontal2": np.float64, "transversal_vertical": np.float64}

    _keyword_map = {"diffusion_coefficient": "diffc",
        "longitudinal_horizontal": "alh", 
        "transversal_horizontal1": "ath1",
        "longitudinal_vertical": "alv",
        "transversal_horizontal2": "ath2",
        "transversal_vertical": "atv" }

    def __init__(self, diffusion_coefficient, longitudinal_horizontal,  
    transversal_horizontal1, longitudinal_vertical = None,  transversal_horizontal2= None, 
    transversal_vertical=None , xt3d_off=False, xt3d_rhs=False ):     
        super().__init__(locals())
        self.dataset["XT3D_OFF"] = xt3d_off
        self.dataset["XT3D_RHS"] = xt3d_rhs        
        self.dataset["diffusion_coefficient"] = diffusion_coefficient
        self.dataset["longitudinal_horizontal"] = longitudinal_horizontal
        self.dataset["transversal_horizontal1"] = transversal_horizontal1
        if longitudinal_vertical is not None:
            self.dataset["longitudinal_vertical"] = longitudinal_vertical
        if transversal_horizontal2 is not None:                    
            self.dataset["transversal_horizontal2"] = transversal_horizontal2
        if transversal_vertical is not None:   
            self.dataset["transversal_vertical"] = transversal_vertical