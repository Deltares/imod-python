
from typing import List, Union
import numpy as np
import pandas as pd
from imod.mf6.package import Package
from imod.mf6.gwfgwf import GWFGWF
from imod.mf6.write_context import WriteContext
from numpy import ndarray
import jinja2



class exchanges ( Package):
    _keyword_map = {}
    _pkg_id = "gwfgwf"

    def __init__(self, exchanges: GWFGWF, print_input: bool, print_flows: bool,save_flows: bool, cell_averaging: bool, variablecv: bool,newton: bool ):
        
        self.dataset = exchanges._cell_id1.to_dataset()
        self.dataset["model_name_1"] = exchanges._model_name1
        self.dataset["model_name_2"] = exchanges._model_name2        
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["cell_averaging"] = cell_averaging
        self.dataset["variable_cv"] = variablecv
        self.dataset["newton"] = newton



    def render(self, directory, pkgname, globaltimes, binary):

        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        _template =  env.get_template("exg-gwfgwf.j2")

        d = {}
        d["exg"]
        for varname in self.dataset.data_vars:
            key = self._keyword_map.get(varname, varname)


            value = self[varname].values[()]
            if self._valid(value):  # skip False or None
                d[key] = value
        
        return _template.render(d)
    
    def model_name_1 (self)-> str:
        return         self.dataset["model_name_1"].values[()]      

    def model_name_2 (self)-> str:
        return         self.dataset["model_name_2"].values[()]      
    

    def filename(self) ->str:
        return f"{self.packagename() }.{self._pkg_id}"

    def packagename(self) ->str:
        return f"{self.model_name_1()}_{self.model_name_2() }"
        