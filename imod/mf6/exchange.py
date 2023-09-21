
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
        self.dataset["cell_id2"] = exchanges._cell_id2
        self.dataset["layer"] = exchanges._layer
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
        d["nexg"] = len(self.dataset["cell_id1"].values[0])
        for varname in self.dataset.data_vars:
            key = self._keyword_map.get(varname, varname)


            value = self[varname].values[()]
            if self._valid(value):  # skip False or None
                d[key] = value
        #check dimensionality of cell_1d - it has 2 columns for structured grids and 1 column for unstructured grids
        if (self.dataset["cell_id1"].shape[0]==2):
            exchangeblock = {"layer1" : self.dataset["layer"].transpose(), 
                            "row1": self.dataset["cell_id1"].values.transpose()[:,0]  ,
                            "col1": self.dataset["cell_id1"].values.transpose()[:,1],
                            "layer2" : self.dataset["layer"].transpose(),
                            "row2":   self.dataset["cell_id2"].values.transpose()[:,0]  ,
                            "col2":   self.dataset["cell_id2"].values.transpose()[:,1] }
        elif (self.dataset["cell_id2"].shape[0]==1):
            exchangeblock = {"layer1" : self.dataset["layer"].transpose(), 
                            "cellindex1": self.dataset["cell_id1"].values.transpose()[:,0]  ,
                            "layer2" : self.dataset["layer"].transpose(),
                            "cellindex2":   self.dataset["cell_id2"].values.transpose()[:,0] }
      
        d["exchangesblock"] = pd.DataFrame(exchangeblock).to_csv(sep=" ", header=False, index=False, line_terminator = '\r')
        
        return _template.render(d)
    
    def model_name_1 (self)-> str:
        return         self.dataset["model_name_1"].values[()]      

    def model_name_2 (self)-> str:
        return         self.dataset["model_name_2"].values[()]      
    

    def filename(self) ->str:
        return f"{self.packagename() }.{self._pkg_id}"

    def packagename(self) ->str:
        return f"{self.model_name_1()}_{self.model_name_2() }"
        