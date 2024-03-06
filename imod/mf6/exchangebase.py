from pathlib import Path
from typing import  Union

import numpy as np
import pandas as pd

from imod.mf6.package import Package

_pkg_id_to_type = {"gwfgwf": "GWF6-GWF6", "gwfgwt": "GWF6-GWT6", "gwtgwt": "GWT6-GWT6"}


class ExchangeBase(Package):
    """
    Base class for all the exchanges.
    This class enables writing the exchanges to file in a uniform way.
    """

    _keyword_map: dict[str, str] = {}

    @property
    def model_name1(self) -> str:
        if "model_name_1" not in self.dataset:
            raise ValueError("model_name_1 not present in dataset")
        return self.dataset["model_name_1"].values[()].take(0)

    @property
    def model_name2(self) -> str:
        if "model_name_2" not in self.dataset:
            raise ValueError("model_name_2 not present in dataset")
        return self.dataset["model_name_2"].values[()].take(0)

    def package_name(self) -> str:
        return f"{self.model_name1}_{self.model_name2}"

    def get_specification(self) -> tuple[str, str, str, str]:
        """
        Returns a tuple containing the exchange type, the exchange file name, and the model names. This can be used
        to write the exchange information in the simulation .nam input file
        """
        filename = f"{self.package_name()}.{self._pkg_id}"
        return (
            _pkg_id_to_type[self._pkg_id],
            filename,
            self.model_name1,
            self.model_name2,
        )
    
    def render_with_geometric_constants(self,  directory: Path, pkgname: str, globaltimes: Union[list[np.datetime64], np.ndarray], binary: bool) -> str:
        
        if hasattr(self, "_template"):
            template = self._template
        else:
            raise RuntimeError("exchange package does not have a template")
        
        d = Package._get_render_dictionary(self, directory, pkgname, globaltimes, binary)
        
        datablock = pd.DataFrame()
        datablock["layer1"] =  self.dataset["layer"].values[:] 

        # If the grid is structured, the cell_id arrays will have both a row and a column dimension, 
        # but if it is unstructured, it will have only a cellid dimension
        is_structured = len(self.dataset["cell_id1"].shape) > 1
        is_structured = is_structured and self.dataset["cell_id1"].shape[1] > 1

        datablock["cell_id1_1"] = self.dataset["cell_id1"].values[:,0]  
        if is_structured:
             datablock["cell_id1_2"] = self.dataset["cell_id1"].values[:,1]
        datablock["layer2"] =  self.dataset["layer"].values[:]  
        datablock["cell_id2_1"] = self.dataset["cell_id2"].values[:,0]
        if is_structured: 
             datablock["cell_id2_2"] = self.dataset["cell_id2"].values[:,1]

        for key in ["ihc", "cl1", "cl2", "hwva", "angldegx", "cdist" ]:
            if key in  self.dataset.keys():
                  datablock[key] = self.dataset[key].values[:]      
                                      
        d["datablock"] = datablock.to_string(index=False,  header=False )
        return template.render(d)        
