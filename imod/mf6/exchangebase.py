from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr

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

    def render_with_geometric_constants(
        self,
        directory: Path,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        binary: bool,
    ) -> str:
        if hasattr(self, "_template"):
            template = self._template
        else:
            raise RuntimeError("exchange package does not have a template")

        d = Package._get_render_dictionary(
            self, directory, pkgname, globaltimes, binary
        )
        vars_to_render = {}
        index_dim = self.dataset["layer"].dims[0]
        for i in [1, 2]:
            vars_to_render[f"layer{i}"] = (index_dim, self.dataset["layer"].data)
            cell_id_dim = self.dataset[f"cell_id{i}"].dims[1]
            # length of cell_id_dims is 1 for unstructured and 2 for structured
            for j in range(self.dataset.sizes[cell_id_dim]):
                varname = f"cell_id{i}_{j + 1}"
                vars_to_render[varname] = (
                    index_dim,
                    self.dataset[f"cell_id{i}"].data[:, j],
                )

        all_geometric_vars = ["ihc", "cl1", "cl2", "hwva", "angldegx", "cdist"]
        for var in all_geometric_vars:
            if var in self.dataset.data_vars:
                vars_to_render[var] = (index_dim, self.dataset[var].data)
        datablock = xr.merge([vars_to_render], join="exact").to_dataframe()
        d["datablock"] = datablock.to_string(index=False, header=False)
        return template.render(d)
