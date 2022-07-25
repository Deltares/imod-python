import numpy as np
from imod.mf6.pkgbase import BoundaryCondition
from imod.mf6 import GroundwaterFlowModel


def with_index_dim(array_like):
    # At least1d will also extract the values if array_like is a DataArray.
    arr1d = np.atleast_1d(array_like)
    if arr1d.ndim > 1:
        raise ValueError("array must be 1d")
    return ("index", arr1d)


class SourceSinkMixing(BoundaryCondition):
    """
    Parameters
    ----------
    package_names: array_like of str
    concentration_boundary_type: array_like of str
    auxiliary_variable_name: array_like of str
    print_flows: bool
    save_flows: bool
    """
    _pkg_id = "ssm"
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        package_names,
        concentration_boundary_type,
        auxiliary_variable_name,
        print_flows:bool = False,
        save_flows:bool = False,
    ):
        super().__init__()
        # By sharing the index, this will raise an error if lengths do not
        # match.
        self.dataset["package_names"] = with_index_dim(package_names)
        self.dataset["concentration_boundary_type"] = with_index_dim(concentration_boundary_type)
        self.dataset["auxiliary_variable_name"] = with_index_dim(auxiliary_variable_name)
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows

    def render(self, directory, pkgname, globaltimes, binary):
        d = {
            "print_flows": self._valid(self["print_flows"].values[()]),    
            "save_flows": self._valid(self["save_flows"].values[()]),    
            "sources": [(a, b, c) for a, b, c in zip(
                self["package_names"].values,
                self["concentration_boundary_type"].values,
                self["auxiliary_variable_name"].values,
            )]
        }
        return self._template.render(d)

    @staticmethod
    def from_flow_model(model: GroundwaterFlowModel, species: str):
        """
        Derive a Source and Sink Mixing package from a Groundwater Flow model.
        """
        if not isinstance(model, GroundwaterFlowModel):
            raise TypeError(
                "model must be a GroundwaterFlowModel, received instead: "
                f"{type(model).__name__}"
            )

        names = []
        boundary_types = []
        aux_var_names = []
        for name, package in model.items():
            if isinstance(package, BoundaryCondition):
                ds = package.dataset
                # The package should contain a concentration variable, with a
                # species coordinate.
                if "concentration" not in ds.data_vars:
                    raise ValueError(f"concentration not present in package {name}")
                if "species" not in ds["concentration"].coords:
                    raise ValueError(
                        f"No species coordinate for concentration in package {name}"
                    )

                # While somewhat far-fetched, it is possible for different
                # species to have different mixing behavior.
                type_da = ds["concentration_boundary_type"] 
                if "species" in type_da.dims:
                    type_da = type_da.sel(species=species)
                 
                names.append(name)
                boundary_types.append(type_da.values[()])
                aux_var_names.append(species)
                
        if len(names) == 0:
            raise ValueError("flow model does not contain boundary conditions")

        return SourceSinkMixing(names, boundary_types, aux_var_names)
