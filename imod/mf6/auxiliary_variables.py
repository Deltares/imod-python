from imod.mf6.interfaces.ipackage import IPackage


def get_variable_names(package: IPackage):
    auxiliaries = _get_auxiliary_data_variable_names_mapping(
        package
    )  # returns something like {"concentration": "species"}

    # loop over the types of auxiliary variables (for example concentration)
    for auxvar in auxiliaries.keys():
        # if "concentration" is a variable of this dataset
        if auxvar in package.dataset.data_vars:
            # if our concentration dataset has the species coordinate
            if auxiliaries[auxvar] in package.dataset[auxvar].coords:
                # assign the species names list to d
                return package.dataset[auxiliaries[auxvar]].values.tolist()
            else:
                raise ValueError(
                    f"{auxvar} requires a {auxiliaries[auxvar]} coordinate."
                )
    return []


def _get_auxiliary_data_variable_names_mapping(package: IPackage):
    result = {}
    if hasattr(package, "_auxiliary_data"):
        result.update(package._auxiliary_data)
    return result


def add_periodic_auxiliary_variable(package: IPackage) -> None:
    """
    Splits an auxiliary dataarray (with one or more auxiliary variable dimension) into dataarrays per
    auxiliary variable dimension. For example a concentration auxiliary variable in a flow package
    will have a species dimension, and will be split in several dataarrays- one for each species.
    """

    for aux_var_name, aux_var_dimensions in package.auxiliary_data_fields().items():
        aux_coords = package.dataset[aux_var_name].coords[aux_var_dimensions].values
        for s in aux_coords:
            package.dataset[s] = package.dataset[aux_var_name].sel(
                {aux_var_dimensions: s}
            )


def remove_periodic_auxiliary_variable(package: IPackage) -> None:
    """
    Removes the data arrays created by :meth:add_periodic_auxiliary_variable(...) but does not
    remove the auxiliary dataarray used as source for :meth:add_periodic_auxiliary_variable(...)
    """
    for aux_var_name, aux_var_dimensions in package.auxiliary_data_fields().items():
        if aux_var_dimensions in package.dataset.coords:
            for species in package.dataset.coords[aux_var_dimensions].values:
                if species in list(package.dataset.keys()):
                    package.dataset = package.dataset.drop_vars(species)


def has_auxiliary_variable_source_array(package: IPackage) -> bool:
    """
    Returns True if a package contains an auxiliary data source data array
    """
    for aux_var_name, _ in package.auxiliary_data_fields().items():
        if aux_var_name in package.dataset.keys():
            return True
    return False
