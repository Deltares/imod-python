def get_variable_names(package):
    # now we should add the auxiliary variable names to d
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
                # the error message is more specific than the code at this point.
                raise ValueError(
                    f"{auxvar} requires a {auxiliaries[auxvar]} coordinate."
                )
    return []


def _get_auxiliary_data_variable_names_mapping(package):
    result = {}
    if hasattr(package, "_auxiliary_data"):
        result.update(package._auxiliary_data)
    return result


def add_periodic_auxiliary_variable(package):
    if hasattr(package, "_auxiliary_data"):
        for aux_var_name, aux_var_dimensions in package._auxiliary_data.items():
            aux_coords = package.dataset[aux_var_name].coords[aux_var_dimensions].values
            for s in aux_coords:
                package.dataset[s] = package.dataset[aux_var_name].sel(
                    {aux_var_dimensions: s}
                )
