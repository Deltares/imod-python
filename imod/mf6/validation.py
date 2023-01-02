from imod.schemata import DimsSchema

# Template schemata to avoid code duplication
PKG_DIMS_SCHEMA = (
    DimsSchema("layer", "y", "x")
    | DimsSchema("layer", "{face_dim}")
    | DimsSchema("layer")
    | DimsSchema()
)

BC_DIMS_SCHEMA = (
    DimsSchema("time", "layer", "y", "x")
    | DimsSchema("layer", "y", "x")
    | DimsSchema("time", "layer", "{face_dim}")
    | DimsSchema("layer", "{face_dim}")
    # Layer dim not necessary, as long as there is a layer coordinate
    # present
    | DimsSchema("time", "y", "x")
    | DimsSchema("y", "x")
    | DimsSchema("time", "{face_dim}")
    | DimsSchema("{face_dim}")
)


def validation_model_error_message(model_errors):
    messages = []
    for name, pkg_errors in model_errors.items():
        pkg_header = f"{name}\n" + len(name) * "-" + "\n"
        messages.append(pkg_header)
        messages.append(validation_pkg_error_message(pkg_errors))
    return "\n" + "\n".join(messages)


def validation_pkg_error_message(pkg_errors):
    messages = []
    for var, var_errors in pkg_errors.items():
        messages.append(f"* {var}")
        messages.extend(f"\t- {error}" for error in var_errors)
    return "\n" + "\n".join(messages)
