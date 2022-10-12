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
