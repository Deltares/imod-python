import imod


def prepend_content_with_version_info(content: str) -> str:
    """
    Prepends file content with comment with iMOD Python version information.

    Parameters
    ----------
    content: str
        The file content to prepended.

    Returns
    -------
    str
        Content prepended with the version information.
    """
    version = imod.__version__

    return f"# File written with iMOD Python version: {version}\n\n{content}"
