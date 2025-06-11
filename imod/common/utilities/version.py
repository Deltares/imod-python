from typing import Optional

import imod
from imod.logging import LogLevel, logger


def get_version() -> str:
    """
    Returns the version of the iMOD Python package. This now is a trivial
    function, but it can be later expanded to also include git hashes with a
    package like setuptools-scm or hash-vcs.

    Returns
    -------
    str
        The version of the iMOD Python package.
    """
    return imod.__version__


def prepend_content_with_version_info(
    content: str,
    comment_char="#",
) -> str:
    """
    Prepends file content with comment with iMOD Python version information.

    Parameters
    ----------
    content: str
        The file content to prepended.
    comment_char: str, optional
        The character used for comments in the file. Defaults to "#".

    Returns
    -------
    str
        Content prepended with the version information.
    """
    version = get_version()
    line_text = "File written with iMOD Python version"

    return f"{comment_char} {line_text}: {version}\n\n{content}"


def log_versions(version_saved: Optional[dict[str, str]]) -> None:
    version = get_version()
    logger.log(LogLevel.INFO, f"iMOD Python version in current environment: {version}")
    if version_saved:
        version_msg = (
            f"iMOD Python version in dumped simulation: {version_saved['imod-python']}"
        )
    else:
        version_msg = "No iMOD Python version information found in dumped simulation."
    logger.log(
        LogLevel.INFO,
        version_msg,
    )
