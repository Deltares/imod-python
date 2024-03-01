"""
Conventional IDF filenames can be understood and constructed using
:func:`imod.util.path.decompose` and :func:`imod.util.path.compose`. These are used
automatically in :func:`imod.idf`.
"""
import datetime
import pathlib
import re
import tempfile
from typing import Any, Dict

import cftime
import numpy as np

from imod.util.time import _compose_timestring, to_datetime

Pattern = re.Pattern

def _custom_pattern_to_regex_pattern(pattern: str):
    """
    Compile iMOD Python's simplified custom pattern to regex pattern:
    _custom_pattern_to_regex_pattern({name}_c{species})
    is the same as calling:
    (?P<name>[\\w.-]+)_c(?P<species>[\\w.-]+)).compile()
    """
    pattern = pattern.lower()
    # Get the variables between curly braces
    in_curly = re.compile(r"{(.*?)}").findall(pattern)
    regex_parts = {key: f"(?P<{key}>[\\w.-]+)" for key in in_curly}
    # Format the regex string, by filling in the variables
    simple_regex = pattern.format(**regex_parts)
    return re.compile(simple_regex)


def _groupdict(stem: str, pattern: str | Pattern) -> Dict:
    if pattern is not None:
        if isinstance(pattern, Pattern):
            d = pattern.match(stem).groupdict()
        else:
            re_pattern = _custom_pattern_to_regex_pattern(pattern)
            # Use it to get the required variables
            d = re_pattern.match(stem).groupdict()
    else:  # Default to "iMOD conventions": {name}_c{species}_{time}_l{layer}
        has_layer = bool(re.search(r"_l\d+$", stem))
        has_species = bool(
            re.search(r"conc_c\d{1,3}_\d{8,14}", stem)
        )  # We are strict in recognizing species
        try:  # try for time
            base_pattern = r"(?P<name>[\w-]+)"
            if has_species:
                base_pattern += r"_c(?P<species>[0-9]+)"
            base_pattern += r"_(?P<time>[0-9-]{6,})"
            if has_layer:
                base_pattern += r"_l(?P<layer>[0-9]+)"
            re_pattern = re.compile(base_pattern)
            d = re_pattern.match(stem).groupdict()
        except AttributeError:  # probably no time
            base_pattern = r"(?P<name>[\w-]+)"
            if has_species:
                base_pattern += r"_c(?P<species>[0-9]+)"
            if has_layer:
                base_pattern += r"_l(?P<layer>[0-9]+)"
            re_pattern = re.compile(base_pattern)
            d = re_pattern.match(stem).groupdict()
    return d


def decompose(path, pattern: str = None) -> Dict[str, Any]:
    r"""
    Parse a path, returning a dict of the parts, following the iMOD conventions.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file. Upper case is ignored.
    pattern : str, regex pattern, optional
        If the path is not made up of standard paths, and the default decompose
        does not produce the right result, specify the used pattern here. See
        the examples below.

    Returns
    -------
    d : dict
        Dictionary with name of variable and dimensions

    Examples
    --------
    Decompose a path, relying on default conventions:

    >>> decompose("head_20010101_l1.idf")

    Do the same, by specifying a format string pattern, excluding extension:

    >>> decompose("head_20010101_l1.idf", pattern="{name}_{time}_l{layer}")

    This supports an arbitrary number of variables:

    >>> decompose("head_slr_20010101_l1.idf", pattern="{name}_{scenario}_{time}_l{layer}")

    The format string pattern will only work on tidy paths, where variables are
    separated by underscores. You can also pass a compiled regex pattern.
    Make sure to include the ``re.IGNORECASE`` flag since all paths are lowered.

    >>> import re
    >>> pattern = re.compile(r"(?P<name>[\w]+)L(?P<layer>[\d+]*)")
    >>> decompose("headL11", pattern=pattern)

    However, this requires constructing regular expressions, which is generally
    a fiddly process. The website https://regex101.com is a nice help.
    Alternatively, the most pragmatic solution may be to just rename your files.
    """
    path = pathlib.Path(path)
    # We'll ignore upper case
    stem = path.stem.lower()

    d = _groupdict(stem, pattern)
    dims = list(d.keys())
    # If name is not provided, generate one from other fields
    if "name" not in d.keys():
        d["name"] = "_".join(d.values())
    else:
        dims.remove("name")

    # TODO: figure out what to with user specified variables
    # basically type inferencing via regex?
    # if purely numerical \d* -> int or float
    #    if \d*\.\d* -> float
    # else: keep as string

    # String -> type conversion
    if "layer" in d.keys():
        d["layer"] = int(d["layer"])
    if "species" in d.keys():
        d["species"] = int(d["species"])
    if "time" in d.keys():
        d["time"] = to_datetime(d["time"])
    if "steady-state" in d["name"]:
        # steady-state as time identifier isn't picked up by <time>[0-9] regex
        d["name"] = d["name"].replace("_steady-state", "")
        d["time"] = "steady-state"
        dims.append("time")

    d["extension"] = path.suffix
    d["directory"] = path.parent
    d["dims"] = dims
    return d


def compose(d, pattern=None) -> pathlib.Path:
    """
    From a dict of parts, construct a filename, following the iMOD
    conventions.
    """
    haslayer = "layer" in d
    hastime = "time" in d
    hasspecies = "species" in d

    if pattern is None:
        if hastime:
            time = d["time"]
            d["timestr"] = "_{}".format(_compose_timestring(time))
        else:
            d["timestr"] = ""

        if haslayer:
            d["layerstr"] = "_l{}".format(int(d["layer"]))
        else:
            d["layerstr"] = ""

        if hasspecies:
            d["speciesstr"] = "_c{}".format(int(d["species"]))
        else:
            d["speciesstr"] = ""

        s = "{name}{speciesstr}{timestr}{layerstr}{extension}".format(**d)
    else:
        if hastime:
            time = d["time"]
            if time != "steady-state":
                # Change time to datetime.datetime
                if isinstance(time, np.datetime64):
                    d["time"] = time.astype("datetime64[us]").item()
                elif isinstance(time, cftime.datetime):
                    # Take first six elements of timetuple and convert to datetime
                    d["time"] = datetime.datetime(*time.timetuple()[:6])
        s = pattern.format(**d)

    if "directory" in d:
        return pathlib.Path(d["directory"]) / s
    else:
        return pathlib.Path(s)

def temporary_directory() -> pathlib.Path:
    tempdir = tempfile.TemporaryDirectory()
    return pathlib.Path(tempdir.name)
