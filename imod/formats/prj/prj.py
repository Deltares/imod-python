"""
Utilities for parsing a project file.
"""
import shlex
from collections import defaultdict
from datetime import datetime
from itertools import chain
from os import PathLike
from typing import Any, Dict, List, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

import imod

FilePath = Union[str, "PathLike[str]"]


KEYS = {
    "(bnd)": ("ibound",),
    "(top)": ("top",),
    "(bot)": ("bottom",),
    "(thk)": ("thickness",),
    "(khv)": ("kh",),
    "(kva)": ("vertical_anisotropy",),
    "(kdw)": ("transmissivity",),
    "(kvv)": ("kv",),
    "(vcw)": ("resistance",),
    "(shd)": ("head",),
    "(sto)": ("storage_coefficient",),
    "(spy)": ("specific_yield",),
    "(por)": ("porosity",),
    "(ani)": ("factor", "angle"),
    "(hfb)": ("gen",),
    "(ibs)": (None),
    "(pwt)": (None),
    "(sft)": (None),
    "(obs)": (None),
    "(cbi)": (None),
    "(sco)": (None),
    "(dsp)": (None),
    "(ics)": (None),
    "(fcs)": (None),
    "(ssc)": (None),
    "(fod)": (None),
    "(fos)": (None),
    "(rct)": (None),
    "(con)": (None),
    "(pst)": (None),
}

DATE_KEYS = {
    "(uzf)": (None,),
    "(rch)": ("rate",),
    "(evt)": ("rate", "extinction_depth"),
    "(drn)": ("conductance", "elevation"),
    "(olf)": ("elevation",),
    "(riv)": ("conductance", "stage", "bottom_elevation", "infiltration_factor"),
    "(isg)": ("isg",),
    "(sfr)": ("isg",),
    "(lak)": (None,),
    "(wel)": ("ipf",),
    "(mnw)": (None,),
    "(ghb)": ("conductance", "head"),
    "(chd)": ("head",),
    "(fhb)": (None,),
    "(fde)": (None,),
    "(tvc)": (None,),
}

METASWAP_VARS = (
    "boundary",
    "landuse",
    "rootzone_thickness",
    "soil_physical_unit",
    "meteo_station_number",
    "surface_elevation",
    "artificial_recharge",
    "artifical_recharge_layer",
    "artificial_recharge_capacity",
    "wetted_area",
    "urban_area",
    "urban_ponding_depth",
    "rural_ponding_depth",
    "urban_runoff_resistance",
    "rural_runoff_resistance",
    "urban_runon_resistance",
    "rural_runon_resistance",
    "urban_infiltration_capacity",
    "rural_infiltration_capacity",
    "perched_water_table_level",
    "soil_moisture_fraction",
    "conductivitiy_factor",
    "plot_number",
    "steering_location",
    "plot_drainage_level",
    "plot_drainage_resistance",
)


class _LineIterator:
    """
    Like iter(lines), but we can go back and we check if we're finished.
    """

    def __init__(self, lines: List[List[str]]):
        self.lines = lines
        self.count = -1
        self.length = len(lines)

    def __iter__(self):
        return self

    def __next__(self) -> List[str]:
        if self.finished:
            raise StopIteration
        self.count += 1
        return self.lines[self.count]

    def back(self) -> None:
        self.count = max(self.count - 1, -1)

    @property
    def finished(self) -> bool:
        return (self.count + 1) >= self.length


def _tokenize(line: str) -> List[str]:
    """
    A value separator in Fortran list-directed input is:

    * A comma if period decimal edit mode is POINT.
    * One or more contiguous spaces (blanks); no tabs.

    Other remarks:

    * Values, except for character strings, cannot contain blanks.
    * Strings may be unquoted if they do not start with a digit and no value
      separators.
    * Character strings can be quoted strings, using pairs of quotes ("), pairs
      of apostrophes (').
    * A quote or apostrophe must be preceded by a value separator to initite a
      quoted string.
    * An empty entry consists of two consecutive commas (or semicolons).

    For the use here (parsing IMOD's project files), we ignore:

    * A semicolon value separator if period decimal edit mode is COMMA.
    * Complex constants given as two real constants separated by a comma and
      enclosed in parentheses.
    * Repetition counts: 4*(3.,2.) 2*, 4*'hello'

    Furthermore, we do not expect commas inside of the project file entries,
    since we expect:

    * Package names: unquoted character strings.
    * File paths: will not contain commas, no single apostrophe, nor a single
      quote symbol, may contain whitespace if quoted. * Integers for counts and
      settings.
    * Floats for addition and multiplication values.
    * Simple character strings for period names (summer, winter). These
      technically could contain commas if quoted, which is very unlikely.
    * No quotes or apostrophes are escaped.

    With these assumptions, we can limit complexity considerably (see the
    PyLiDiRe link for a complete implementation):

    * First we split by comma (we don't expected commas in quote strings).
    * Next we split by whitespace, unless quoted.

    We can expect both single and double quotes, even within a single line:
    shlex.split() handles this. Note that additional entries are likely
    allowed, as the Fortran implementation only reads what is necessary,
    then stops parsing.

    See also:
        * https://stackoverflow.com/questions/36165050/python-equivalent-of-fortran-list-directed-input
        * https://gitlab.com/everythingfunctional/PyLiDiRe
        * https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vnc5/index.html
        * The Fortran 2003 Handbook

    Examples
    --------

    Raise ValueError, due to missing closing quotation. (Can be enabled
    shlex.split(s, posix=False)):

    >> _tokenize("That's life")

    >> _tokenize("That 's life'")
    >> ["That", "s life"]

    >> _tokenize("That,'s life'")
    >> ["That", "s life"]
    """
    values = [v.strip() for v in line.split(",")]
    tokens = list(chain.from_iterable(shlex.split(v) for v in values))
    return tokens


def _wrap_error_message(
    exception: Exception, description: str, lines: _LineIterator
) -> None:
    lines.back()
    content = next(lines)
    number = lines.count + 1
    raise type(exception)(
        f"{exception}\n"
        f"Failed to parse {description} for line {number} with content:\n{content}"
    )


def _parse_blockheader(lines: _LineIterator) -> Tuple[int, str, str]:
    try:
        no_result = None, None, None
        line = next(lines)

        # Skip if it's an empty line.
        if len(line) == 0:
            return no_result

        first = line[0].lower()
        if first in ("periods", "species"):
            return 1, first, None
        # The line must contain atleast nper, key, active.
        elif len(line) >= 3:
            n = int(first)
            key = line[1].lower()
            active = line[2]
            return n, key, active
        # It's a comment or something.
        else:
            return no_result
    except Exception as e:
        _wrap_error_message(e, "block header", lines)


def _parse_time(lines: _LineIterator) -> str:
    try:
        line = next(lines)
        date = line[0].lower()

        if date == "steady-state":
            return date

        elif len(line) > 1:
            time = line[1]
        else:
            time = "00:00:00"
        return f"{date} {time}"
    except Exception as e:
        _wrap_error_message(e, "date time", lines)


def _parse_blockline(lines: _LineIterator, time: str = None) -> Dict[str, Any]:
    try:
        line = next(lines)
        content = {
            "active": bool(int(line[0])),
            "is_constant": int(line[1]),
            "layer": int(line[2]),
            "factor": float(line[3]),
            "addition": float(line[4]),
            "constant": float(line[5]),
        }
        if content["is_constant"] == 2:
            content["path"] = line[6]
        if time is not None:
            content["time"] = time
        return content
    except Exception as e:
        _wrap_error_message(e, "entries", lines)


def _parse_nsub_nsystem(lines: _LineIterator) -> Tuple[int, int]:
    try:
        line = next(lines)
        n_entry = int(line[0])
        n_system = int(line[1])
        return n_entry, n_system
    except Exception as e:
        _wrap_error_message(e, "number of sub-entries and number of systems", lines)


def _parse_notimeblock(
    lines: _LineIterator,
    fields: List[str],
) -> Dict[str, Any]:
    n_entry, n_system = _parse_nsub_nsystem(lines)

    if len(fields) != n_entry:
        raise ValueError(
            f"Expected NSUB entry of {len(fields)} for {fields}, read: {n_entry}"
        )
    content = {
        field: [_parse_blockline(lines) for _ in range(n_system)] for field in fields
    }
    content["n_system"] = n_system
    return content


def _parse_capblock(
    lines: _LineIterator,
) -> Dict[str, Any]:
    fields = METASWAP_VARS
    n_entry, n_system = _parse_nsub_nsystem(lines)

    if n_entry == 21:
        # Remove layer entry.
        fields = list(fields[:22]).pop(8)
    elif n_entry == 22:
        fields = fields[:22]
    elif n_entry == 26:
        pass
    else:
        raise ValueError(
            f"Expected NSUB entry of 21, 22, or 26 for {fields}, read: {n_entry}"
        )

    content = {
        field: [_parse_blockline(lines) for _ in range(n_system)] for field in fields
    }
    content["n_system"] = n_system
    return content


def _parse_extrablock(lines: _LineIterator, n: int) -> Dict[str, List[str]]:
    """Parse the MetaSWAP "extra files" block"""
    return {"paths": [next(lines) for _ in range(n)]}


def _parse_timeblock(
    lines: List[str],
    fields: List[str],
    n: int,
) -> Dict[str, Any]:
    n_fields = len(fields)
    content = defaultdict(list)
    for _ in range(n):
        time = _parse_time(lines)
        content["time"].append(time)
        n_entry, n_system = _parse_nsub_nsystem(lines)

        if n_fields != n_entry:
            raise ValueError(
                f"Expected NSUB entry of {n_fields} for {fields}, read: {n_entry}"
            )

        for field in fields:
            content[field].extend(
                [_parse_blockline(lines, time) for _ in range(n_system)]
            )

    content["n_system"] = n_system
    return content


def _parse_pcgblock(lines: _LineIterator) -> Dict[str, Any]:
    try:
        line = next(lines)

        # TODO: which are optional? How many to expect?
        # Check for an empty line to terminate the block?
        types = {
            "mxiter": int,
            "iter1": int,
            "hclose": float,
            "rclose": float,
            "relax": float,
            "npcond": int,
            "iprpcg": int,
            "mutpcg": int,
            "damppcg": float,
            "damppcgt": float,
            "iqerror": int,
            "qerror": float,
        }

        if len(line) == 12:
            line_iterator = iter(line)
            content = {
                k: valuetype(next(line_iterator)) for k, valuetype in types.items()
            }
        elif any("=" in s for s in line):
            pcglines = [line] + [next(lines) for _ in range(11)]
            content = {}
            for line in pcglines:
                # undo separation, partition on equality sign instead.
                line = "".join(line)
                key, _, value = line.lower().partition("=")
                value = types[key](value)
                content[key] = value
        else:
            raise ValueError(
                f"Expected 12 KEY = VALUE pairs, or 12 values. Found {len(line)}"
            )

        return content
    except Exception as e:
        _wrap_error_message(e, "PCG entry", lines)


def _parse_periodsblock(lines: _LineIterator) -> Dict[str, str]:
    try:
        periods = {}
        while not lines.finished:
            line = next(lines)
            # Stop if we encounter an empty line.
            if len(line) == 0:
                break
            # Read the alias
            alias = line[0]
            # Now read the time associated with it.
            start = _parse_time(lines)
            periods[alias] = start
        return periods
    except Exception as e:
        _wrap_error_message(e, "periods data block", lines)


def _parse_block(lines: _LineIterator, content: Dict[str, Any]) -> None:
    """
    Mutates content dict.
    """
    n = key = active = None

    # A project file may contain any number of lines outside of a "topic"
    # block. _parse_blockheader will return triple None in that case.
    while key is None and not lines.finished:
        n, key, active = _parse_blockheader(lines)

    try:
        if key in KEYS:
            if n != 1:
                raise ValueError(f"Expected N=1 for {key}, read: {n}")
            fields = KEYS[key]
            blockcontent = _parse_notimeblock(lines, fields)
        elif key in DATE_KEYS:
            fields = DATE_KEYS[key]
            blockcontent = _parse_timeblock(lines, fields, n)
        elif key == "(cap)":
            blockcontent = _parse_capblock(lines)
        elif key == "(pcg)":
            blockcontent = _parse_pcgblock(lines)
        elif key == "periods":
            blockcontent = _parse_periodsblock(lines)
        elif key == "extra":
            blockcontent = _parse_extrablock(lines, n)
        else:
            other = ("(pcg)", "(gcg)", "(vdf)")
            options = tuple(KEYS.keys()) + tuple(DATE_KEYS.keys()) + other
            lines.back()
            line = next(lines)
            number = lines.count + 1
            raise ValueError(
                f"Failed to recognize header keyword: {key}. Expected one of keywords {options}"
                f"\nErrored in line {number} with entries:\n{line}"
            )

    except Exception as e:
        raise type(e)(f"{e}\nError occurred for keyword: {key}")

    if blockcontent is not None and active is not None:
        blockcontent["active"] = active

    content[key] = blockcontent
    return


def _process_package_entry(entry: Dict):
    """
    The iMOD project file supports constants in lieu of IDFs.
    """
    coords = {"layer": entry["layer"]}
    dims = ("layer",)

    if "path" not in entry:
        path = None
        header = {"coords": coords}
        value = entry["constant"]
    else:
        path = entry["path"]
        header = imod.idf.header(path, pattern="{name}")
        value = None

    header["dims"] = dims
    return path, header, value


def _merge_coords(headers: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    coords = defaultdict(list)
    for header in headers:
        for key, value in header["coords"].items():
            coords[key].append(value)
    return {k: np.unique(coords[k]) for k in coords}


def _create_dataarray(
    paths: List[str], headers: List[Dict[str, Any]], values: List[float]
) -> xr.DataArray:
    """
    Create a DataArray from a list of IDF paths, or from constant values.
    """
    none_paths = [p is None for p in paths]
    if all(none_paths):
        coords = _merge_coords(headers)
        da = xr.DataArray(values, dims=headers[0]["dims"], coords=coords)
    elif any(none_paths):
        raise NotImplementedError(
            "Entries for a system should either all provide a constant, "
            "or all provide a file path."
        )
    else:
        da = imod.formats.array_io.reading._load(
            paths, use_cftime=False, _read=imod.idf._read, headers=headers
        )
    return da


def _open_package_idf(
    block_content: Dict[str, Any], variables: Sequence[str]
) -> List[xr.DataArray]:
    das = {}
    for variable in variables:
        variable_content = block_content[variable]
        paths = []
        headers = []
        values = []
        for entry in variable_content:
            path, header, value = _process_package_entry(entry)
            header["name"] = variable
            header["dims"] = ["layer"]
            header["layer"] = entry["layer"]
            paths.append(path)
            headers.append(header)
            values.append(value)

        das[variable] = _create_dataarray(paths, headers, values)

    return [das]


def _process_boundary_condition_entry(entry: Dict):
    """
    The iMOD project file supports constants in lieu of IDFs.
    """
    coords = {}
    time = entry["time"]
    if time == "steady-state":
        time = None
        dims = ()
    else:
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        coords["time"] = time
        dims = ("time",)

    # 0 signifies that the layer must be determined on the basis of
    # bottom elevation and stage.
    layer = entry["layer"]
    if layer == 0:
        layer is None
    else:
        coords["layer"] = layer
        dims = dims + ("layer",)

    if "path" not in entry:
        path = None
        header = {"coords": coords}
        value = entry["constant"]
    else:
        path = entry["path"]
        header = imod.idf.header(path, pattern="{name}")
        value = None

    header["dims"] = dims
    if layer is not None:
        header["layer"] = layer
    if time is not None:
        header["time"] = time

    return path, header, value


def _open_boundary_condition_idf(
    block_content, variables
) -> List[Dict[str, xr.DataArray]]:
    """
    Read the variables specified from block_content.
    """
    n_system = block_content["n_system"]
    n_time = len(block_content["time"])
    n_total = n_system * n_time

    das = [{}] * n_system
    for variable in variables:
        variable_content = block_content[variable]

        n = len(variable_content)
        if n != n_total:
            raise ValueError(
                f"Expected n_time * n_system = {n_time} * {n_system} = "
                f"{n_total} entries for variable {variable}. Received: {n}"
            )

        # Group the paths and headers by system.
        system_paths = defaultdict(list)
        system_headers = defaultdict(list)
        system_values = defaultdict(list)
        for i, entry in enumerate(variable_content):
            path, header, value = _process_boundary_condition_entry(entry)
            header["name"] = variable
            key = i % n_system
            system_paths[key].append(path)
            system_headers[key].append(header)
            system_values[key].append(value)

        # Concat one system at a time.
        for i, (paths, headers, values) in enumerate(
            zip(system_paths.values(), system_headers.values(), system_values.values())
        ):
            das[i][variable] = _create_dataarray(paths, headers, values)

    return das


def _read_package_gen(block_content: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for entry in block_content["gen"]:
        d = {
            "geodataframe": imod.gen.read(entry["path"]),
            "layer": entry["layer"],
        }
        out.append(d)
    return out


def _read_package_ipf(block_content: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for entry in block_content["ipf"]:
        d = {
            "dataframe": imod.ipf.read(entry["path"]),
            "layer": entry["layer"],
            "time": entry["time"],
        }
        out.append(d)
    return out


def read_projectfile(path: FilePath) -> Dict[str, Any]:
    """
    Read an iMOD project file into a collection of nested dictionaries.

    The top-level keys are the "topic" entries such "bnd" or "riv" in the
    project file. An example structure of the dictionaries is visualized below:

    ```
    content
    ├── bnd
    │   ├── active: bool
    │   └── ibound: list of dictionaries for each layer
    ├── riv
    │   ├── active: bool
    │   ├── conductance: list of dictionaries for each time and layer.
    │   ├── stage: idem.
    │   ├── bottom_elevation: idem.
    │   └── infiltration_factor: idem.
    etc.
    ```

    Time and layer are flattened into a single list and time is included in
    every dictionary:

    ```
    stage
    ├── 0  # First entry in list
    │   ├── active: bool
    │   ├── is_constant: bool
    │   ├── layer: int
    │   ├── factor: float
    │   ├── addition: float
    │   ├── constant: float
    │   ├── path: str
    │   └── time: str
    │
    ├── 1  # Second entry in list
    │   ├── etc.
    etc.
    ```

    Parameters
    ----------
    path: str or Path

    Returns
    -------
    content: Dict[str, Any]
    """
    with open(path) as f:
        lines = f.readlines()

    tokenized = []
    for i, line in enumerate(lines):
        try:
            tokenized.append(_tokenize(line))
        except Exception as e:
            raise type(e)(f"{e}\nError occurred in line {i}")

    lines = _LineIterator(tokenized)
    content = {}
    while not lines.finished:
        _parse_block(lines, content)

    return content


def open_projectfile_data(path: FilePath) -> Dict[str, Any]:
    """
    Read the contents of an iMOD project file and read/open the data present in
    it:

    * IDF data is lazily loaded into xarray.DataArrays.
    * GEN data is eagerly loaded into geopandas.GeoDataFrames
    * IPF data is eagerly loaded into pandas.DataFrames
    * Non-file based entries (such as the PCG settings) are kept as a dictionary.

    When multiple systems are present, they are numbered starting from one, e.g.:

    * drn-1
    * drn-2

    Parameters
    ----------
    path: pathlib.Path or str.

    Returns
    -------
    data: Dict[str, Any]
    Keys are the iMOD project file "topics", without parentheses.
    """
    content = read_projectfile(path)
    prj_data = {}
    for key, block_content in content.items():
        try:
            if key == "(hfb)":
                data = _read_package_gen(block_content)
            elif key == "(wel)":
                data = _read_package_ipf(block_content)
            elif key == "(cap)":
                variables = set(METASWAP_VARS).intersection(block_content.keys())
                data = _open_package_idf(block_content, variables)
            elif key in ("extra", "(pcg)"):
                data = [block_content]
            elif key in ("periods"):
                data = [
                    {
                        key: datetime.strptime(time, "%d-%m-%Y %H:%M:%S")
                        for key, time in block_content.items()
                    }
                ]
            elif key in KEYS:
                variables = KEYS[key]
                data = _open_package_idf(block_content, variables)
            elif key in DATE_KEYS:
                variables = DATE_KEYS[key]
                data = _open_boundary_condition_idf(block_content, variables)
            else:
                raise ValueError("Unsupported key")
        except Exception as e:
            raise type(e)(f"{e}. Errored while opening/reading data entries for: {key}")

        strippedkey = key.strip("(").strip(")")
        if len(data) > 1:
            for i, da in enumerate(data):
                prj_data[f"{strippedkey}-{i+1}"] = da
        else:
            prj_data[strippedkey] = data[0]

    return prj_data
