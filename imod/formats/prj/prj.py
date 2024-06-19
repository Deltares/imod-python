"""
Utilities for parsing a project file.
"""

import shlex
from collections import defaultdict
from datetime import datetime
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

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
    "(evt)": ("rate", "surface", "depth"),
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
    values = [v.strip().replace("\\", "/") for v in line.split(",")]
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
        if len(line) > 1:
            time = line[1]
            return f"{date} {time}"
        else:
            return date
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
            content["path"] = Path(line[6]).resolve()
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


def _parse_speciesblock(lines: _LineIterator):
    try:
        species = {}
        while not lines.finished:
            line = next(lines)
            # Stop if we encounter an empty line.
            if len(line) == 0:
                break
            name, nr = line
            nr = int(nr)
            species[nr] = name
        return species
    except Exception as e:
        _wrap_error_message(e, "species entry", lines)


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
        elif key == "species":
            blockcontent = _parse_speciesblock(lines)
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

def _try_read_with_func(func, path, *args, **kwargs):
    try:
        return func(path, *args, **kwargs)
    except Exception as e:
        raise type(e)(f"{e}. Error thrown while opening file: {path}")

def _create_datarray_from_paths(paths: List[str], headers: List[Dict[str, Any]]):
    da = _try_read_with_func(
        imod.formats.array_io.reading._load, paths, use_cftime=False, _read=imod.idf._read, headers=headers
    )
    return da


def _create_dataarray_from_values(values: List[float], headers: List[Dict[str, Any]]):
    coords = _merge_coords(headers)
    firstdims = headers[0]["dims"]
    shape = [len(coord) for coord in coords.values()]
    da = xr.DataArray(np.reshape(values, shape), dims=firstdims, coords=coords)
    return da


def _create_dataarray(
    paths: List[str], headers: List[Dict[str, Any]], values: List[float]
) -> xr.DataArray:
    """
    Create a DataArray from a list of IDF paths, or from constant values.
    """
    values_valid = []
    paths_valid = []
    headers_paths = []
    headers_values = []
    for path, header, value in zip(paths, headers, values):
        if path is None:
            headers_values.append(header)
            values_valid.append(value)
        else:
            headers_paths.append(header)
            paths_valid.append(path)

    if paths_valid and values_valid:
        dap = _create_datarray_from_paths(paths_valid, headers_paths)
        dav = _create_dataarray_from_values(values_valid, headers_values)
        dap.name = "tmp"
        dav.name = "tmp"
        da = xr.merge((dap, dav), join="outer")["tmp"]
    elif paths_valid:
        da = _create_datarray_from_paths(paths_valid, headers_paths)
    elif values_valid:
        da = _create_dataarray_from_values(values_valid, headers_values)

    da = apply_factor_and_addition(headers, da)
    return da


def apply_factor_and_addition(headers, da):
    if not ("layer" in da.coords or "time" in da.dims):
        factor = headers[0]["factor"]
        addition = headers[0]["addition"]
        da = da * factor + addition
    elif "layer" in da.coords and "time" not in da.dims:
        da = apply_factor_and_addition_per_layer(headers, da)
    else:
        header_per_time = defaultdict(list)
        for time in da.coords["time"].values:
            for header in headers:
                if np.datetime64(header["time"]) == time:
                    header_per_time[time].append(header)

        for time in da.coords["time"]:
            da.loc[{"time": time}] = apply_factor_and_addition(
                header_per_time[np.datetime64(time.values)],
                da.sel(time=time, drop=True),
            )
    return da


def apply_factor_and_addition_per_layer(headers, da):
    layer = da.coords["layer"].values
    header_per_layer = {}
    for header in headers:
        if header["layer"] in header_per_layer.keys():
            raise ValueError("error in project file: layer repetition")
        header_per_layer[header["layer"]] = header
    addition_values = [header_per_layer[lay]["addition"] for lay in layer]
    factor_values = [header_per_layer[lay]["factor"] for lay in layer]
    addition = xr.DataArray(addition_values, coords={"layer": layer}, dims=("layer"))
    factor = xr.DataArray(factor_values, coords={"layer": layer}, dims=("layer",))
    da = da * factor + addition
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
            header["addition"] = entry["addition"]
            header["factor"] = entry["factor"]
            paths.append(path)
            headers.append(header)
            values.append(value)

        das[variable] = _create_dataarray(paths, headers, values)

    return [das]


def _process_time(time: str, yearfirst: bool = True):
    if time == "steady-state":
        time = None
    else:
        if yearfirst:
            if len(time) == 19:
                time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
            elif len(time) == 10:
                time = datetime.strptime(time, "%Y-%m-%d")
            else:
                raise ValueError(
                    f"time data {time} does not match format "
                    '"%Y-%m-%d %H:%M:%S" or "%Y-%m-%d"'
                )
        else:
            if len(time) == 19:
                time = datetime.strptime(time, "%d-%m-%Y %H:%M:%S")
            elif len(time) == 10:
                time = datetime.strptime(time, "%d-%m-%Y")
            else:
                raise ValueError(
                    f"time data {time} does not match format "
                    '"%d-%m-%Y %H:%M:%S" or "%d-%m-%Y"'
                )
    return time


def _process_boundary_condition_entry(entry: Dict, periods: Dict[str, datetime]):
    """
    The iMOD project file supports constants in lieu of IDFs.

    Also process repeated stress periods (on a yearly basis): substitute the
    original date here.
    """
    coords = {}
    timestring = entry["time"]

    # Resolve repeating periods first:
    time = periods.get(timestring)
    if time is not None:
        repeat = time
    else:
        # this resolves e.g. "steady-state"
        time = _process_time(timestring)
        repeat = None

    if time is None:
        dims = ()
    else:
        dims = ("time",)
        coords["time"] = time

    # 0 signifies that the layer must be determined on the basis of
    # bottom elevation and stage.
    layer = entry["layer"]
    if layer <= 0:
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
    header["addition"] = entry["addition"]
    header["factor"] = entry["factor"]
    header["dims"] = dims
    if layer is not None:
        header["layer"] = layer
    if time is not None:
        header["time"] = time

    return path, header, value, repeat


def _open_boundary_condition_idf(
    block_content, variables, periods: Dict[str, datetime]
) -> Tuple[List[Dict[str, xr.DataArray]], List[datetime]]:
    """
    Read the variables specified from block_content.
    """
    n_system = block_content["n_system"]
    n_time = len(block_content["time"])
    n_total = n_system * n_time

    das = [{} for _ in range(n_system)]
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
        all_repeats = set()
        for i, entry in enumerate(variable_content):
            path, header, value, repeat = _process_boundary_condition_entry(
                entry, periods
            )
            header["name"] = variable
            key = i % n_system
            system_paths[key].append(path)
            system_headers[key].append(header)
            system_values[key].append(value)
            if repeat:
                all_repeats.add(repeat)

        # Concat one system at a time.
        for i, (paths, headers, values) in enumerate(
            zip(system_paths.values(), system_headers.values(), system_values.values())
        ):
            das[i][variable] = _create_dataarray(paths, headers, values)

    repeats = sorted(all_repeats)
    return das, repeats


def _read_package_gen(
    block_content: Dict[str, Any], has_topbot: bool
) -> List[Dict[str, Any]]:
    out = []
    for entry in block_content["gen"]:
        gdf = imod.gen.read(entry["path"])
        if has_topbot:
            gdf["resistance"] = entry["factor"] * entry["addition"]
        else:
            gdf["multiplier"] = entry["factor"] * entry["addition"]
        d = {
            "geodataframe": gdf,
            "layer": entry["layer"],
        }
        out.append(d)
    return out


def _read_package_ipf(
    block_content: Dict[str, Any], periods: Dict[str, datetime]
) -> Tuple[List[Dict[str, Any]], List[datetime]]:
    out = []
    repeats = []
    for entry in block_content["ipf"]:
        timestring = entry["time"]
        layer = entry["layer"]
        time = periods.get(timestring)
        factor = entry["factor"]
        addition = entry["addition"]
        if time is None:
            time = _process_time(timestring)
        else:
            repeats.append(time)

        # Ensure the columns are identifiable.
        path = Path(entry["path"])
        ipf_df, indexcol, ext = _try_read_with_func(imod.ipf._read_ipf, path)
        if indexcol == 0:
            # No associated files
            columns = ("x", "y", "rate")
            if layer <= 0:
                df = ipf_df.iloc[:, :5]
                columns = columns + ("top", "bottom")
            else:
                df = ipf_df.iloc[:, :3]
            df.columns = columns
        else:
            dfs = []
            for row in ipf_df.itertuples():
                filename = row[indexcol]
                path_assoc = path.parent.joinpath(f"{filename}.{ext}")
                df_assoc = _try_read_with_func(imod.ipf.read_associated, path_assoc).iloc[:, :2]
                df_assoc.columns = ["time", "rate"]
                df_assoc["x"] = row[1]
                df_assoc["y"] = row[2]
                df_assoc["id"] = path_assoc.stem
                if layer <= 0:
                    df_assoc["top"] = row[4]
                    df_assoc["bottom"] = row[5]
                dfs.append(df_assoc)
            df = pd.concat(dfs, ignore_index=True, sort=False)
        df["rate"] = df["rate"] * factor + addition

        d = {
            "dataframe": df,
            "layer": layer,
            "time": time,
        }
        out.append(d)
    repeats = sorted(repeats)
    return out, repeats


def read_projectfile(path: FilePath) -> Dict[str, Any]:
    """
    Read an iMOD project file into a collection of nested dictionaries.

    The top-level keys are the "topic" entries such "bnd" or "riv" in the
    project file. An example structure of the dictionaries is visualized below:

    .. code-block::

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

    Time and layer are flattened into a single list and time is included in
    every dictionary:

    .. code-block::

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


    Parameters
    ----------
    path: str or Path

    Returns
    -------
    content: Dict[str, Any]
    """
    # Force to Path
    path = Path(path)

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
    wdir = path.parent
    # Change dir temporarely to projectfile dir to resolve relative paths
    with imod.util.cd(wdir):
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

    Xarray requires valid dates for the time coordinate. Aliases such as
    "summer" and "winter" that are associated with dates in the project file
    Periods block cannot be used in the time coordinate. Hence, this function
    will instead insert the dates associated with the aliases, with the year
    replaced by 1899; as the iMOD calendar starts at 1900, this ensures that
    the repeats are always first and that no date collisions will occur.

    Parameters
    ----------
    path: pathlib.Path or str.

    Returns
    -------
    data: Dict[str, Any]
    Keys are the iMOD project file "topics", without parentheses.
    """
    content = read_projectfile(path)
    periods_block = content.pop("periods", None)
    if periods_block is None:
        periods = {}
    else:
        # Set the year of a repeat date to 1899: this ensures it falls outside
        # of the iMOD calendar. Collisions are then always avoided.
        periods = {
            key: _process_time(time, yearfirst=False).replace(year=1899)
            for key, time in periods_block.items()
        }

    # Pop species block, at the moment we do not do much with,
    # since most regional models are without solute transport
    content.pop("species", None)

    has_topbot = "(top)" in content and "(bot)" in content
    prj_data = {}
    repeat_stress = {}
    for key, block_content in content.items():
        repeats = None
        try:
            if key == "(hfb)":
                data = _read_package_gen(block_content, has_topbot)
            elif key == "(wel)":
                data, repeats = _read_package_ipf(block_content, periods)
            elif key == "(cap)":
                variables = set(METASWAP_VARS).intersection(block_content.keys())
                data = _open_package_idf(block_content, variables)
            elif key in ("extra", "(pcg)"):
                data = [block_content]
            elif key in KEYS:
                variables = KEYS[key]
                data = _open_package_idf(block_content, variables)
            elif key in DATE_KEYS:
                variables = DATE_KEYS[key]
                data, repeats = _open_boundary_condition_idf(
                    block_content, variables, periods
                )
            else:
                raise KeyError(f"Unsupported key: '{key}'")
        except Exception as e:
            raise type(e)(f"{e}. Errored while opening/reading data entries for: {key}")

        strippedkey = key.strip("(").strip(")")
        if len(data) > 1:
            for i, da in enumerate(data):
                numbered_key = f"{strippedkey}-{i + 1}"
                prj_data[numbered_key] = da
                repeat_stress[numbered_key] = repeats
        else:
            prj_data[strippedkey] = data[0]
            repeat_stress[strippedkey] = repeats

    repeat_stress = {k: v for k, v in repeat_stress.items() if v}
    return prj_data, repeat_stress


def read_timfile(path: FilePath) -> List[Dict]:
    def parsetime(time: str) -> datetime:
        # Check for steady-state:
        if time == "00000000000000":
            return None
        return datetime.strptime(time, "%Y%m%d%H%M%S")

    with open(path, "r") as f:
        lines = f.readlines()

    # A line contains 2, 3, or 4 values:
    # time, isave, nstp, tmult
    casters = {
        "time": parsetime,
        "save": lambda x: bool(int(x)),
        "n_timestep": int,
        "timestep_multiplier": float,
    }
    content = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue
        parts = stripped.split(",")
        entry = {k: cast(s.strip()) for s, (k, cast) in zip(parts, casters.items())}
        content.append(entry)

    return content
