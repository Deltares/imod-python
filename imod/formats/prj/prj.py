"""
Utilities for parsing a project file.
"""

import shlex
import textwrap
from collections import defaultdict
from dataclasses import asdict, dataclass
from dataclasses import field as data_field
from datetime import datetime
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

import imod
import imod.logging
from imod.logging.loglevel import LogLevel

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
    "artificial_recharge_layer",
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


def _get_array_transformation_parameters(
    headers: List[Dict[str, Any]], key: str, dim: str
) -> Union[xr.DataArray | float]:
    """
    In imod5 prj files one can add linear transformation parameters to transform
    the data read from an idf file: we can specify a multiplication factor and a
    constant that will be added to the values. The factor and addition
    parameters can be can be scalar (if applied to 1 idf), or they can be
    xr.DataArrays if the factor and addition are for example layer- or
    time-dependent  (if both we apply the transformations one at a time)

    Parameters
    ----------
    headers: List[Dict[str, Any]]
        prj-file lines which we want to import, serialized as a dictionary.
    key: str
        specifies the name of the transformation parameter in the idf file.
        Usually "factor" or "addition"
    dim: str
        the name of the dimension over which transformation parameters are
        expected to differ for the current import. Usually "time"or "layer"
    """
    if dim in headers[0].keys():
        return xr.DataArray(
            data=[header[key] for header in headers],
            dims=(dim,),
            coords={dim: [header[dim] for header in headers]},
        )
    else:
        return headers[0][key]


def _create_dataarray_from_paths(
    paths: List[str], headers: List[Dict[str, Any]], dim: str
) -> xr.DataArray:
    factor = _get_array_transformation_parameters(headers, "factor", dim)
    addition = _get_array_transformation_parameters(headers, "addition", dim)
    da = _try_read_with_func(
        imod.formats.array_io.reading._load,
        paths,
        use_cftime=False,
        _read=imod.idf._read,
        headers=headers,
    )

    # Ensure factor and addition do not have more dimensions than da
    if isinstance(factor, xr.DataArray):
        missing_dims = set(factor.dims) - set(da.dims)
        if missing_dims:
            factor = factor.isel({d: 0 for d in missing_dims}, drop=True)
            addition = addition.isel({d: 0 for d in missing_dims}, drop=True)

    return da * factor + addition


def _create_dataarray_from_values(
    values: List[float], headers: List[Dict[str, Any]], dim: str
):
    factor = _get_array_transformation_parameters(headers, "factor", dim)
    addition = _get_array_transformation_parameters(headers, "addition", dim)
    coords = _merge_coords(headers)
    firstdims = headers[0]["dims"]
    shape = [len(coord) for coord in coords.values()]
    da = xr.DataArray(np.reshape(values, shape), dims=firstdims, coords=coords)
    return da * factor + addition


def _create_dataarray(
    paths: List[str], headers: List[Dict[str, Any]], values: List[float], dim: str
) -> xr.DataArray:
    """
    Create a DataArray from a list of IDF paths, or from constant values.

    There are mixed cases possible, where some of the layers or stress periods
    contain only a single constant value, and the others are specified as IDFs.
    In that case, we cannot do a straightforward concatenation.
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
        # Both lists contain entries: mixed case.
        dap = _create_dataarray_from_paths(paths_valid, headers_paths, dim=dim)
        dav = _create_dataarray_from_values(values_valid, headers_values, dim=dim)
        dap.name = "tmp"
        dav.name = "tmp"
        da = xr.merge((dap, dav), join="outer")["tmp"]
    elif paths_valid:
        # Only paths provided
        da = _create_dataarray_from_paths(paths_valid, headers_paths, dim=dim)
    elif values_valid:
        # Only scalar values provided
        da = _create_dataarray_from_values(values_valid, headers_values, dim=dim)

    return da


def _open_package_idf(
    block_content: Dict[str, Any], variables: Sequence[str]
) -> list[dict[str, xr.DataArray]]:
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

        das[variable] = _create_dataarray(paths, headers, values, dim="layer")

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
    # -1 signifies that the layer must be determined on the basis of
    # top active cells.
    layer = entry["layer"]

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
            das[i][variable] = _create_dataarray(paths, headers, values, dim="time")

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


IPF_LOG_MESSAGE_TEMPLATE = """\
    A well with the same x, y, id, filter_top and
    filter_bot was already imported. This happened at x =
    {row[1]}, y = {row[2]}, id = {path_assoc.stem}. Now the
    ID for this new well was appended with the suffix _{suffix})
    """


@dataclass
class IpfResult:
    has_associated: bool = data_field(default_factory=bool)
    dataframe: list[pd.DataFrame] = data_field(default_factory=list)
    layer: list[int] = data_field(default_factory=list)
    time: list[str] = data_field(default_factory=list)
    factor: list[float] = data_field(default_factory=list)
    addition: list[float] = data_field(default_factory=list)

    def append(
        self,
        dataframe: pd.DataFrame,
        layer: int,
        time: str,
        factor: float,
        addition: float,
    ):
        self.dataframe.append(dataframe)
        self.layer.append(layer)
        self.time.append(time)
        self.factor.append(factor)
        self.addition.append(addition)


def _process_ipf_time(
    entry: Dict[str, Any], periods: Dict[str, datetime], repeats: list[str]
) -> tuple[Optional[str], list[str]]:
    timestring = entry["time"]
    time = periods.get(timestring)
    if time is None:
        time = _process_time(timestring)
    else:
        repeats.append(time)
    return time, repeats


def _prepare_df_unassociated(ipf_df: pd.DataFrame, layer: int) -> pd.DataFrame:
    columns = ("x", "y", "rate")
    if layer <= 0:
        df = ipf_df.iloc[:, :5]
        columns = columns + ("filt_top", "filt_bot")
    else:
        df = ipf_df.iloc[:, :3]
    df.columns = columns
    return df


def _prepare_df_associated(
    ipf_df: pd.DataFrame,
    imported_wells: dict[tuple, int],
    indexcol: int,
    path: Path,
    ext: str,
) -> pd.DataFrame:
    dfs = []
    for row in ipf_df.itertuples():
        filename = row[indexcol]
        path_assoc = path.parent.joinpath(f"{filename}.{ext}")
        well_characteristics_dict = {
            "x": row[1],
            "y": row[2],
            "id": path_assoc.stem,
            "filt_top": row[4],
            "filt_bot": row[5],
        }
        df_assoc = _try_read_with_func(imod.ipf.read_associated, path_assoc).iloc[:, :2]
        df_assoc.columns = ["time", "rate"]
        df_assoc = df_assoc.assign(**well_characteristics_dict)
        well_characteristics = tuple(well_characteristics_dict.values())
        if well_characteristics not in imported_wells.keys():
            imported_wells[well_characteristics] = 0
        else:
            suffix = imported_wells[well_characteristics] + 1
            imported_wells[well_characteristics] = suffix
            df_assoc["id"] = df_assoc["id"] + f"_{suffix}"

            log_message = textwrap.dedent(
                IPF_LOG_MESSAGE_TEMPLATE.format(
                    row=row, path_assoc=path_assoc, suffix=suffix
                )
            )
            imod.logging.logger.log(
                loglevel=LogLevel.WARNING,
                message=log_message,
                additional_depth=2,
            )

        dfs.append(df_assoc)
    return pd.concat(dfs, ignore_index=True, sort=False)


def _read_package_ipf(
    block_content: Dict[str, Any], periods: Dict[str, datetime]
) -> Tuple[Dict[str, Dict], List[datetime]]:
    out = defaultdict(IpfResult)
    repeats = []

    # we will store in this set the tuples of (x, y, id, well_top, well_bot)
    # which should be unique for each well
    imported_wells = {}

    for entry in block_content["ipf"]:
        time, repeats = _process_ipf_time(entry, periods, repeats)
        layer = entry["layer"]
        factor = entry["factor"]
        addition = entry["addition"]

        # Ensure the columns are identifiable.
        path = Path(entry["path"])
        ipf_df, indexcol, ext = _try_read_with_func(imod.ipf._read_ipf, path)
        if indexcol == 0:
            # No associated files
            has_associated = False
            df = _prepare_df_unassociated(ipf_df, layer)
        else:
            has_associated = True
            df = _prepare_df_associated(ipf_df, imported_wells, indexcol, path, ext)
        df["rate"] = df["rate"] * factor + addition

        out[path.stem].has_associated = has_associated
        out[path.stem].append(df, layer, time, factor, addition)

    out_dict_ls: dict[str, dict] = {key: asdict(o) for key, o in out.items()}
    repeats = sorted(repeats)
    return out_dict_ls, repeats


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


def _is_var_ipf_and_path(block_content: dict[str, Any], var: str):
    block_item = block_content[var]
    path = block_item[0]["path"]
    is_ipf = path.suffix.lower() == ".ipf"
    return is_ipf, path


def open_projectfile_data(path: FilePath) -> tuple[dict[str, Any], dict[str, Any]]:
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
                maybe_ipf_var = "artificial_recharge_layer"
                is_ipf, maybe_ipf_path = _is_var_ipf_and_path(
                    block_content, maybe_ipf_var
                )
                # If its an ipf drop it and manually add it by calling the
                # ipf.read function. If its not an ipf then its an idf, which
                # doesnt need any special treatment.
                if is_ipf:
                    block_content.pop(maybe_ipf_var)
                variables = set(METASWAP_VARS).intersection(block_content.keys())
                data = _open_package_idf(block_content, variables)
                # Read and reattach ipf data to package data.
                if is_ipf:
                    data[0][maybe_ipf_var] = imod.ipf.read(maybe_ipf_path)
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
        if strippedkey == "wel":
            for key, d in data.items():
                named_key = f"{strippedkey}-{key}"
                prj_data[named_key] = d
                repeat_stress[named_key] = repeats
        elif len(data) > 1:
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
