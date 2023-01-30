"""
Utilities for parsing a project file.
"""
import shlex
from collections import defaultdict
from itertools import chain
from os import PathLike
from typing import Any, Dict, List, Tuple, Union

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
    "(cap)": (None),
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


class LineIterator:
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


def tokenize(line: str) -> List[str]:
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

    >> tokenize("That's life")

    >> tokenize("That 's life'")
    >> ["That", "s life"]

    >> tokenize("That,'s life'")
    >> ["That", "s life"]
    """
    values = [v.strip() for v in line.split(",")]
    tokens = list(chain.from_iterable(shlex.split(v) for v in values))
    return tokens


def wrap_error_message(exception, description, lines):
    lines.back()
    content = next(lines)
    number = lines.count + 1
    raise type(exception)(
        f"{exception}\n"
        f"Failed to parse {description} for line {number} with content:\n{content}"
    )


def parse_blockheader(lines: List[str]) -> Tuple[int, str, bool]:
    try:
        no_result = None, None, None
        line = next(lines)

        # Skip if it's an empty line.
        if len(line) == 0:
            return no_result

        first = line[0].lower()
        if first in ("periods", "species"):
            return 1, first, True
        # The line must contain atleast nper, key, active.
        elif len(line) >= 3:
            n = int(first)
            key = line[1].lower()
            active = bool(int(line[2]))
            return n, key, active
        # It's a comment or something.
        else:
            return no_result
    except Exception as e:
        wrap_error_message(e, "block header", lines)


def parse_time(lines: List[str]) -> str:
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
        wrap_error_message(e, "date time", lines)


def parse_blockline(lines, time=None):
    try:
        line = next(lines)
        content = {
            "active": bool(line[0]),
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
        wrap_error_message(e, "entries", lines)


def read_nsub_nsystem(lines):
    try:
        line = next(lines)
        n_entry = int(line[0])
        n_system = int(line[1])
        return n_entry, n_system
    except Exception as e:
        wrap_error_message(e, "number of sub-entries and number of systems", lines)


def parse_notimeblock(
    lines: List[str],
    fields: List[str],
) -> Dict[str, Dict[str, List]]:
    n_entry, n_system = read_nsub_nsystem(lines)

    if len(fields) != n_entry:
        raise ValueError(
            f"Expected NSYSTEM entry of {len(fields)} for {fields}, read: {n_entry}"
        )
    content = {
        field: [parse_blockline(lines) for _ in range(n_system)] for field in fields
    }
    content["n_system"] = n_system
    return content


def parse_timeblock(
    lines: List[str],
    fields: List[str],
    n: int,
) -> Dict[str, List[Dict]]:
    n_fields = len(fields)
    content = defaultdict(list)
    for _ in range(n):
        time = parse_time(lines)
        content["time"].append(time)
        n_entry, n_system = read_nsub_nsystem(lines)

        if n_fields != n_entry:
            raise ValueError(
                f"Expected NSYSTEM entry of {n_fields} for {fields}, read: {n_entry}"
            )

        for field in fields:
            content[field].extend(
                [parse_blockline(lines, time) for _ in range(n_system)]
            )

    return content


def parse_pcgblock(lines):
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

        if len(line) == 2:
            # TODO: not clear what's allowed:
            # MXITER= 250  # This one's confirmed okay.
            # MXITER = 250  # This generates three entries.
            # MXITER =250  # This generates two entries, but other grouping.
            pcglines = line + [next(line) for _ in range(11)]

            content = {}
            for line in pcglines:
                key = line[0].strip("=").lower()
                value = types[key](line[1])
                content[key] = value

        elif len(line) == 12:
            line_iterator = iter(line)
            content = {
                k: valuetype(next(line_iterator)) for k, valuetype in types.items()
            }

        else:
            raise ValueError(
                f"Expected 12 KEY= VALUE pairs, or 12 values. Found {len(line)}"
            )

        return content
    except Exception as e:
        wrap_error_message(e, "PCG entry", lines)


def parse_periodsblock(lines):
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
            start = parse_time(lines)
            periods[alias] = start
        return periods
    except Exception as e:
        wrap_error_message(e, "periods data block", lines)


def parse_block(lines: LineIterator, content: Dict[str, Any]) -> None:
    """
    Mutates content dict.
    """
    n = key = active = None

    # A project file may contain any number of lines outside of a "topic"
    # block. parse_blockheader will return triple None in that case.
    while key is None and not lines.finished:
        n, key, active = parse_blockheader(lines)

    try:
        if key in KEYS:
            if n != 1:
                raise ValueError(f"Expected N=1 for {key}, read: {n}")
            fields = KEYS[key]
            blockcontent = parse_notimeblock(lines, fields)
        elif key in DATE_KEYS:
            fields = DATE_KEYS[key]
            blockcontent = parse_timeblock(lines, fields, n)
        elif key == "(pcg)":
            blockcontent = parse_pcgblock(lines)
        elif key == "periods":
            blockcontent = parse_periodsblock(lines)
        else:
            other = ("(pcg)", "(gcg)", "(vdf)")
            options = tuple(KEYS.keys()) + tuple(DATE_KEYS.keys()) + other
            lines.back()
            line = next(lines)
            raise ValueError(
                f"Failed to recognize header keyword: {key}. Expected one of keywords {options}"
                f"\nErrored in line with entries:\n{line}"
            )

    except Exception as e:
        raise type(e)(f"{e}\nError occured for keyword: {key}")

    if blockcontent is not None:
        blockcontent["active"] = active

    content[key] = blockcontent
    return


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

    lines = LineIterator([tokenize(line) for line in lines])
    content = {}
    while not lines.finished:
        parse_block(lines, content)

    return content


def open_package_idf(block_content, variables):
    import imod

    das = {}
    for variable in variables:
        variable_content = block_content[variable]
        paths = []
        headers = []
        for entry in variable_content:
            path = entry["path"]
            header = imod.idf.header(path, pattern="{name}")
            header["name"] = variable
            header["dims"] = ["layer"]
            header["layer"] = entry["layer"]
            paths.append(path)
            headers.append(header)

        das[variable] = imod.array_io.reading._load(
            paths, use_cftime=False, _read=imod.idf._read, headers=headers
        )

    return das


def open_boundary_condition_idf(block_content, variables):
    import imod

    # TODO
    # Groupby time, but not by layer.
    # Every layer is a separate system!
    # Use n_system
    das = {}
    for variable in variables:
        variable_content = block_content[variable]
        paths = []
        headers = []
        for entry in variable_content:
            path = entry["path"]
            header = imod.idf.header(path, pattern="{name}")
            header["name"] = variable
            header["layer"] = entry["layer"]

            datetime = entry["time"]
            if datetime == "steady-state":
                header["dims"] = ["layer"]
            else:
                header["dims"] = ["time", "layer"]
                header["time"] = datetime.strptime("%Y-%m-%d %H:%M:%S")

            paths.append(path)
            headers.append(header)

        das[variable] = imod.array_io.reading._load(
            paths, use_cftime=False, _read=imod.idf._read, headers=headers
        )

    return das


def open_projectfile(path):
    content = read_projectfile(path)
    data = {}
    for key, block_content in content.items():
        if key in ("(hfb)", "(wel)"):
            continue
        if key in KEYS:
            variables = KEYS[key]
            das = open_package_idf(block_content, variables)
        elif key in DATE_KEYS:
            variables = DATE_KEYS[key]
            das = open_boundary_condition_idf(block_content, variables)
        strippedkey = key[1:-1]
        data[strippedkey] = das
    return data
