"""
Commonly shared utilities for reading MODFLOW6 input files.
"""
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Tuple

import numpy as np


def end_of_file(line: str) -> bool:
    return line == ""


def strip_line(line: str) -> str:
    # remove possible comments
    before, _, _ = line.partition("#")
    return before.strip().lower()


def flatten(iterable: Any) -> List[Any]:
    out = []
    for part in iterable:
        out.extend(part)
    return out


def split_line(line: str) -> List[str]:
    # Maybe check: https://stackoverflow.com/questions/36165050/python-equivalent-of-fortran-list-directed-input
    # Split on comma and whitespace, like a FORTRAN read would do.
    flat = flatten([part.split(",") for part in line.split()])
    return [part for part in flat if part != ""]


def to_float(string: str) -> float:
    # Fortran float may contain d (e.g. 1.0d0), Python only accepts e.
    string = string.replace("d", "e")
    # Fortran may specify exponents without a letter, e.g. 1.0+1 for 1.0e+1
    if "e" not in string:
        string = string.replace("+", "e+").replace("-", "e-")
    return float(string)


def find_entry(line: str, entry: str, cast: Callable) -> str:
    if entry not in line:
        return None
    else:
        _, after = line.split(entry)
        return cast(after.split()[0])


def read_internal(f: IO[str], dtype: type, max_rows: int) -> np.ndarray:
    return np.loadtxt(
        fname=f,
        dtype=dtype,
        max_rows=max_rows,
    )


def read_external_binaryfile(path: Path, dtype: type, max_rows: int) -> np.ndarray:
    return np.fromfile(
        file=path,
        dtype=dtype,
        count=max_rows,
        sep="",
    )


def read_fortran_deflated_text_array(
    path: Path, dtype: type, max_rows: int
) -> np.ndarray:
    """
    The Fortran read intrinsic is capable of parsing arrays in many forms.
    One of those is:

        1.0
        2*2.0
        3.0

    Which should be interpreted as: [1.0, 2.0, 2.0, 3.0]

    This function attempts this.
    """
    out = np.empty(max_rows, dtype)
    with open(path, "r") as f:
        lines = [line.strip() for line in f]

    iterable_lines = iter(lines)
    start = 0
    while start < max_rows:

        line = next(iterable_lines)
        if "*" in line:
            n, value = line.split("*")
            n = int(n)
            value = dtype(value)
        else:
            n = 1
            value = dtype(line)

        end = start + n
        out[start:end] = value
        start = end

    return out


def read_external_textfile(path: Path, dtype: type, max_rows: int) -> np.ndarray:
    try:
        return np.loadtxt(
            fname=path,
            dtype=dtype,
            max_rows=max_rows,
        )
    except ValueError as e:
        if str(e).startswith("could not convert string to float"):
            return read_fortran_deflated_text_array(path, dtype, max_rows)
        else:
            raise e


def advance_to_header(f: IO[str], section) -> None:
    line = None
    start = f"begin {section}"
    # Empty line is end-of-file
    while not end_of_file(line):
        line = f.readline()
        stripped = strip_line(line)
        # Return if start has been found
        if stripped == start:
            return
        # Continue if line is empty
        elif stripped == "":
            continue
        # Raise if the line has (unexpected) content
        else:
            break
    # Also raise if no further content is found
    raise ValueError(f'"{start}" is not present in file {f.name}')


def parse_option(stripped: str, fname: str) -> Tuple[str, Any]:
    separated = stripped.split()
    nwords = len(separated)
    if nwords == 0:
        raise ValueError(f"Cannot parse option in {fname}")
    elif nwords == 1:
        key = separated[0]
        value = True
    elif nwords == 2:
        key, value = separated
    else:
        key = separated[0]
        value = separated[1:]
    return key, value


def read_key_value_block(f: IO[str], parse: Callable) -> Dict[str, str]:
    fname = f.name
    content = {}
    line = None
    while not end_of_file(line):
        line = f.readline()
        stripped = strip_line(line)
        # Return if end has been found
        if stripped[:3] == "end":
            return content
        # Continue in case of an empty line
        elif stripped == "":
            continue
        # Valid entry
        else:
            key, value = parse(stripped, fname)
            content[key] = value

    # Also raise if no further content is found
    raise ValueError(f'"end" of block is not present in file {fname}')


def read_iterable_block(f: IO[str], parse: Callable) -> List[Any]:
    fname = f.name
    content = []
    line = None
    while not end_of_file(line):
        line = f.readline()
        stripped = strip_line(line)
        # Return if end has been found
        if stripped[:3] == "end":
            return content
        # Continue in case of an empty line
        elif stripped == "":
            continue
        # Valid entry
        else:
            content.append(parse(stripped, fname))

    # Also raise if no further content is found
    raise ValueError(f'"end" of block is not present in file {fname}')


def parse_dimension(stripped: str, fname: str) -> Tuple[str, int]:
    key, value = parse_option(stripped, fname)
    return key, int(value)


def advance_to_period(f: IO[str]) -> int:
    line = None
    start = "begin period"
    # Empty line is end-of-file
    while not end_of_file(line):
        line = f.readline()
        stripped = strip_line(line)
        # Return if start has been found
        if stripped[:12] == start:
            return int(stripped.split()[2])
        # Continue if line is empty
        elif stripped == "":
            continue
        # Raise if the line has (unexpected) content
        else:
            break
    else:
        return None
