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


def find_entry(line: str, entry: str, type: type) -> str:
    if entry not in line:
        return None
    else:
        _, after = line.split(entry)
        return type(after.split()[0])


def read_internal(f: IO[str], max_rows: int, dtype: type) -> np.ndarray:
    return np.loadtxt(
        fname=f,
        dtype=dtype,
        max_rows=max_rows,
    )


def read_external_binaryfile(path: Path, dtype: type) -> np.ndarray:
    return np.fromfile(
        file=path,
        dtype=dtype,
        sep="",
    )


def read_external_textfile(path: Path, dtype: type) -> np.ndarray:
    return np.loadtxt(
        fname=path,
        dtype=dtype,
    )


def advance_to_header(f: IO[str], section) -> None:
    line = None
    start = f"begin {section}"
    # Empty line is end-of-file
    while not end_of_file(line):
        line = f.readline()
        stripped = line.strip().lower()
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


def parse_option(stripped: str, fname: str) -> Tuple[str, str]:
    separated = stripped.split()
    nwords = len(separated)
    if nwords == 1:
        key = separated[0]
        value = True
    elif nwords == 2:
        key, value = separated
    else:
        raise ValueError(
            "More than two words found in block:"
            f"{','.join(separated)} in file {fname}"
        )
    return key, value


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
