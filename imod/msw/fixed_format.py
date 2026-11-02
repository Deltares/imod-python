from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VariableMetaData:
    column_width: int
    min_value: int | float | None
    max_value: int | float | None
    dtype: type
    optional: bool = False


def format_fixed_width(value: Any, metadata: VariableMetaData) -> str:
    if metadata.dtype == str:
        format_string = "{:" + f"{metadata.column_width}" + "}"
    elif metadata.dtype == int:
        format_string = "{:" + f"{metadata.column_width}d" + "}"
    elif metadata.dtype == float:
        whole_number_digits = len(str(int(abs(value))))
        decimal_number_width = max(0, metadata.column_width - whole_number_digits - 2)
        format_string = "{:" + f"{metadata.column_width}.{decimal_number_width}f" + "}"
    else:
        raise TypeError(f"dtype {metadata.dtype} is not supported")

    converted_value = metadata.dtype(value)
    return format_string.format(converted_value)


def fixed_format_parser(
    file: str | Path, metadata_dict: dict[str, VariableMetaData]
) -> dict[str, list[Any]]:
    """
    Read fixed format file, using a metadata_dict from a MetaSWAP package.

    Parameters
    ----------
    file: str or Path
        Fixed format file to read, likely a MetaSWAP input file
    metadata_dict: dict
        Dictionary with the VariableMetaData. Access this dictionary in a
        package by calling <pkg>._metadata_dict
    """
    results: dict[str, list[Any]] = defaultdict(list)

    with open(file) as f:
        lines = f.readlines()
        troublesome = set()
        for line in lines:
            if line == "\n":
                continue
            for varname, metadata in metadata_dict.items():
                # Take first part of line
                value = line[: metadata.column_width]
                # Convert to correct type
                try:
                    converted_value = metadata.dtype(value)
                except ValueError:
                    if not metadata.optional:
                        troublesome.add(varname)
                    converted_value = None if metadata.optional else np.nan
                # Add to results
                if converted_value is not None:
                    results[varname].append(converted_value)
                # Truncate line
                line = line[metadata.column_width :]
        if len(troublesome) > 0:
            raise ValueError(f"Had trouble reading the variables: {troublesome}")
    return results
