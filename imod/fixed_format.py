from dataclasses import dataclass
from numbers import Number
from typing import Dict, Union
from pathlib import Path

import numpy as np
import warnings


@dataclass
class VariableMetaData:
    column_width: int
    min_value: Number
    max_value: Number
    dtype: type


def format_fixed_width(value, metadata):
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
    file: Union[str, Path], metadata_dict: Dict[str, VariableMetaData]
):
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
    results = {}
    for key in metadata_dict:
        results[key] = []

    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                continue
            troublesome = set()
            for varname, metadata in metadata_dict.items():
                # Take first part of line
                value = line[: metadata.column_width]
                # Convert to correct type
                try:
                    converted_value = metadata.dtype(value)
                except ValueError:
                    troublesome.add(varname)
                    converted_value = np.nan
                # Add to results
                results[varname].append(converted_value)
                # Truncate line
                line = line[metadata.column_width :]
        if len(troublesome) > 0:
            warnings.warn(
                f"Had trouble reading the variables: {troublesome}",
                UserWarning,
            )
    return results
