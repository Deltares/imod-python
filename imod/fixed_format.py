from dataclasses import dataclass
from numbers import Number


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
        format_string = "{:" + f"+{metadata.column_width}.{decimal_number_width}f" + "}"
    else:
        raise TypeError(f"dtype {metadata.dtype} is not supported")

    converted_value = metadata.dtype(value)
    return format_string.format(converted_value)
