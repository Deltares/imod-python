from pathlib import Path
from typing import TypeAlias

ScalarType: TypeAlias = str | float | int


def _try_parsing_string_to_number(s: str) -> ScalarType:
    """
    Convert string to number:

    "1" -> 1
    "1.0" -> 1.0
    "a" -> "a"
    """
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def read_para_sim(file: Path | str) -> dict[str, ScalarType]:
    with open(file, "r") as f:
        lines = f.readlines()
        out = {}
        for line in lines:
            # Strip comments starting with "!" and split keys from values at the
            # equals sign.
            key_values = line[0 : line.find("!")].split("=")
            if len(key_values) > 1:
                key = key_values[0].strip()
                value = _try_parsing_string_to_number(key_values[1].strip())
                out[key] = value
    return out
