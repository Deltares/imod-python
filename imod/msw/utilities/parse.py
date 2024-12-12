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
            ll = line[0 : line.find("!")].split("=")
            if len(ll) > 1:
                key = ll[0].strip()
                value = _try_parsing_string_to_number(ll[1].strip())
                out[key] = value
    return out
