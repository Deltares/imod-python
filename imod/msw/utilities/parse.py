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


def correct_unsa_svat_path(unsa_path: ScalarType) -> str:
    """
    Correct the path to the UNSA SVAT executable in the parameter file. Drop any
    quotes and trailing backslashes, and replace any dollar signs. These could
    have been added by ``MetaSwapModel._render_unsaturated_database_path``.
    """
    if not isinstance(unsa_path, str):
        raise TypeError(f"Unexcepted type for unsa_path, expected str, got {unsa_path}")

    unsa_path = unsa_path.replace('"', "")

    if unsa_path.endswith("\\"):
        unsa_path = unsa_path[:-1]

    if unsa_path.startswith("$"):
        unsa_path = unsa_path.replace("$", "./")

    return unsa_path


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

    out["unsa_svat_path"] = correct_unsa_svat_path(out["unsa_svat_path"])

    return out
