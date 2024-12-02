from pathlib import Path


def _try_parsing_string_to_value(s: str):
    """
    Convert string to value:

    "1" -> 1
    "1.0" -> 1.0
    "a" -> "a"
    """
    try:
        value = int(s)
    except ValueError:
        try:
            value = float(s)
        except ValueError:
            value = s
    return value


def read_para_sim(file: Path | str) -> dict[str, str]:
    with open(file, "r") as f:
        lines = f.readlines()
        out = {}
        for line in lines:
            ll = line[0 : line.find("!")].split("=")
            if len(ll) > 1:
                key = ll[0].strip()
                value = _try_parsing_string_to_value(ll[1].strip())
                out[key] = value
    return out
