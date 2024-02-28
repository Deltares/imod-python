import collections
import functools
from typing import Any, Dict, List

def initialize_nested_dict(depth: int) -> collections.defaultdict:
    """
    Initialize a nested dict with a fixed depth

    Parameters
    ----------
    depth : int
        depth of returned nested dict

    Returns
    -------
    nested defaultdicts of n depth

    """
    # In explicit form, say we have ndims=5
    # Then, writing it out, we get:
    # a = partial(defaultdict, {})
    # b = partial(defaultdict, a)
    # c = partial(defaultdict, b)
    # d = defaultdict(c)
    # This can obviously be done iteratively.
    if depth == 0:
        return {}
    elif depth == 1:
        return collections.defaultdict(dict)
    else:
        d = functools.partial(collections.defaultdict, dict)
        for _ in range(depth - 2):
            d = functools.partial(collections.defaultdict, d)
        return collections.defaultdict(d)


def set_nested(d: collections.defaultdict, keys: List[str], value: Any) -> None:
    """
    Set in the deepest dict of a set of nested dictionaries, as created by the
    initialize_nested_dict function above.

    Mutates d.

    Parameters
    ----------
    d : (Nested dict of) dict
    keys : list of keys
        Each key is a level of nesting
    value : dask array, typically

    Returns
    -------
    None
    """
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        set_nested(d[keys[0]], keys[1:], value)


def append_nested_dict(dict1: Dict, dict2: Dict) -> None:
    """
    Recursively walk through two dicts to append dict2 to dict1.

    Mutates dict1

    Modified from:
    https://stackoverflow.com/a/58742155

    Parameters
    ----------
    dict1 : nested dict
        Nested dict to be appended to
    dict2 : nested dict
        Nested dict to append

    """
    for key, val in dict1.items():
        if isinstance(val, dict):
            if key in dict2 and isinstance(dict2[key], dict):
                append_nested_dict(dict1[key], dict2[key])
        else:
            if key in dict2:
                dict1[key] = dict2[key]

    for key, val in dict2.items():
        if key not in dict1:
            dict1[key] = val


def sorted_nested_dict(d: Dict) -> Dict:
    """
    Sorts a variably nested dict (of dicts) by keys.

    Each dictionary will be sorted by its keys.

    Parameters
    ----------
    d : (Nested dict of) dict

    Returns
    -------
    sorted_lists : list (of lists)
        Values sorted by keys, matches the nesting of d.
    """
    firstkey = next(iter(d.keys()))
    if not isinstance(d[firstkey], dict):  # Base case
        return [v for (_, v) in sorted(d.items(), key=lambda t: t[0])]
    else:  # Recursive case
        return [
            sorted_nested_dict(v) for (_, v) in sorted(d.items(), key=lambda t: t[0])
        ]