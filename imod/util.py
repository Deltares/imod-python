import re
from datetime import datetime
import numpy as np
from collections import OrderedDict
from pathlib import Path


def decompose(path):
    """Parse a path, returning a dict of the parts,
    following the iMOD conventions"""
    if isinstance(path, str):
        path = Path(path)

    parts = path.stem.split('_')
    name = parts[0]
    assert name != '', ValueError("Name cannot be empty")
    d = OrderedDict()
    d['extension'] = path.suffix
    d['directory'] = path.parent
    d['name'] = name
    if len(parts) == 1:
        return d
    try:
        # TODO try pandas parse date?
        d['time'] = np.datetime64(datetime.strptime(parts[1], '%Y%m%d%H%M%S'))
    except ValueError:
        pass  # no time in dict
    # layer is always last
    p = re.compile(r'^l\d+$', re.IGNORECASE)
    if p.match(parts[-1]):
        d['layer'] = int(parts[-1][1:])
    return d


def compose(d):
    """From a dict of parts, construct a filename,
    following the iMOD conventions"""
    haslayer = 'layer' in d
    hastime = 'time' in d
    if hastime:
        d['timestr'] = d['time'].strftime('%Y%m%d%H%M%S')
    if haslayer:
        d['layer'] = int(d['layer'])
        if hastime:
            s = '{name}_{timestr}_l{layer}{extension}'.format(**d)
        else:
            s = '{name}_l{layer}{extension}'.format(**d)
    else:
        if hastime:
            s = '{name}_{timestr}{extension}'.format(**d)
        else:
            s = '{name}{extension}'.format(**d)
    if 'directory' in d:
        return d['directory'].joinpath(s)
    else:
        return s
