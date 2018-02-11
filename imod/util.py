import os
import re
from datetime import datetime
import numpy as np
from collections import OrderedDict


def decompose(path):
    """Parse a filename, returning a dict of the parts,
    following the iMOD conventions"""
    noext, ext = os.path.splitext(path)
    dirpath, filename = os.path.split(noext)
    parts = filename.split('_')
    name = parts[0]
    assert name != '', ValueError("Name cannot be empty")
    d = OrderedDict()
    d['extension'] = ext
    d['directory'] = dirpath
    d['name'] = name
    if len(parts) == 1:
        return d
    try:
        # TODO try pandas parse date?
        d['time'] = np.datetime64(datetime.strptime(parts[1], '%Y%m%d%H%M%S'))
    except ValueError:
        pass  # no time in dict
    # layer is always last
    p = re.compile('^l\d+$', re.IGNORECASE)
    if p.match(parts[-1]):
        d['layer'] = int(parts[-1][1:])
    return d


def compose(d):
    extension = d['extension']
    haslayer = 'layer' in d
    hastime = 'time' in d
    if hastime:
        d['timestr'] = d['time'].strftime('%Y%m%d%H%M%S')
    if haslayer:
        if hastime:
            s = '{name}_{timestr}_l{layer}.{extension}'.format(**d)
        else:
            s = '{name}_l{layer}.{extension}'.format(**d)
    else:
        if hastime:
            s = '{name}_{timestr}.{extension}'.format(**d)
        else:
            s = '{name}.{extension}'.format(**d)
    if 'directory' in d:
        return os.path.join(d['directory'], s)
    else:
        return s
