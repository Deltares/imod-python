import os
import csv
import pandas as pd
from glob import glob
from imod import util

# TODO, check this implementation with the format specification in the iMOD manual
def read(path):
    """Read an IPF file to a pandas.DataFrame"""
    with open(path) as f:
        nrow = int(f.readline().strip())
        ncol = int(f.readline().strip())
        colnames = [f.readline().strip().strip("'").strip('"') for _ in range(ncol)]
        _ = f.readline()  # links to other files not handled
        df = pd.read_csv(f, header=None, names=colnames, nrows=nrow)
    return df


def load(path):
    """Load one or more IPF files to a single pandas.DataFrame
    
    The different IPF files can be from different model layers,
    but otherwise have to have identical columns"""
    paths = glob(path)
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(
            'Could not find any files matching {}'.format(path))

    dfs = []
    for p in paths:
        layer = util.decompose(p).get('layer')
        df = read(p)
        if layer is not None:
            df['layer'] = layer
        dfs.append(df)

    bigdf = pd.concat(dfs, ignore_index=True)
    # concat sorts the columns, restore original order, see pandas issue 4588
    bigdf = bigdf.reindex(dfs[0].columns, axis=1)
    return bigdf


def write(path, df):
    """Write a pandas.DataFrame to an IPF file"""
    nrecords, nfields = df.shape
    with open(path, 'w') as f:
        f.write('{}\n{}\n'.format(nrecords, nfields))
        [f.write('{}\n'.format(colname)) for colname in df.columns]
        f.write('0,TXT\n')
        df.to_csv(f, index=False, header=False)


def save(path, df):
    """Save a pandas.DataFrame to one or more IPF files, split per layer"""
    d = util.decompose(path)
    d['extension'] = 'ipf'
    dirpath = d['directory']
    if dirpath != '':  # would give a FileNotFoundError
        os.makedirs(d['directory'], exist_ok=True)

    if 'layer' in df.columns:
        for layer, group in df.groupby('layer'):
            d['layer'] = layer
            fn = util.compose(d)
            write(fn, group)
    else:
        fn = util.compose(d)
        write(fn, df)
