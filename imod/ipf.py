import pandas as pd
from glob import glob


# TODO, check this implementation with the format specification in the iMOD manual
def read(path):
    with open(path) as f:
        nrow = int(f.readline().strip())
        ncol = int(f.readline().strip())
        colnames = [f.readline().strip() for _ in range(ncol)]
        _ = f.readline()  # links to other files not handled
        df = pd.read_csv(f, header=None, names=colnames, nrows=nrow)
    return df


def write(path, df):
    nrecords, nfields = df.shape
    with open(path, 'w') as f:
        f.write('{}\n{}\n'.format(nrecords, nfields))
        [f.write('{}\n'.format(colname)) for colname in list(df)]
        f.write('0,TXT\n')
        df.to_csv(f, index=False, header=False)


def load(globpath):
    paths = glob(globpath)

    dfs = []
    for path in paths:
        layer = parse_filename(path)['layer']
        df = readipf(path)
        df['layer'] = layer
        dfs.append(df)

    bigdf = pd.concat(dfs, ignore_index=True)
    return bigdf

# TODO save function
