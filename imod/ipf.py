import pandas as pd
from imod import util
from glob import glob
from pathlib import Path
from collections import OrderedDict
import numpy as np

# the simple CSV format IPF cannot be read like this, it is best to use pandas.read_csv
def read(path):
    """Read an IPF file to a pandas.DataFrame"""
    with open(path) as f:
        nrow = int(f.readline().strip())
        ncol = int(f.readline().strip())
        colnames = [f.readline().strip().strip("'").strip('"') for _ in range(ncol)]
        indexcol, ext = map(str.strip, f.readline().split(","))
        indexcol = int(indexcol) - 1  # convert to 0-based index
        if indexcol > 0:
            df = pd.read_csv(f, header=None, names=colnames, nrows=nrow)
            s = df.iloc[:, indexcol]
            for filename in s:
                # TODO should we resolve relative paths to the working dir or the IPF?
                # now it is relative to the working dir
                path_assoc = filename + "." + ext
                df_assoc = read_associated(path_assoc)
                # TODO work out how to join
        else:
            df = pd.read_csv(f, header=None, names=colnames, nrows=nrow)
    return df


# TODO check if this also supports space separators like the iMOD manual 9.7.1
# TODO allow passing kwargs to pandas.read_csv, like delim_whitespace=True
# TODO check if it can infer the datetime formats
def read_associated(path):
    """Read a file that is associated from an IPF file"""
    with open(path) as f:
        nrow = int(f.readline().strip())
        ncol, itype = map(int, map(str.strip, f.readline().split(",")))
        na_values = OrderedDict()
        for _ in range(ncol):
            # TODO need to use csv parsing to handle "name, unit", -
            colname, nodata = map(str.strip, f.readline().split(","))
            colname = colname.strip("'").strip('"')
            na_values[colname] = [nodata, "-"]  # - seems common enough to ignore
        colnames = list(na_values.keys())
        itype_kwargs = {}
        if itype == 1:
            # check if first column is time in in [yyyymmdd] or [yyyymmddhhmmss]
            itype_kwargs["parse_dates"] = True  # this parses the index
        elif itype in (2, 3):
            # enforce first column is a float
            itype_kwargs["dtype"] = {colnames[0]: np.float64}
        elif itype == 4:
            # enforce first 3 columns are float
            itype_kwargs["dtype"] = {
                colnames[0]: np.float64,
                colnames[1]: np.float64,
                colnames[2]: np.float64,
            }
        df = pd.read_csv(
            f,
            header=None,
            names=colnames,
            nrows=nrow,
            na_values=na_values,
            **itype_kwargs
        )
    return df


def load(path):
    """Load one or more IPF files to a single pandas.DataFrame
    
    The different IPF files can be from different model layers,
    but otherwise have to have identical columns"""
    # convert since for Path.glob non-relative patterns are unsupported
    if isinstance(path, Path):
        path = str(path)

    paths = [Path(p) for p in glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError("Could not find any files matching {}".format(path))

    dfs = []
    for p in paths:
        layer = util.decompose(p).get("layer")
        df = read(p)
        if layer is not None:
            df["layer"] = layer
        dfs.append(df)

    bigdf = pd.concat(dfs, ignore_index=True, sort=False)
    # concat sorts the columns, restore original order, see pandas issue 4588
    bigdf = bigdf.reindex(dfs[0].columns, axis=1)
    return bigdf


def write(path, df):
    """Write a pandas.DataFrame to an IPF file"""
    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write("{}\n{}\n".format(nrecords, nfields))
        [f.write("{}\n".format(colname)) for colname in df.columns]
        f.write("0,TXT\n")
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
    df.to_csv(path, index=False, header=False, mode="a")


def save(path, df):
    """Save a pandas.DataFrame to one or more IPF files, split per layer"""
    d = util.decompose(path)
    d["extension"] = ".ipf"
    d["directory"].mkdir(exist_ok=True, parents=True)

    if "layer" in df.columns:
        for layer, group in df.groupby("layer"):
            d["layer"] = layer
            fn = util.compose(d)
            write(fn, group)
    else:
        fn = util.compose(d)
        write(fn, df)
