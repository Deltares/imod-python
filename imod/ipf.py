import pandas as pd
import re
from imod import util
from glob import glob
from pathlib import Path
from collections import OrderedDict
import numpy as np
from io import StringIO

# Maybe look at dask dataframes, if we run into very large tabular datasets:
# http://dask.pydata.org/en/latest/examples/dataframe-csv.html
# the simple CSV format IPF cannot be read like this, it is best to use pandas.read_csv
def read(path):
    """Read an IPF file to a pandas.DataFrame"""

    if isinstance(path, str):
        path = Path(path)

    with open(path) as f:
        nrow = int(f.readline().strip())
        ncol = int(f.readline().strip())
        colnames = [f.readline().strip().strip("'").strip('"') for _ in range(ncol)]
        indexcol, ext = map(str.strip, f.readline().split(","))
        indexcol = int(indexcol)
        if indexcol > 1:
            df = pd.read_csv(f, header=None, names=colnames, nrows=nrow)
            dfs = []
            for row in df.itertuples():
                filename = row[indexcol]
                # associate paths are relative to "mother" ipf
                path_assoc = path.join_path(filename + "." + ext)
                df_assoc = read_associated(path_assoc)

                for name, value in zip(colnames, row[1:]):  # ignores df.index in row
                    df_assoc[name] = value

                dfs.append(df_assoc)
            df = pd.concat(dfs, ignore_index=True, sort=False)
        else:
            df = pd.read_csv(f, header=None, names=colnames, nrows=nrow)
    return df


# TODO check if this also supports space separators like the iMOD manual 9.7.1
# TODO allow passing kwargs to pandas.read_csv, like delim_whitespace=True
# TODO check if it can infer the datetime formats
def read_associated(path):
    """Read a file that is associated from an IPF file"""

    # deal with e.g. incorrect capitalization
    if isinstance(path, str):
        path = Path(path)
    path = path.resolve()

    with open(path) as f:
        nrow = int(f.readline().strip())
        ncol, itype = map(int, map(str.strip, f.readline().split(",")))
        na_values = OrderedDict()

        # use pandas for csv parsing: stuff like commas within quotes
        # this is a workaround for a pandas bug, probable related issue:
        # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
        lines = []
        for _ in range(ncol):
            lines.append(f.readline())
        lines = "".join(lines)
        # Normally, this ought to work:
        # metadata = pd.read_csv(f, header=None, nrows=ncol).values
        metadata = pd.read_csv(StringIO(lines), header=None, nrows=ncol)
        # header description possibly includes nodata
        usecols = np.arange(ncol)[pd.notnull(metadata[0])]
        metadata = metadata.iloc[usecols, :]

        for colname, nodata in metadata.values:
            na_values[colname] = [nodata, "-"]  # - seems common enough to ignore

        colnames = list(na_values.keys())
        itype_kwargs = {}
        if itype == 1:  # Timevariant information: timeseries
            # check if first column is time in in [yyyymmdd] or [yyyymmddhhmmss]
            itype_kwargs["parse_dates"] = True  # this parses the index
            itype_kwargs["infer_datetime_format"] = True
        elif itype in (2, 3):  # 1D borehole or Cone Penetration
            # enforce first column is a float
            itype_kwargs["dtype"] = {colnames[0]: np.float64}
        elif itype == 4:  # 3D borehole
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
            usecols=usecols,
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
    # TODO: Decide why reindex? This throws out columns of other dataframes that may
    # not be present in the first dataframe.
    # bigdf = bigdf.reindex(dfs[0].columns, axis=1)
    return bigdf


def write(path, df, indexcolumn=0):
    """Write a pandas.DataFrame to an IPF file"""
    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write("{}\n{}\n".format(nrecords, nfields))
        [f.write("{}\n".format(colname)) for colname in df.columns]
        f.write("{},TXT\n".format(indexcolumn))
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
    df.to_csv(path, index=False, header=False, mode="a")


def save(path, df, itype=None):
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

   # if itype in [None, 1, 2, 3, 4]:
   #     pass
   # elif itype.lower() == "timeseries":
   #     itype = 1
   # elif itype.lower() == "borehole1d":
   #     itype = 2
   # elif itype.lower() == "cpt":
   #     itype = 3
   # elif itype.lower() == "borehole3d":
   #     itype = 4
   # else
   #     raise ValueError("Invalid IPF itype")

    if itype == 1:
        assert "time" in df.columns
        try:
            df["time"] = df["time"].to_datetime()
    if itype == 2 or itype == 3:
        # lame name? but top is less confusing than e.g. z
        assert "top" in df.columns
    if itype == 4:
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns

    if "layer" in df.columns:
        for layer, group in df.groupby("layer"):
            d["layer"] = layer

            for idcode, regroup in group.groupby("id"):
                write_associated_itype1(path, regroup)

    else:
        fn = util.compose(d)
        write(fn, df)
    


def write_associated_itype1(path, df):
    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write("{}\n{}\n".format(nrecords, nfields))
        [f.write("{},{}\n".format(colname, na_value)) for colname in df.columns]
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
    df.to_csv(path, index=False, header=False, mode="a")


def write_associated_itype2(path, df):
    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write("{}\n{},2\n".format(nrecords, nfields))
        [f.write("{}\n".format(colname)) for colname in df.columns]
        f.write("1,TXT\n")
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
    df.to_csv(path, index=False, header=False, mode="a")

def write_associated_itype2(path, df):
    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write("{}\n{}\n".format(nrecords, nfields))
        [f.write("{}\n".format(colname)) for colname in df.columns]
        f.write("1,TXT\n")
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
    df.to_csv(path, index=False, header=False, mode="a")


def write_associated_itype4(path, df):
    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write("{}\n{}\n".format(nrecords, nfields))
        [f.write("{}\n".format(colname)) for colname in df.columns]
        f.write("1,TXT\n")
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
    df.to_csv(path, index=False, header=False, mode="a")