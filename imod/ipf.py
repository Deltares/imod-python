import pandas as pd
import re
import numpy as np
from collections import OrderedDict
from glob import glob
from io import StringIO
from pathlib import Path
from imod import util

# Maybe look at dask dataframes, if we run into very large tabular datasets:
# http://dask.pydata.org/en/latest/examples/dataframe-csv.html
# the simple CSV format IPF cannot be read like this, it is best to use pandas.read_csv
def read(path, kwargs={}, assoc_kwargs={}):
    """Read an IPF file to a pandas.DataFrame"""

    if isinstance(path, str):
        path = Path(path)

    with open(path) as f:
        nrow = int(f.readline().strip())
        ncol = int(f.readline().strip())
        colnames = [f.readline().strip().strip("'").strip('"') for _ in range(ncol)]
        line = f.readline()
        # TODO: are space and comma the only allowed separators?
        try:
            indexcol, ext = map(str.strip, line.split(","))
        except ValueError:
            indexcol, ext = map(str.strip, line.split())

        indexcol = int(indexcol)
        if indexcol > 1:
            df = pd.read_csv(f, header=None, names=colnames, nrows=nrow, **kwargs)
            dfs = []
            for row in df.itertuples():
                filename = row[indexcol]
                # associate paths are relative to "mother" ipf
                path_assoc = path.parent.joinpath(filename + "." + ext)
                df_assoc = read_associated(path_assoc, assoc_kwargs)

                for name, value in zip(colnames, row[1:]):  # ignores df.index in row
                    df_assoc[name] = value

                dfs.append(df_assoc)
            df = pd.concat(dfs, ignore_index=True, sort=False)
        else:
            df = pd.read_csv(f, header=None, names=colnames, nrows=nrow, **kwargs)
    return df


# TODO check if this also supports space separators like the iMOD manual 9.7.1
# TODO check if it can infer the datetime formats
def read_associated(path, kwargs={}):
    """Read a file that is associated from an IPF file"""
    # TODO: passing on pandas.read_csv kwargs maybe not be enough?
    # since txt's don't have to be consistent within a single ipf ...

    # avoid kwargs mutation
    assoc_kwargs = kwargs.copy()

    # deal with e.g. incorrect capitalization
    if isinstance(path, str):
        path = Path(path)
    path = path.resolve()

    with open(path) as f:
        nrow = int(f.readline().strip())
        line = f.readline()
        try:
            ncol, itype = map(int, map(str.strip, line.split(",")))
        except ValueError:
            ncol, itype = map(int, map(str.strip, line.split()))
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
        # TODO: replace when bugfix is released
        metadata = pd.read_csv(StringIO(lines), header=None, nrows=ncol, **assoc_kwargs)
        # header description possibly includes nodata
        usecols = np.arange(ncol)[pd.notnull(metadata[0])]
        metadata = metadata.iloc[usecols, :]

        for colname, nodata in metadata.values:
            na_values[colname] = [nodata, "-"]  # - seems common enough to ignore

        colnames = list(na_values.keys())
        itype_kwargs = {}
        if itype == 1:  # Timevariant information: timeseries
            # check if first column is time in in [yyyymmdd] or [yyyymmddhhmmss]
            itype_kwargs["parse_dates"] = [0]  # this parses the first column
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

        assoc_kwargs.update(itype_kwargs)

        df = pd.read_csv(
            f,
            header=None,
            names=colnames,
            usecols=usecols,
            nrows=nrow,
            na_values=na_values,
            **assoc_kwargs
        )
    return df


def load(path, kwargs={}, assoc_kwargs={}):
    """Load one or more IPF files to a single pandas.DataFrame
    
    The different IPF files can be from different model layers,
    but otherwise have to have identical columns"""
    # TODO: there's no real necessity to force identical columns?
    # pd.concatenate will just stuff everything together, with a lot of columns
    # the user can then just correct.
    # Because the IPF format mostly relies on column order, there is an
    # extreme lack of consistency for ipf column names. Not supporting
    # heterogeneous column names limits functionality regarding existing
    # IPFs very strongly.

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
        df = read(p, kwargs, assoc_kwargs)
        if layer is not None:
            df["layer"] = layer
        dfs.append(df)

    bigdf = pd.concat(dfs, ignore_index=True, sort=False)
    # concat sorts the columns, restore original order, see pandas issue 4588
    # TODO: Decide why reindex? This throws out columns of other dataframes that may
    # not be present in the first dataframe. See todo comment above.
    # bigdf = bigdf.reindex(dfs[0].columns, axis=1)
    return bigdf


def _coerce_itype(itype):
    if itype in [None, 1, 2, 3, 4]:
        pass
    elif itype.lower() == "timeseries":
        itype = 1
    elif itype.lower() == "borehole1d":
        itype = 2
    elif itype.lower() == "cpt":
        itype = 3
    elif itype.lower() == "borehole3d":
        itype = 4
    else:
        raise ValueError("Invalid IPF itype")
    return itype


def write_assoc(path, df, itype=1, nodata=np.nan):
    # TODO: default nodata value?
    itype = _coerce_itype(itype)
    required_columns = {1: ["time"], 2: ["top"], 3: ["top"], 4: ["x", "y", "z"]}

    # Ensure columns are in the right order for the itype
    colnames = [s.lower() for s in list(df)]
    columnorder = []
    for colname in required_columns[itype]:
        assert colname in df.columns, "given itype requires column {}".format(colname)
        colnames.remove(colname)
        columnorder.append(colname)
    columnorder += colnames

    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write("{}\n{},{}\n".format(nrecords, nfields, itype))
        [f.write("{},{}\n".format(colname, nodata)) for colname in columnorder]
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163

    df = df.fillna(nodata)
    df = df[columnorder]
    df.to_csv(path, index=False, header=False, mode="a", date_format="%Y%m%d%H%M%S")


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


def _is_single_value(group):
    return len(pd.unique(group)) == 1


def _compose_ipf(path, df, itype, assoc_ext, nodata):
    if itype is None:
        write(path, df)
    else:
        itype = _coerce_itype(itype)
        colnames = [s.lower() for s in list(df)]
        for refname in ["x", "y", "id"]:
            assert refname in colnames, "given itype requires column {}".format(refname)
            colnames.remove(refname)

        grouped = df.groupby("id")
        assert (
            grouped["x"].apply(_is_single_value).all()
        ), "column x contains more than one value per id"
        assert (
            grouped["y"].apply(_is_single_value).all()
        ), "column y contains more than one value per id"
        if "layer" in colnames:
            assert (
                grouped["layer"].apply(_is_single_value).all()
            ), "column layer contains more than one value per id"
        # get columns that have only one value within a group, to save them in ipf also

        ipf_columns = [
            (colname, "first")
            for colname in colnames
            if grouped[colname].apply(_is_single_value).all()
        ]

        for idcode, group in grouped:
            assoc_path = path.parent.joinpath(idcode + "." + assoc_ext)
            write_assoc(assoc_path, group, itype, nodata)

        # ensures right order for x, y, id; so that also indexcolumn == 3
        agg_kwargs = OrderedDict([("x", "first"), ("y", "first"), ("id", "first")])
        agg_kwargs.update(ipf_columns)
        agg_df = grouped.agg(agg_kwargs)
        write(path, agg_df, 3)


def save(path, df, itype=None, assoc_ext="txt", nodata=np.nan):
    """Save a pandas.DataFrame to one or more IPF files, split per layer"""
    d = util.decompose(path)
    d["extension"] = ".ipf"
    d["directory"].mkdir(exist_ok=True, parents=True)

    if "layer" in df.columns:
        for layer, group in df.groupby("layer"):
            d["layer"] = layer
            fn = util.compose(d)
            _compose_ipf(fn, group, itype, assoc_ext, nodata)
    else:
        fn = util.compose(d)
        _compose_ipf(fn, df, itype, assoc_ext, nodata)
