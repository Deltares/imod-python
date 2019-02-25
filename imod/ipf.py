import csv
from collections import OrderedDict
from glob import glob
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from imod import util


# Maybe look at dask dataframes, if we run into very large tabular datasets:
# http://dask.pydata.org/en/latest/examples/dataframe-csv.html
# the simple CSV format IPF cannot be read like this, it is best to use pandas.read_csv
def read(path, kwargs={}, assoc_kwargs={}):
    """
    Load one IPF file to a single pandas.DataFrame, including associated (TXT) files.

    Parameters
    ----------
    path: pathlib.Path or str
        globpath for IPF files to load.
    kwargs : dict
        Dictionary containing the `pandas.read_csv()` keyword arguments for the
        IPF files (e.g. `whitespace_delimited: True`)
    assoc_kwargs: dict
        Dictionary containing the `pandas.read_csv()` keyword arguments for the
        associated (TXT) files (e.g. `whitespace_delimited: True`)

    Returns
    -------
    pandas.DataFrame
    """

    if isinstance(path, str):
        path = Path(path)

    with open(path) as f:
        nrow = int(f.readline().strip())
        ncol = int(f.readline().strip())
        colnames = [f.readline().strip().strip("'").strip('"') for _ in range(ncol)]
        line = f.readline()
        try:
            # csv.reader parse one line
            # this catches commas in quotes
            indexcol, ext = map(str.strip, next(csv.reader([line])))
        except ValueError:  # then try whitespace delimited
            indexcol, ext = map(str.strip, next(csv.reader([line], delimiter=" ")))

        indexcol = int(indexcol)
        if indexcol > 1:
            df = pd.read_csv(f, header=None, names=colnames, nrows=nrow, **kwargs)
            dfs = []
            for row in df.itertuples():
                filename = row[indexcol]
                # associate paths are relative to the ipf
                path_assoc = path.parent.joinpath(filename + "." + ext)
                # Note that these kwargs handle all associated files, which might differ
                # within an IPF. If this happens we could consider supporting a dict
                # or function that maps assoc filenames to different kwargs.
                df_assoc = read_associated(path_assoc, assoc_kwargs)

                for name, value in zip(colnames, row[1:]):  # ignores df.index in row
                    df_assoc[name] = value

                dfs.append(df_assoc)
            df = pd.concat(dfs, ignore_index=True, sort=False)
        else:
            df = pd.read_csv(f, header=None, names=colnames, nrows=nrow, **kwargs)
    return df


def read_associated(path, kwargs={}):
    """
    Read an IPF associated file (TXT).

    Parameters
    ----------
    path : pathlib.Path or str
        Path to associated file.
    kwargs : dict
        Dictionary containing the `pandas.read_csv()` keyword arguments for the
        associated (TXT) file (e.g. `whitespace_delimited: True`)

    Returns
    -------
    pandas.DataFrame
    """

    # deal with e.g. incorrect capitalization
    if isinstance(path, str):
        path = Path(path)
    path = path.resolve()

    with open(path) as f:
        nrow = int(f.readline().strip())
        line = f.readline()
        try:
            # csv.reader parse one line
            # this catches commas in quotes
            ncol, itype = map(int, map(str.strip, next(csv.reader([line]))))
        # itype can be implicit, in which case it's a timeseries
        except ValueError:  
            try:
                ncol = int(line.strip()) 
                itype = 1
            except ValueError:  # then try whitespace delimited
                ncol, itype = map(
                    int, map(str.strip, next(csv.reader([line], delimiter=" ")))
                )
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
        metadata = pd.read_csv(StringIO(lines), header=None, nrows=ncol, **kwargs)
        # header description possibly includes nodata
        usecols = np.arange(ncol)[pd.notnull(metadata[0])]
        metadata = metadata.iloc[usecols, :]

        for colname, nodata in metadata.values:
            na_values[colname] = [nodata, "-"]  # "-" seems common enough to ignore

        colnames = list(na_values.keys())
        itype_kwargs = {}
        if itype == 1:  # Timevariant information: timeseries
            # check if first column is time in [yyyymmdd] or [yyyymmddhhmmss]
            itype_kwargs["parse_dates"] = [0]  # this parses the first column
            itype_kwargs["infer_datetime_format"] = True
        elif itype == 2:  # 1D borehole
            # enforce first column is a float
            itype_kwargs["dtype"] = {colnames[0]: np.float64}
        elif itype == 3:  # cpt
            # all columns must be numeric
            itype_kwargs["dtype"] = {colname: np.float64 for colname in colnames}
        elif itype == 4:  # 3D borehole
            # enforce first 3 columns are float
            itype_kwargs["dtype"] = {
                colnames[0]: np.float64,
                colnames[1]: np.float64,
                colnames[2]: np.float64,
            }

        itype_kwargs.update(kwargs)

        df = pd.read_csv(
            f,
            header=None,
            names=colnames,
            usecols=usecols,
            nrows=nrow,
            na_values=na_values,
            **itype_kwargs,
        )
    return df


def load(path, kwargs={}, assoc_kwargs={}):
    """
    Load one or more IPF files to a single pandas.DataFrame, including associated
    (TXT) files.

    The different IPF files can be from different model layers,
    and column names may differ between them.

    Parameters
    ----------
    path: pathlib.Path or str
        globpath for IPF files to load.
    kwargs : dict
        Dictionary containing the `pandas.read_csv()` keyword arguments for the
        IPF files (e.g. `whitespace_delimited: True`)
    assoc_kwargs: dict
        Dictionary containing the `pandas.read_csv()` keyword arguments for the
        associated (TXT) files (e.g. `whitespace_delimited: True`)

    Returns
    -------
    pandas.DataFrame
    """

    # convert since for Path.glob non-relative patterns are unsupported
    if isinstance(path, Path):
        path = str(path)

    paths = [Path(p) for p in glob(path)]
    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")

    dfs = []
    for p in paths:
        layer = util.decompose(p).get("layer")
        df = read(p, kwargs, assoc_kwargs)
        if layer is not None:
            df["layer"] = layer
        dfs.append(df)

    bigdf = pd.concat(dfs, ignore_index=True, sort=False)  # this sorts in pandas < 0.23
    return bigdf


def _coerce_itype(itype):
    """Changes string itype to int"""
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


def _lower(colnames):
    """Lowers colnames, checking for uniqueness"""
    lowered_colnames = [s.lower() for s in colnames]
    if len(set(lowered_colnames)) != len(colnames):
        seen = set()
        for name in lowered_colnames:
            if name in seen:
                raise ValueError(f'Column name "{name}" is not unique, after lowering.')
            else:
                seen.add(name)
    return lowered_colnames


def write_assoc(path, df, itype=1, nodata=1.0e20):
    """
    Writes a single IPF associated (TXT) file.

    Parameters
    ----------
    path : pathlib.Path or str
        Path for the written associated file.
    df : pandas.DataFrame
        DataFrame containing the data to write.
    itype : int or str
        IPF type.
        Possible values, either integer or string:

        1 : "timeseries"
        2 : "borehole1d"
        3 : "cpt"
        4 : "borehole3d"
    nodata : float
        The value given to nodata values. These are generally NaN (Not-a-Number)
        in pandas, but this leads to errors in iMOD(FLOW) for IDFs.
        Defaults to value of 1.0e20 instead.

    Returns
    -------
    None
        Writes a file.
    """

    itype = _coerce_itype(itype)
    required_columns = {
        1: ["time"],
        2: ["top"],
        3: ["top"],
        4: ["x_offset", "y_offset", "top"],
    }

    # Ensure columns are in the right order for the itype
    colnames = _lower(list(df))
    df.columns = colnames
    columnorder = []
    for colname in required_columns[itype]:
        assert colname in colnames, f'given itype requires column "{colname}"'
        colnames.remove(colname)
        columnorder.append(colname)
    columnorder += colnames

    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write(f"{nrecords}\n{nfields},{itype}\n")
        for colname in columnorder:
            if "," in colname or " " in colname:
                colname = '"' + colname + '"'
            f.write(f"{colname},{nodata}\n")
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163

    df = df.fillna(nodata)
    df = df[columnorder]
    df.to_csv(
        path,
        index=False,
        header=False,
        mode="a",
        date_format="%Y%m%d%H%M%S",
        quoting=csv.QUOTE_NONE,
    )


def write(path, df, indexcolumn=0, assoc_ext="txt"):
    """
    Writes a single IPF file.

    Parameters
    ----------
    path : pathlib.Path or str
        path of the written IPF file.
        Any associated files are written relative to this path, based on the ID
        column.
    df : pandas.DataFrame
        DataFrame containing the data to write.
    indexcolumn : integer
        number of the column containg the paths to the associated (TXT) files.
        Defaults to a value of 0 (no associated files).
    assoc_ext : str
        Extension of the associated files. Defaults to "txt".

    Returns
    -------
    None
        Writes a file.
    """

    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write(f"{nrecords}\n{nfields}\n")
        for colname in df.columns:
            if "," in colname or " " in colname:
                colname = '"' + colname + '"'
            f.write(f"{colname}\n")
        f.write(f"{indexcolumn},{assoc_ext}\n")
    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
    df.to_csv(path, index=False, header=False, mode="a", quoting=csv.QUOTE_NONE)


def _is_single_value(group):
    return len(pd.unique(group)) == 1


def _compose_ipf(path, df, itype, assoc_ext, nodata=1.0e20):
    """
    When itype is not None, breaks down the pandas DataFrame into its IPF part
    and its associated TXT files, creating the IPF data structure.

    Parameters
    ----------
    path : pathlib.Path or str
        path of the written IPF file.
        Any associated files are written relative to this path, based on the ID
        column.
    df : pandas.DataFrame
        DataFrame containing the data to write.
    itype : int or str or None
        If `None` no associated files are written.
        Other possible values, either integer or string:
        
        * `1` or `"timeseries"`
        * `2` or `"borehole1d"`
        * `3` or `"cpt"`
        * `4` or `"borehole3d"`
    assoc_ext : str
        Extension of the associated files. Normally ".txt".
    nodata : float
        The value given to nodata values. These are generally NaN (Not-a-Number)
        in pandas, but this leads to errors in iMOD(FLOW) for IDFs.
        Defaults to value of 1.0e20 instead.

    Returns
    -------
    None
        Writes files.
    """
    if itype is None:
        write(path, df)
    else:
        itype = _coerce_itype(itype)
        colnames = _lower(list(df))
        df.columns = colnames
        for refname in ["x", "y", "id"]:
            assert refname in colnames, f'given itype requires column "{refname}"'
            colnames.remove(refname)

        grouped = df.groupby("id")
        assert (
            grouped["x"].apply(_is_single_value).all()
        ), "column x contains more than one value per id"
        assert (
            grouped["y"].apply(_is_single_value).all()
        ), "column y contains more than one value per id"
        # get columns that have only one value within a group, to save them in ipf
        ipf_columns = [
            (colname, "first")
            for colname in colnames
            if grouped[colname].apply(_is_single_value).all()
        ]

        for idcode, group in grouped:
            assoc_path = path.parent.joinpath(str(idcode) + "." + str(assoc_ext))
            assoc_path.parent.mkdir(parents=True, exist_ok=True)
            selection = [colname for colname in colnames if colname not in ipf_columns]
            out_df = group[selection]
            write_assoc(assoc_path, out_df, itype, nodata)

        # ensures right order for x, y, id; so that also indexcolumn == 3
        agg_kwargs = OrderedDict([("x", "first"), ("y", "first"), ("id", "first")])
        agg_kwargs.update(ipf_columns)
        agg_df = grouped.agg(agg_kwargs)
        # Quote so spaces don't mess up paths
        agg_df["id"] = '"' + agg_df["id"] + '"'
        write(path, agg_df, 3, assoc_ext)


def save(path, df, itype=None, assoc_ext="txt", nodata=1.0e20):
    """
    Saves the contents of a pandas DataFrame to one or more IPF files, and
    associated (TXT) files.

    Can write multiple IPF files if one of the columns is named "layer".
    In turn, multiple associated (TXT) files may written for each of these IPF
    files.

    Parameters
    ----------
    path : pathlib.Path or str
        path of the written IPF file.
        Any associated files are written relative to this path, based on the ID
        column.
    df : pandas.DataFrame
        DataFrame containing the data to write.
    itype : int or str or None
        IPF type. Defaults to `None`, in which case no associated files are
        created. Possible other values, either integer or string:
        
        * `1` or `"timeseries"`
        * `2` or `"borehole1d"`
        * `3` or `"cpt"`
        * `4` or `"borehole3d"`
    assoc_ext : str
        Extension of the associated files. Defaults to "txt".
    nodata : float
        The value given to nodata values. These are generally NaN (Not-a-Number)
        in pandas, but this leads to errors in iMOD(FLOW) for IDFs.
        Defaults to value of 1.0e20 instead.

    Returns
    -------
    None
        Writes files.
    """

    d = util.decompose(path)
    d["extension"] = ".ipf"
    d["directory"].mkdir(exist_ok=True, parents=True)

    colnames = _lower(list(df))
    df.columns = colnames
    if "layer" in colnames:
        for layer, group in df.groupby("layer"):
            d["layer"] = layer
            fn = util.compose(d)
            _compose_ipf(fn, group, itype, assoc_ext, nodata)
    else:
        fn = util.compose(d)
        _compose_ipf(fn, df, itype, assoc_ext, nodata)
