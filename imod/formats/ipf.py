"""
Functions for reading and writing iMOD Point Files (IDFs) to ``pandas.DataFrame``.

The primary functions to use are :func:`imod.ipf.read` and
:func:`imod.ipf.save`, though lower level functions are also available.
"""

import collections
import csv
import glob
import io
import pathlib
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from imod import util


def _infer_delimwhitespace(line, ncol):
    n_elem = len(next(csv.reader([line])))
    if n_elem == 1:
        return True
    elif n_elem == ncol:
        return False
    else:
        warnings.warn(
            f"Inconsistent IPF: header states {ncol} columns, first line contains {n_elem}"
        )
        return False


def _read_ipf(path, kwargs=None) -> Tuple[pd.DataFrame, int, str]:
    path = pathlib.Path(path)
    if kwargs is None:
        kwargs = {}

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

        position = f.tell()
        line = f.readline()
        delim_whitespace = _infer_delimwhitespace(line, ncol)
        f.seek(position)

        ipf_kwargs = {
            "delim_whitespace": delim_whitespace,
            "header": None,
            "names": colnames,
            "nrows": nrow,
            "skipinitialspace": True,
        }
        ipf_kwargs.update(kwargs)
        df = pd.read_csv(f, **ipf_kwargs)

    return df, int(indexcol), ext


def _read(path, kwargs=None, assoc_kwargs=None):
    """
    Read one IPF file to a single pandas.DataFrame, including associated (TXT) files.

    Parameters
    ----------
    path: pathlib.Path or str
        globpath for IPF files to read.
    kwargs : dict
        Dictionary containing the ``pandas.read_csv()`` keyword arguments for the
        IPF files (e.g. `{"delim_whitespace": True}`)
    assoc_kwargs: dict
        Dictionary containing the ``pandas.read_csv()`` keyword arguments for the
        associated (TXT) files (e.g. `{"delim_whitespace": True}`)

    Returns
    -------
    pandas.DataFrame
    """
    df, indexcol, ext = _read_ipf(path, kwargs)
    if assoc_kwargs is None:
        assoc_kwargs = {}

    # See if reading associated files is necessary
    if indexcol > 1:
        colnames = df.columns
        # df = pd.read_csv(f, header=None, names=colnames, nrows=nrow, **kwargs)
        dfs = []
        for row in df.itertuples():
            filename = row[indexcol]
            # associate paths are relative to the ipf
            path_assoc = path.parent.joinpath(f"{filename}.{ext}")
            # Note that these kwargs handle all associated files, which might differ
            # within an IPF. If this happens we could consider supporting a dict
            # or function that maps assoc filenames to different kwargs.
            try:  # Capture the error and print the offending path
                df_assoc = read_associated(path_assoc, assoc_kwargs)
            except Exception as e:
                raise type(e)(
                    f'{e}\nWhile reading associated file "{path_assoc}" of IPF file "{path}"'
                ) from e

            # Include records of the "mother" ipf file.
            for name, value in zip(colnames, row[1:]):  # ignores df.index in row
                df_assoc[name] = value
            # Append to list
            dfs.append(df_assoc)
        # Merge into a single whole
        df = pd.concat(dfs, ignore_index=True, sort=False)

    return df


def read_associated(path, kwargs={}):
    """
    Read an IPF associated file (TXT).

    Parameters
    ----------
    path : pathlib.Path or str
        Path to associated file.
    kwargs : dict
        Dictionary containing the ``pandas.read_csv()`` keyword arguments for the
        associated (TXT) file (e.g. `{"delim_whitespace": True}`).

    Returns
    -------
    pandas.DataFrame
    """

    # deal with e.g. incorrect capitalization
    path = pathlib.Path(path).resolve()

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

        # use pandas for csv parsing: stuff like commas within quotes
        # this is a workaround for a pandas bug, probable related issue:
        # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
        lines = [f.readline() for _ in range(ncol)]
        delim_whitespace = _infer_delimwhitespace(lines[0], 2)
        # Normally, this ought to work:
        # metadata = pd.read_csv(f, header=None, nrows=ncol).values
        # TODO: replace when bugfix is released
        # try both comma and whitespace delimited, everything can be be mixed
        # in a single file...
        lines = "".join(lines)

        # TODO: find out whether this can be replace by csv.reader
        # the challenge lies in replacing the pd.notnull for nodata values.
        # is otherwise quite a bit faster for such a header block.
        metadata_kwargs = {
            "delim_whitespace": delim_whitespace,
            "header": None,
            "nrows": ncol,
            "skipinitialspace": True,
        }
        metadata_kwargs.update(kwargs)
        metadata = pd.read_csv(io.StringIO(lines), **metadata_kwargs)
        # header description possibly includes nodata
        usecols = np.arange(ncol)[pd.notnull(metadata[0])]
        metadata = metadata.iloc[usecols, :]

        # Collect column names and nodata values
        colnames = []
        na_values = collections.OrderedDict()
        for colname, nodata in metadata.values:
            na_values[colname] = [nodata, "-"]  # "-" seems common enough to ignore
            if isinstance(colname, str):
                colnames.append(colname.strip())
            else:
                colnames.append(colname)

        # Sniff the first line of the data block
        position = f.tell()
        line = f.readline()
        f.seek(position)
        delim_whitespace = _infer_delimwhitespace(line, ncol)

        itype_kwargs = {
            "delim_whitespace": delim_whitespace,
            "header": None,
            "names": colnames,
            "usecols": usecols,
            "nrows": nrow,
            "na_values": na_values,
            "skipinitialspace": True,
        }
        if itype == 1:  # Timevariant information: timeseries
            # check if first column is time in [yyyymmdd] or [yyyymmddhhmmss]
            itype_kwargs["dtype"] = {colnames[0]: str}
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
        df = pd.read_csv(f, **itype_kwargs)

    if nrow > 0 and itype == 1:
        time_column = colnames[0]
        len_date = len(df[time_column].iloc[0])
        if len_date == 14:
            df[time_column] = pd.to_datetime(df[time_column], format="%Y%m%d%H%M%S")
        elif len_date == 8:
            df[time_column] = pd.to_datetime(df[time_column], format="%Y%m%d")
        else:
            raise ValueError(
                f"{path.name}: datetime format must be yyyymmddhhmmss or yyyymmdd"
            )
    return df


def read(path, kwargs={}, assoc_kwargs={}):
    """
    Read one or more IPF files to a single pandas.DataFrame, including associated
    (TXT) files.

    The different IPF files can be from different model layers,
    and column names may differ between them.

    Note that this function always returns a ``pandas.DataFrame``. IPF files
    always contain spatial information, for which ``geopandas.GeoDataFrame``
    is a better fit, in principle. However, GeoDataFrames are not the best fit
    for the associated data.

    To perform spatial operations on the points, you're likely best served by
    (temporarily) creating a GeoDataFrame, doing the spatial operation, and
    then using the output to select values in the original DataFrame. Please
    refer to the examples.

    Parameters
    ----------
    path: str, Path or list
        This can be a single file, 'wells_l1.ipf', a glob pattern expansion,
        'wells_l*.ipf', or a list of files, ['wells_l1.ipf', 'wells_l2.ipf'].
        Note that each file needs to have the same columns, such that they can
        be combined in a single pd.DataFrame.
    kwargs : dict
        Dictionary containing the ``pandas.read_csv()`` keyword arguments for the
        IPF files (e.g. `{"delim_whitespace": True}`)
    assoc_kwargs: dict
        Dictionary containing the ``pandas.read_csv()`` keyword arguments for the
        associated (TXT) files (e.g. `{"delim_whitespace": True}`)

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    Read an IPF file into a dataframe:

    >>> import imod
    >>> df = imod.ipf.read("example.ipf")

    Convert the x and y data into a GeoDataFrame, do a spatial operation, and
    use it to select points within a polygon.
    Note: ``gpd.points_from_xy()`` requires a geopandas version >= 0.5.

    >>> import geopandas as gpd
    >>> polygon = gpd.read_file("polygon.shp").geometry[0]
    >>> ipf_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df["x"], df["y"]))
    >>> within_polygon = ipf_points.within(polygon)
    >>> selection = df[within_polygon]

    The same exercise is a little more complicated when associated files (like
    timeseries) are involved, since many duplicate values of x and y will exist.
    The easiest way to isolate these is by applying a groupby, and then taking
    first of x and y of every group:

    >>> df = imod.ipf.read("example_with_time.ipf")
    >>> first = df.groupby("id").first()  # replace "id" by what your ID column is called
    >>> x = first["x"]
    >>> y = first["y"]
    >>> id_code = first.index  # id is a reserved keyword in python
    >>> ipf_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y))
    >>> within_polygon = ipf_points.within(polygon)

    Using the result is a little more complicated as well, since it has to be
    mapped back to many duplicate values of the original dataframe.
    There are two options. First, by using the index:

    >>> within_polygon.index = id_code
    >>> df = df.set_index("id")
    >>> selection = df[within_polygon]

    If you do not wish to change index on the original dataframe, use
    ``pandas.DataFrame.merge()`` instead.

    >>> import pandas as pd
    >>> within_polygon = pd.DataFrame({"within": within_polygon})
    >>> within_polygon["id"] = id_code
    >>> df = df.merge(within_polygon, on="id")
    >>> df = df[df["within"]]
    """
    if isinstance(path, list):
        paths = path
    elif isinstance(path, (str, pathlib.Path)):
        # convert since for Path.glob non-relative patterns are unsupported
        path = str(path)
        paths = [pathlib.Path(p) for p in glob.glob(path)]
    else:
        raise ValueError("Path should be either a list, str or pathlib.Path")

    n = len(paths)
    if n == 0:
        raise FileNotFoundError(f"Could not find any files matching {path}")
    elif n == 1:
        bigdf = _read(paths[0], kwargs, assoc_kwargs)
    else:
        dfs = []
        for p in paths:
            layer = util.decompose(p).get("layer")
            try:
                df = _read(p, kwargs, assoc_kwargs)
            except Exception as e:
                raise type(e)(f'{e}\nWhile reading IPF file "{p}"') from e
            if layer is not None:
                df["layer"] = layer
            dfs.append(df)
        bigdf = pd.concat(
            dfs, ignore_index=True, sort=False
        )  # this sorts in pandas < 0.23

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


def write_assoc(path, df, itype=1, nodata=1.0e20, assoc_columns=None):
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
    assoc_columns : optional, list or dict
        Columns to store in the associated file. In case of a dictionary, the
        columns will be renamed according to the mapping in the dictionary.
        Defaults to None.

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
        if colname not in colnames:
            raise ValueError(f'given itype requires column "{colname}"')
        colnames.remove(colname)
        columnorder.append(colname)
    columnorder += colnames

    # Check if columns have to be renamed
    if isinstance(assoc_columns, dict):
        columnorder = [assoc_columns[col] for col in columnorder]
        df = df.rename(columns=assoc_columns)

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

    # We cannot rely on the quoting=QUOTE_NONNUMERIC policy
    # The reason is that datetime columns are converted to string as well
    # and then quoted. This causes trouble with some iMOD(batch) functions.
    for column in df.columns:
        if df.loc[:, column].dtype == np.dtype("O"):
            df.loc[:, column] = df.loc[:, column].astype(str)
            df.loc[:, column] = '"' + df.loc[:, column] + '"'

    df.to_csv(
        path,
        index=False,
        header=False,
        mode="a",
        date_format="%Y%m%d%H%M%S",
        quoting=csv.QUOTE_NONE,
    )


def write(path, df, indexcolumn=0, assoc_ext="txt", nodata=1.0e20):
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
    df = df.fillna(nodata)
    nrecords, nfields = df.shape
    with open(path, "w") as f:
        f.write(f"{nrecords}\n{nfields}\n")
        for colname in df.columns:
            if "," in colname or " " in colname:
                colname = '"' + colname + '"'
            f.write(f"{colname}\n")
        f.write(f"{indexcolumn},{assoc_ext}\n")

    # We cannot rely on the quoting=QUOTE_NONNUMERIC policy
    # The reason is that datetime columns are converted to string as well
    # and then quoted. This causes trouble with some iMOD(batch) functions.
    for column in df.columns:
        if df.loc[:, column].dtype == np.dtype("O"):
            df.loc[:, column] = df.loc[:, column].astype(str)
            df.loc[:, column] = '"' + df.loc[:, column] + '"'

    # workaround pandas issue by closing the file first, see
    # https://github.com/pandas-dev/pandas/issues/19827#issuecomment-398649163
    df.to_csv(path, index=False, header=False, mode="a", quoting=csv.QUOTE_NONE)


def _is_single_value(group):
    return len(pd.unique(group)) == 1


def _compose_ipf(path, df, itype, assoc_ext, nodata=1.0e20, assoc_columns=None):
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
        If ``None`` no associated files are written.
        Other possible values, either integer or string:

        * ``1`` or ``"timeseries"``
        * ``2`` or ``"borehole1d"``
        * ``3`` or ``"cpt"``
        * ``4`` or ``"borehole3d"``
    assoc_ext : str
        Extension of the associated files. Normally ".txt".
    nodata : float
        The value given to nodata values. These are generally NaN (Not-a-Number)
        in pandas, but this leads to errors in iMOD(FLOW) for IDFs.
        Defaults to value of 1.0e20 instead.
    assoc_columns : optional, list or dict
        Columns to store in the associated file. In case of a dictionary, the
        columns will be renamed according to the mapping in the dictionary.
        Defaults to None.

    Returns
    -------
    None
        Writes files.
    """
    if itype is None:
        write(path, df, nodata=nodata)
    else:
        itype = _coerce_itype(itype)
        colnames = _lower(list(df))
        df.columns = colnames
        for refname in ["x", "y", "id"]:
            if refname not in colnames:
                raise ValueError(f'given itype requires column "{refname}"')
            colnames.remove(refname)

        grouped = df.groupby("id")
        if not grouped["x"].apply(_is_single_value).all():
            raise ValueError("column x contains more than one value per id")
        if not grouped["y"].apply(_is_single_value).all():
            raise ValueError("column y contains more than one value per id")
        # get columns that have only one value within a group, to save them in ipf
        ipf_columns = [
            (colname, "first")
            for colname in colnames
            if grouped[colname].apply(_is_single_value).all()
        ]

        for idcode, group in grouped:
            assoc_path = path.parent.joinpath(str(idcode) + "." + str(assoc_ext))
            assoc_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(assoc_columns, list):
                selection = assoc_columns
            elif isinstance(assoc_columns, dict):
                selection = list(assoc_columns.keys())
            else:
                selection = [
                    colname for colname in colnames if colname not in ipf_columns
                ]
            out_df = group[selection]
            write_assoc(assoc_path, out_df, itype, nodata, assoc_columns)

        # ensures right order for x, y, id; so that also indexcolumn == 3
        agg_kwargs = collections.OrderedDict(
            [("x", "first"), ("y", "first"), ("id", "first")]
        )
        agg_kwargs.update(ipf_columns)
        agg_df = grouped.agg(agg_kwargs)
        write(path, agg_df, 3, assoc_ext, nodata=nodata)


def save(path, df, itype=None, assoc_ext="txt", nodata=1.0e20, assoc_columns=None):
    """
    Saves the contents of a pandas DataFrame to one or more IPF files, and
    associated (TXT) files.

    Can write multiple IPF files if one of the columns is named "layer". In
    turn, multiple associated (TXT) files may written for each of these IPF
    files. Note that the ID must be unique for each layer. See the examples.

    Parameters
    ----------
    path : pathlib.Path or str
        path of the written IPF file.
        Any associated files are written relative to this path, based on the ID
        column.
    df : pandas.DataFrame
        DataFrame containing the data to write.
    itype : int or str or None
        IPF type. Defaults to ``None``, in which case no associated files are
        created. Possible other values, either integer or string:

        * ``1`` or ``"timeseries"``
        * ``2`` or ``"borehole1d"``
        * ``3`` or ``"cpt"``
        * ``4`` or ``"borehole3d"``
    assoc_ext : str
        Extension of the associated files. Defaults to "txt".
    nodata : float
        The value given to nodata values. These are generally NaN (Not-a-Number)
        in pandas, but this leads to errors in iMOD(FLOW) for IDFs.
        Defaults to value of 1.0e20 instead.
    assoc_columns : optional, list or dict
        Columns to store in the associated file. In case of a dictionary, the
        columns will be renamed according to the mapping in the dictionary.
        Defaults to None.

    Returns
    -------
    None
        Writes files.

    Examples
    --------
    To write a single IPF without associated timeseries or boreholes:

    >>> imod.ipf.save("static-data.ipf", df)

    To write timeseries data:

    >>> imod.ipf.save("transient-data.ipf", df, itype="timeseries")

    If a ``"layer"`` column is present, make sure the ID is unique per layer:

    >>> df["id"] = df["id"].str.cat(df["layer"], sep="_")
    >>> imod.ipf.save("layered.ipf", df, itype="timeseries")

    An error will be raised otherwise.
    """

    path = pathlib.Path(path)

    d = {"extension": ".ipf", "name": path.stem, "directory": path.parent}
    d["directory"].mkdir(exist_ok=True, parents=True)

    colnames = _lower(list(df))
    # Lower assoc_columns as well if available
    if isinstance(assoc_columns, list):
        assoc_columns = _lower(assoc_columns)
    elif isinstance(assoc_columns, dict):
        keys = _lower(assoc_columns.keys())
        values = _lower(assoc_columns.values())
        assoc_columns = {k: v for k, v in zip(keys, values)}

    df.columns = colnames
    if "layer" in colnames:
        if "time" in colnames:
            groupcols = ["time", "id"]
        else:
            groupcols = "id"

        n_layer_per_id = df.groupby(groupcols)["layer"].nunique()
        if (n_layer_per_id > 1).any():
            raise ValueError(
                "Multiple layer values for a single ID detected. "
                "Unique IDs are required for each layer."
            )

        for layer, group in df.groupby("layer"):
            d["layer"] = layer
            fn = util.compose(d)
            _compose_ipf(fn, group, itype, assoc_ext, nodata, assoc_columns)
    else:
        fn = util.compose(d)
        _compose_ipf(fn, df, itype, assoc_ext, nodata, assoc_columns)
