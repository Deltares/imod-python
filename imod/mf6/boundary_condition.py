import abc
import pathlib
from copy import copy
from typing import Dict, List

import numpy as np
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.package import Package


def _dis_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("row", np.int32), ("column", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 2:
        recarr["row"], recarr["column"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, irow, icolumn = np.argwhere(notnull).transpose()
        recarr["row"] = irow + 1
        recarr["column"] = icolumn + 1
        recarr["layer"] = layer[ilayer]
    return recarr


def _disv_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("cell2d", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 1 and layer.size == 1:
        recarr["cell2d"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, icell2d = np.argwhere(notnull).transpose()
        recarr["cell2d"] = icell2d + 1
        recarr["layer"] = layer[ilayer]
    return recarr


class BoundaryCondition(Package, abc.ABC):
    """
    BoundaryCondition is used to share methods for specific stress packages
    with a time component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    This class only supports `list input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Package`.
    """

    def set_repeat_stress(self, times) -> None:
        """
        Set repeat stresses: re-use data of earlier periods.

        Parameters
        ----------
        times: Dict of datetime-like to datetime-like.
            The data of the value datetime is used for the key datetime.
        """
        keys = [
            imod.wq.timeutil.to_datetime(key, use_cftime=False) for key in times.keys()
        ]
        values = [
            imod.wq.timeutil.to_datetime(value, use_cftime=False)
            for value in times.values()
        ]
        self.dataset["repeat_stress"] = xr.DataArray(
            data=np.column_stack((keys, values)),
            dims=("repeat", "repeat_items"),
        )

    def _max_active_n(self):
        """
        Determine the maximum active number of cells that are active
        during a stress period.
        """
        da = self.dataset[self.period_data()[0]]
        if "time" in da.coords:
            nmax = int(da.groupby("time").count(xr.ALL_DIMS).max())
        else:
            nmax = int(da.count())
        return nmax

    def _write_binaryfile(self, outpath, sparse_data):
        with open(outpath, "w") as f:
            sparse_data.tofile(f)

    def _write_textfile(self, outpath, sparse_data):
        fields = sparse_data.dtype.fields
        fmt = [self._number_format(field[0]) for field in fields.values()]
        header = " ".join(list(fields.keys()))
        with open(outpath, "w") as f:
            np.savetxt(fname=f, X=sparse_data, fmt=fmt, header=header)

    def write_datafile(self, outpath, ds, binary):
        """
        Writes a modflow6 binary data file
        """
        layer = ds["layer"].values if "layer" in ds.coords else None
        arrdict = self._ds_to_arrdict(ds)
        sparse_data = self._to_sparse(arrdict, layer)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        if binary:
            self._write_binaryfile(outpath, sparse_data)
        else:
            self._write_textfile(outpath, sparse_data)

    def _ds_to_arrdict(self, ds):
        arrdict = {}
        for datavar in ds.data_vars:
            if ds[datavar].shape == ():
                raise ValueError(
                    f"{datavar} in {self._pkg_id} package cannot be a scalar"
                )
            auxiliary_vars = (
                self.get_auxiliary_variable_names()
            )  # returns something like {"concentration": "species"}
            if datavar in auxiliary_vars.keys():  # if datavar is concentration
                if (
                    auxiliary_vars[datavar] in ds[datavar].dims
                ):  # if this concentration array has the species dimension
                    for s in ds[datavar].values:  # loop over species
                        arrdict[s] = (
                            ds[datavar]
                            .sel({auxiliary_vars[datavar]: s})
                            .values  # store species array under its species name
                        )
            else:
                arrdict[datavar] = ds[datavar].values
        return arrdict

    def _to_sparse(self, arrdict, layer):
        """Convert from dense arrays to list based input"""
        # TODO stream the data per stress period
        # TODO add pkgcheck that period table aligns
        # Get the number of valid values
        if layer is None:
            raise ValueError("Layer should be provided")

        data = next(iter(arrdict.values()))
        notnull = ~np.isnan(data)

        if isinstance(self.dataset, xr.Dataset):
            recarr = _dis_recarr(arrdict, layer, notnull)
        elif isinstance(self.dataset, xu.UgridDataset):
            recarr = _disv_recarr(arrdict, layer, notnull)
        else:
            raise TypeError(
                "self.dataset should be xarray.Dataset or xugrid.UgridDataset,"
                f" is {type(self.dataset)} instead"
            )
        # Fill in the data
        for key, arr in arrdict.items():
            values = arr[notnull].astype(np.float64)
            recarr[key] = values

        return recarr

    def period_paths(self, directory, pkgname, globaltimes, bin_ds, binary):
        pkg_directory = pathlib.Path(directory.stem) / pkgname

        if binary:
            ext = "bin"
        else:
            ext = "dat"

        periods = {}
        if "time" in bin_ds:  # one of bin_ds has time
            package_times = bin_ds.coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for i, start in enumerate(starts):
                path = pkg_directory / f"{self._pkg_id}-{i}.{ext}"
                periods[start] = path.as_posix()

            repeat_stress = self.dataset.get("repeat_stress")
            if repeat_stress is not None and repeat_stress.values[()] is not None:
                keys = repeat_stress.isel(repeat_items=0).values
                values = repeat_stress.isel(repeat_items=1).values
                repeat_starts = np.searchsorted(globaltimes, keys) + 1
                values_index = np.searchsorted(globaltimes, values) + 1
                for i, start in zip(values_index, repeat_starts):
                    periods[start] = periods[i]
                # Now make sure the periods are sorted by key.
                periods = dict(sorted(periods.items()))
        else:
            path = pkg_directory / f"{self._pkg_id}.{ext}"
            periods[1] = path.as_posix()

        return periods

    def get_options(self, predefined_options: Dict, not_options: List = None):
        options = copy(predefined_options)

        if not_options is None:
            not_options = self.period_data()

        for varname in self.dataset.data_vars.keys():  # pylint:disable=no-member
            if varname in not_options:
                continue
            v = self.dataset[varname].values[()]
            if self._valid(v):  # skip None and False
                options[varname] = v
        return options

    def _get_bin_ds(self):
        """
        Get binary dataset data for stress periods, this data will be written to
        datafiles. This method can be overrided to do some extra operations on
        this dataset before writing.
        """
        return self[self.period_data()]

    def render(self, directory, pkgname, globaltimes, binary):
        """Render fills in the template only, doesn't write binary data"""
        d = {"binary": binary}
        bin_ds = self._get_bin_ds()
        d["periods"] = self.period_paths(
            directory, pkgname, globaltimes, bin_ds, binary
        )
        # construct the rest (dict for render)
        d = self.get_options(d)
        d["maxbound"] = self._max_active_n()

        # now we should add the auxiliary variable names to d
        auxiliaries = (
            self.get_auxiliary_variable_names()
        )  # returns someting like {"concentration": "species"}

        # loop over the types of auxiliary variables (for example concentration)
        for auxvar in auxiliaries.keys():
            # if "concentration" is a variable of this dataset
            if auxvar in self.dataset.data_vars:
                # if our concentration dataset has the species coordinate
                if auxiliaries[auxvar] in self.dataset[auxvar].coords:
                    # assign the species names list to d
                    d["auxiliary"] = self.dataset[auxiliaries[auxvar]].values
                else:
                    # the error message is more specific than the code at this point.
                    raise ValueError(
                        f"{auxvar} requires a {auxiliaries[auxvar]} coordinate."
                    )

        return self._template.render(d)

    def write_perioddata(self, directory, pkgname, binary):
        if len(self.period_data()) == 0:
            return
        bin_ds = self._get_bin_ds()

        if binary:
            ext = "bin"
        else:
            ext = "dat"

        if "time" in bin_ds:  # one of bin_ds has time
            for i in range(len(self.dataset.time)):
                path = directory / pkgname / f"{self._pkg_id}-{i}.{ext}"
                self.write_datafile(
                    path, bin_ds.isel(time=i), binary=binary
                )  # one timestep
        else:
            path = directory / pkgname / f"{self._pkg_id}.{ext}"
            self.write_datafile(path, bin_ds, binary=binary)

    def write(self, directory, pkgname, globaltimes, binary):
        """
        writes the blockfile and binary data

        directory is modelname
        """
        directory = pathlib.Path(directory)
        self.write_blockfile(
            directory=directory,
            pkgname=pkgname,
            globaltimes=globaltimes,
            binary=binary,
        )
        self.write_perioddata(
            directory=directory,
            pkgname=pkgname,
            binary=binary,
        )

    def assign_dims(self, arg) -> Dict:
        is_da = isinstance(arg, xr.DataArray)
        if is_da and "time" in arg.coords:
            if arg.ndim != 2:
                raise ValueError("time varying variable: must be 2d")
            if arg.dims[0] != "time":
                arg = arg.transpose()
            da = xr.DataArray(
                data=arg.values, coords={"time": arg["time"]}, dims=["time", "index"]
            )
            return da
        elif is_da:
            return ("index", arg.values)
        else:
            return ("index", arg)


class AdvancedBoundaryCondition(BoundaryCondition, abc.ABC):
    """
    Class dedicated to advanced boundary conditions, since MF6 does not support
    binary files for Advanced Boundary conditions.

    The advanced boundary condition packages are: "uzf", "lak", "maw", "sfr".

    """

    def _get_field_spec_from_dtype(self, recarr):
        """
        From https://stackoverflow.com/questions/21777125/how-to-output-dtype-to-a-list-or-dict
        """
        return [
            (x, y[0])
            for x, y in sorted(recarr.dtype.fields.items(), key=lambda k: k[1])
        ]

    def _write_file(self, outpath, sparse_data):
        """
        Write to textfile, which is necessary for Advanced Stress Packages
        """
        fields = sparse_data.dtype.fields
        fmt = [self._number_format(field[0]) for field in fields.values()]
        header = " ".join(list(fields.keys()))
        np.savetxt(fname=outpath, X=sparse_data, fmt=fmt, header=header)

    @abc.abstractmethod
    def _package_data_to_sparse(self):
        """
        Get packagedata, override with function for the advanced boundary
        condition in particular
        """
        return

    def write_packagedata(self, directory, pkgname, binary):
        outpath = directory / pkgname / f"{self._pkg_id}-pkgdata.dat"
        outpath.parent.mkdir(exist_ok=True, parents=True)
        package_data = self._package_data_to_sparse()
        self._write_file(outpath, package_data)

    def write(self, directory, pkgname, globaltimes, binary):
        self.fill_stress_perioddata()
        self.write_blockfile(directory, pkgname, globaltimes, binary=False)
        self.write_perioddata(directory, pkgname, binary=False)
        self.write_packagedata(directory, pkgname, binary=False)


class DisStructuredBoundaryCondition(BoundaryCondition):
    def _to_sparse(self, arrdict, layer):
        spec = []
        for key in arrdict:
            if key in ["layer", "row", "column"]:
                spec.append((key, np.int32))
            else:
                spec.append((key, np.float64))

        sparse_dtype = np.dtype(spec)
        nrow = next(iter(arrdict.values())).size
        recarr = np.empty(nrow, dtype=sparse_dtype)
        for key, arr in arrdict.items():
            recarr[key] = arr
        return recarr


class DisVerticesBoundaryCondition(BoundaryCondition):
    def _to_sparse(self, arrdict, layer):
        spec = []
        for key in arrdict:
            if key in ["layer", "cell2d"]:
                spec.append((key, np.int32))
            else:
                spec.append((key, np.float64))

        sparse_dtype = np.dtype(spec)
        nrow = next(iter(arrdict.values())).size
        recarr = np.empty(nrow, dtype=sparse_dtype)
        for key, arr in arrdict.items():
            recarr[key] = arr
        return recarr
