import pathlib

import jinja2
import numba
import numpy as np
import xarray as xr


class Package(xr.Dataset):
    """
    Package is used to share methods for specific packages with no time component.

    It is not meant to be used directly, only to inherit from, to implement new packages.
    
    This class only supports `array input <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=16>`_,
    not the list input which is used in :class:`BoundaryCondition`.
    """

    _template = None

    def _initialize_template(self):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader)
        if self._pkg_id == "model":
            fname = "gwf-nam.j2"
        elif self._pkg_id == "ims":
            fname = "sln-ims.j2"
        elif self._pkg_id == "simulation":
            fname = "sim-nam.j2"
        elif self._pkg_id == "tdis":
            fname = "sim-tdis.j2"
        else:
            fname = f"gwf-{self._pkg_id}.j2"
        self._template = env.get_template(fname)

    def write_blockfile(self, directory, pkgname, globaltimes=None):
        content = self.render(directory, pkgname, globaltimes)
        filename = directory / f"{pkgname}.{self._pkg_id}"
        with open(filename, "w") as f:
            f.write(content)

    def to_sparse(self, arrlist):
        """Convert from dense arrays to list based input"""
        # TODO add pkgcheck that period table aligns
        data = arrlist[0]
        notnull = ~np.isnan(data)
        indices = np.argwhere(notnull)

        nrow = len(indices)
        # 3 columns for i j k
        # times 2 for int32 view of float64 values
        # (such that we can write as a single block)
        ncol = 3 + len(arrlist) * 2

        listarr = np.empty((nrow, ncol), dtype=np.int32)
        listarr[:, 0:3] = indices

        for i, arr in enumerate(arrlist):
            values = arr[notnull].astype(np.float64)
            c = 3 + i * 2
            listarr[:, c : c + 2] = values.reshape(values.size, 1).view(np.int32)

        # flatten to 1D such that numpy tofile doesn't write extra array dims
        return listarr.flatten()

    def write_binaryfile(self, outpath, ds):
        """
        data is a xr.Dataset with only the binary variables"""
        arrays = []
        for datavar in ds.data_vars:
            arrays.append(ds[datavar].values)
        sparse_data = self.to_sparse(arrays)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        with open(outpath, "w") as f:
            sparse_data.tofile(f)

    def write_binary_griddata(self, outpath, da, dtype):
        # From the modflow6 source, the header is defined as:
        # integer(I4B) :: kstp --> np.int8
        # integer(I4B) :: pertim --> np.int8
        # character(len=16) :: text --> 16 * np.int8
        # integer(I4B) :: m1, m2, m3 --> 3 * np.int8
        # so writing 21 bytes suffices to create a header.
        haslayer = "layer" in da.dims
        if haslayer:
            _, nrow, ncol = da.shape
        else:
            nrow, ncol = da.shape

        header = np.zeros(21, np.int8)
        header[18] = np.int8(ncol)
        header[19] = np.int8(nrow)
        header[20] = 1
        with open(outpath, "w") as f:
            if haslayer:
                for layer in da.dims["layer"]:
                    a = da.sel(layer=layer)
                    header.to_file(f)
                    a.values.flatten().astype(dtype).to_file(f)
            else:
                header.to_file(f)
                da.values.flatten().astype(dtype).to_file(f)

    def render(self, *args, **kwargs):
        d = {}
        for k, v in self.data_vars.items():
            value = v.values[()]
            if value:  # skip None and False
                d[k] = value
        return self._template.render(d)

    def _compose_values(self, da, directory, *args, **kwargs):
        """
        Compose values of dictionary.

        Ignores times. Time dependent boundary conditions use the method from
        BoundaryCondition.

        See documentation of wq
        """
        layered = False
        values = []
        if "x" in da.dims and "y" in da.dims:
            values.append(f"open/close {directory}/{self._pkg_id}.bin (binary)")
        else:
            if "layer" in da.dims:
                layered = True
                for layer in da.coords["layer"]:
                    values.append(f"constant {da.sel(layer=layer).values[()]}")
            else:
                value = da.values[()]
                if value:  # skip None or False
                    values.append(f"constant {value}")

        return layered, values


class BoundaryCondition(Package):
    """
    BoundaryCondition is used to share methods for specific stress packages with a time component.

    It is not meant to be used directly, only to inherit from, to implement new packages.

    This class only supports `list input <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Package`.
    """

    def _max_active_n(self):
        """
        Determine the maximum active number of cells that are active
        during a stress period.
        """
        da = self[self._binary_data[0]]
        if "time" in da.coords:
            nmax = int(da.groupby("time").count(xr.ALL_DIMS).max())
        else:
            nmax = int(da.count())
        return nmax

    def render(self, directory, pkgname, globaltimes):
        """Render fills in the template only, doesn't write binary data"""
        d = {}

        # period = {1: f"{directory}/{self._pkg_id}-{i}.bin"}

        bin_ds = self[[*self._binary_data]]
        periods = {}
        if "time" in bin_ds:  # one of bin_ds has time
            package_times = bin_ds.coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for i, s in enumerate(starts):
                path = directory / pkgname / f"{self._pkg_id}-{i}.bin"
                periods[s] = path.as_posix()
        else:
            path = directory / pkgname / f"{self._pkg_id}.bin"
            periods[1] = path.as_posix()

        d["periods"] = periods

        # construct the rest (dict for render)
        for varname in self.data_vars.keys():
            if varname in self._binary_data:
                continue
            d[varname] = self[varname].values[()]

        d["maxbound"] = self._max_active_n()

        return self._template.render(d)

    def write(self, directory, pkgname, globaltimes):
        """
        writes the blockfile and binary data
        
        directory is modelname
        """

        if isinstance(directory, str):
            directory = pathlib.Path(directory)

        self.write_blockfile(directory, pkgname, globaltimes)

        bin_ds = self[[*self._binary_data]]

        if "time" in bin_ds:  # one of bin_ds has time
            for i in range(len(self.time)):
                path = directory / pkgname / f"{self._pkg_id}-{i}.bin"
                self.write_binaryfile(path, bin_ds.isel(time=i))  # one timestep
        else:
            path = directory / pkgname / f"{self._pkg_id}.bin"
            self.write_binaryfile(path, bin_ds)
