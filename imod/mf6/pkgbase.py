import pathlib

import jinja2
import numba
import numpy as np
import xarray as xr


class Package(xr.Dataset):
    _template = None

    def _initialize_template(self):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader)
        self._template = env.get_template(f"gwf-{self._pkg_id}.j2")

    def render(self, *args, **kwargs):
        d = {k: v.values for k, v in self.data_vars.items()}
        if hasattr(self, "_keywords"):
            for key in self._keywords.keys():
                self._replace_fomat(d, key)
        return self._template.format(**d)

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
        sparse_data.tofile(outpath)

    def render(self):
        d = {}
        for k, v in self.data_vars.items():
            d[k] = v
        self._template.render(**d)

    def _compose_values(self, varname, directory, *args, **kwargs):
        """
        Compose values of dictionary.

        Ignores times. Time dependent boundary conditions use the method from
        BoundaryCondition.

        See documentation of wq
        """
        da = self[varname]

        layered = False
        values = []
        if "x" in da.coords and "y" in da.coords:
            values.append(f"open/close {directory}/{self._pkg_id}_{s}.bin (binary)")
        else:
            if "layer" in da.coords:
                layered = True
                for layer in da.coords["layer"]:
                    values.append(f"constant {da.sel(layer=layer).values[()]}")
            else:
                values.append(f"constant {da.values[()]}")

        return layered, values


class BoundaryCondition(Package):
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
