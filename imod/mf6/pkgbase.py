import pathlib

import jinja2
import numba
import numpy as np
import xarray as xr


class Package(xr.Dataset):
    def _initialize_template(self):
        loader = jinja2.PackageLoader("imod", "mf6", "templates")
        env = jinja2.Environment(loader=loader)
        self._template = env.get_template(f"{self._pkg_id}.j2")

    def render(self, *args, **kwargs):
        d = {k: v.values for k, v in self.data_vars.items()}
        if hasattr(self, "_keywords"):
            for key in self._keywords.keys():
                self._replace_fomat(d, key)
        return self._template.format(**d)

    def write_blockfile(self, pgkname, directory):
        content = self.render(directory, pkgname)
        filename = directory.join(f"{pkgname}.{self._pkg_id}")
        with open(filename) as f:
            f.write(content)

    def to_sparse(self, data):
        """Convert from dense arrays to list based input"""
        notnull = ~np.isnan(data)
        indices = np.argwhere(notnull)
        values = data[notnull]
        out = np.concatenate(
            [indices.astype(np.int32), values.reshape(values.size, 1).view(np.int32)],
            axis=-1,
        ).flatten()
        return out

    def write_binaryfile(self, outpath, data):
        sparse_data = self.to_sparse(data)
        sparse_data.to_file(outpath)

    def render(self):
        d = {}
        for k, v in self.data_vars.items():
            d[k] = v
        self._template.render(**d)

    def _compose_values(self, varname, directory, *args, **kwargs):
        """
        Compose values of dictionary

        See documentation of wq
        """
        da = self[varname]
        
        layered = False
        values = []
        if "x" in da.coords and "y" in da.coords:
            values.append(f"open/close {directory}/{self.pkg_id}_{s}.bin (binary)")          
        else:
            if "layer" in da.coords:
                layered = True
                for layer in da.coords["layer"]:
                    values.append(f"constant {da.sel(layer=layer).values[()]}")
            else:
                values.append(f"constant {da.values[()]}")
        
        return layered, values

    def write(self, pkgname, directory):
        outdir = directory.join(pgkname)
        outdir.mkdir(parents=True, exist_ok=True)

        # Or add incrementally?
        self.write_blockfile(pkgname, directory)

        for periodnumber, data in self:
            outpath = outdir.join(f"{self._pkg_id}_{periodnumber}.bin")
            self.write_binaryfile(outpath, data)


class BoundaryCondition(Package):
    def _max_active_n(self, varname, nlayer):
        """
        Determine the maximum active number of cells that are active
        during a stress period.

        Parameters
        ----------
        varname : str
            name of the variable to use to calculate the maximum number of
            active cells. Generally conductance.
        nlayer : int
            number of layers, taken from ibound.
        """
        if "time" in self[varname].coords:
            nmax = int(self[varname].groupby("time").count(xr.ALL_DIMS).max())
        else:
            nmax = int(self[varname].count())
        if not "layer" in self.coords:  # Then it applies to every layer
            nmax *= nlayer
        return nmax

    def _compose_values(self, varname, globaltimes, directory):
        da = self[varname]
        values = {}

        if "x" not in da.coords or "y" not in da.coords:
            raise ValueError("Boundary conditions only accept")
            
        if "time" in da.coords:
            package_times = da.coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for s in starts:
                values[s] = f"open/close {directory}/{self.pkg_id}_{s}.bin (binary)"
        else:
            values[1] = f"open/close {directory}/{self.pkg_id}_1.bin (binary)"
        return False, values

    def render(self, directory, globaltimes):
        mapping = tuple([(k, v) for k, v in self._mapping if v in self.data_vars])
        d = {"mapping": mapping}
        dicts = {}

        for varname in self.data_vars.keys():
            dicts[varname] = self._compose_values(varname, globaltimes, directory)

        d["dicts"] = dicts

        return self._template.render(d)
