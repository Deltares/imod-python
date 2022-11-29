import pathlib
import re
import shutil

import numpy as np

import imod
from imod.wq.pkgbase import Package


class HorizontalAnisotropyFile(Package):
    """
    Horizontal Anisotropy package.

    Parameters
    ----------
    anifile: str
        is the file location of the imod-wq ani-file. This file contains the
        anisotropy factor and angle of each layer, either as a constant or a
        reference to the file location of an '.arr' file. No checks are
        implemented for this file, user is responsible for consistency with
        model.
    """

    _pkg_id = "ani"

    _template = "[ani]\n" "    anifile = {anifile}\n\n"

    def __init__(
        self,
        anifile,
    ):
        super().__init__()
        self["anifile"] = anifile

    def _render(self, modelname, directory, nlayer):
        # write ani file
        # in render function, as here we know where the model is run from
        # and how many layers: store this info for later use in save
        self.anifile = f"{modelname}.ani"
        self.rendir = directory
        self.nlayer = nlayer

        d = {"anifile": f"{directory.as_posix()}/{modelname}.ani"}

        return self._template.format(**d)

    def save(self, directory):
        """Overload save function.
        Saves anifile to location, along with referenced .arr files
        assumes _render() to have run previously"""
        directory.mkdir(exist_ok=True)  # otherwise handled by idf.save

        path_ani = pathlib.Path(str(self["anifile"].values))

        # regex adapted from stackoverflow: https://stackoverflow.com/questions/54990405/a-general-regex-to-extract-file-paths-not-urls
        rgx = r"((?:[a-zA-Z]:|(?<![:/\\])[\\\/](?!CLOSE)(?!close )|\~[\\\/]|(?:\.{1,2}[\\\/])+)[\w+\\\s_\-\(\)\/]*(?:\.\w+)*)"
        with open(path_ani, "r") as f, open(directory / self.anifile, "w") as f2:
            for line in f:
                p = re.search(rgx, line)
                if p:
                    # path to file detected,
                    # replace to relative and
                    # copy to directory
                    path = pathlib.Path(p[0])
                    f2.write(
                        line.replace(p[0], f"{self.rendir.as_posix()}/{path.name}")
                    )
                    if not path.is_absolute():
                        path = path_ani.parent / path
                    shutil.copyfile(path, directory / path.name)
                else:
                    f2.write(line)

    def _pkgcheck(self, ibound=None):
        pass


class HorizontalAnisotropy(Package):
    """
    Horizontal Anisotropy package.
    Anisotropy is a phenomenon for which the permeability k is not equal along the x- and y Cartesian axis.

    Parameters
    ----------
    factor : float or xr.DataArray of floats
        The anisotropic factor perpendicular to the main principal axis (axis of highest permeability).
        Factor between 0.0 (full anisotropic) and 1.0 (full isotropic).
    angle : float or xr.DataArray of floats
        The angle along the main principal axis (highest permeability) measured in degrees from north (0),
        east (90), south (180) and west (270).
    """

    _pkg_id = "ani"

    _template = "[ani]\n" "    anifile = {anifile}\n\n"

    def __init__(
        self,
        factor,
        angle,
    ):
        super().__init__()
        self["factor"] = factor
        self["angle"] = angle

    def _render(self, modelname, directory, nlayer):
        # write ani file
        # in render function, as here we know where the model is run from
        # and how many layers: store this info for later use in save
        self.anifile = f"{modelname}.ani"
        self.rendir = directory
        self.nlayer = nlayer

        d = {"anifile": f"{directory.as_posix()}/{modelname}.ani"}

        return self._template.format(**d)

    def save(self, directory):
        """Overload save function.
        Saves anifile to location, along with created .arr files
        assumes _render() to have run previously"""
        directory.mkdir(exist_ok=True)  # otherwise handled by idf.save

        nodata_val = {"factor": 1.0, "angle": 0.0}

        def _check_all_equal(da):
            return np.all(np.isnan(da)) or np.all(
                da.values[~np.isnan(da)] == da.values[~np.isnan(da)][0]
            )

        def _write(path, da, nodata=1.0e20, dtype=np.float32):
            if not _check_all_equal(da):
                dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(da)
                ncol, nrow = da.shape
                footer = f" DIMENSIONS\n{ncol}\n{nrow}\n{xmin}\n{ymin}\n{xmax}\n{ymax}\n{nodata}\n0\n{dx}\n{dx}"
                a = np.nan_to_num(da.values, nan=nodata)
                return np.savetxt(
                    path, a, fmt="%.5f", delimiter=" ", footer=footer, comments=""
                )
            else:
                # write single value to ani file
                pass

        for name, da in self.dataset.data_vars.items():  # pylint: disable=no-member
            if "y" in da.coords and "x" in da.coords:
                path = pathlib.Path(directory).joinpath(f"{name}.arr")
                imod.array_io.writing._save(
                    path,
                    da,
                    nodata=nodata_val[name],
                    pattern=None,
                    dtype=np.float32,
                    write=_write,
                )

        # save anifile with data stored during _render
        with open(directory / self.anifile, "w") as f:
            for l in range(1, self.nlayer + 1):
                for prm in ["factor", "angle"]:
                    da = self.dataset[prm]
                    if "layer" in da.coords and "y" in da.coords and "x" in da.coords:
                        a = da.sel(layer=l)
                        if not _check_all_equal(a):
                            f.write(
                                f"open/close {self.rendir.as_posix()}/{prm}_l{float(l):.0f}.arr 1.0D0 (FREE) -1 {prm}_l{float(l):.0f}\n"
                            )
                        else:
                            if np.all(np.isnan(a)):
                                val = nodata_val[prm]
                            else:
                                val = a[~np.isnan()][0]
                            f.write(
                                f"constant {float(val):.5f} {prm}_l{float(l):.0f}\n"
                            )
                    else:
                        f.write(
                            f"constant {float(self.dataset[prm].values):.5f} {prm}_l{float(l):.0f}\n"
                        )

    def _pkgcheck(self, ibound=None):
        pass
