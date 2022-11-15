import numpy as np
import pathlib
import re
import shutil

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

    _template = (
        "[ani]\n"
        "    anifile = {anifile}\n"
    )

    def __init__(
        self,
        anifile,
    ):
        super().__init__()
        self["anifile"] = anifile
    
    def _render(self, directory, *args, **kwargs):
        path_ani = pathlib.Path(str(self['anifile'].values))
        d = {"anifile": f"ani/{path_ani.name}"}

        return self._template.format(**d)
            
    def save(self, directory):
        """Overload save function. 
        Saves anifile to location, along with referenced .arr files"""
        directory.mkdir(exist_ok=True)
        
        path_ani = pathlib.Path(str(self['anifile'].values))
        
        # regex adapted from stackoverflow: https://stackoverflow.com/questions/54990405/a-general-regex-to-extract-file-paths-not-urls
        rgx = r'((?:[a-zA-Z]:|(?<![:/\\])[\\\/](?!CLOSE)(?!close )|\~[\\\/]|(?:\.{1,2}[\\\/])+)[\w+\\\s_\-\(\)\/]*(?:\.\w+)*)'
        with open(path_ani, "r") as f, open(directory / path_ani.name, "w") as f2:
            for line in f:
                p = re.search(rgx, line)
                if p:
                    # path to file detected, 
                    # replace to relative and
                    # copy to directory
                    path = pathlib.Path(p[0])
                    f2.write(line.replace(p[0], f"{directory.stem}/{path.name}"))
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
    fn_ani : filename for created anifile, optional
        Default: ani.ani
    """

    _pkg_id = "ani"

    _template = (
        "[ani]\n"
        "    anifile = {anifile}\n"
    )

    def __init__(
        self,
        factor,
        angle,
        fn_ani="ani.ani"
    ):
        super().__init__()
        self["factor"] = factor
        self["factor"] = self["factor"].fillna(1.)
        self["angle"] = angle
        self["angle"] = self["angle"].fillna(0.)
        self["fn_ani"] = fn_ani
    
    def _render(self, directory, *args, **kwargs):
        d = {"anifile": f"ani/{str(self['fn_ani'].values)}"}

        return self._template.format(**d)
            
    def save(self, directory):
        """Overload save function. 
        Saves anifile to location, along with referenced .arr files"""
        def _write(path, a, nodata=1.0e20, dtype=np.float32):
            if not np.all(a == a[0][0]):
                return np.savetxt(path, a, fmt="%.5f", delimiter=" ")
            else:
                # write single value to ani file
                pass

        for name, da in self.dataset.data_vars.items():  # pylint: disable=no-member
            if "y" in da.coords and "x" in da.coords:
                path = pathlib.Path(directory).joinpath(f"{name}.arr")
                if name == "factor":
                    nodata = 1.
                elif name == "angle":
                    nodata = 0.
                else:
                    nodata = 1.0e20
                imod.array_io.writing._save(path, da, nodata=nodata, pattern=None, dtype=np.float32, write=_write)

        # write ani file
        with open(directory / f"{str(self['fn_ani'].values)}", "w") as f:
            for l in self.dataset.layer.values:
                for prm in ["factor","angle"]:
                    a = self.dataset[prm].sel(layer=l).values
                    if not np.all(a == a[0][0]):
                        f.write(f"open/close {directory.as_posix()}/{prm}_l{float(l):.0f}.arr 1.0D0 (FREE) -1 {prm}_l{float(l):.0f}\n")
                    else:
                        f.write(f"constant {float(a[0][0]):.5f} {prm}_l{float(l):.0f}\n")

    def _pkgcheck(self, ibound=None):
        pass
