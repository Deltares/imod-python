import numpy as np

import imod
from imod.wq.pkgbase import Package


FLOAT_FORMAT = "%.18G"


def _write_arr(path, da, nodata=1.0e20, *_):
    """
    Write an ".ARR" file: a plaintext array file which can be understood by MODFLOW.
    """
    dx, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(da)
    ncol, nrow = da.shape
    footer = f" DIMENSIONS\n{ncol}\n{nrow}\n{xmin}\n{ymin}\n{xmax}\n{ymax}\n{nodata}\n0\n{dx}\n{dx}"
    a = np.nan_to_num(da.values, nan=nodata)
    np.savetxt(
        path, a, fmt=FLOAT_FORMAT, delimiter=" ", footer=footer, comments=""
    )
    return


class HorizontalAnisotropy(Package):
    """
    Horizontal Anisotropy package.
    Anisotropy is a phenomenon for which the permeability k is not equal along the x- and y Cartesian axis.

    Parameters
    ----------
    factor : float or xr.DataArray of floats
        The anisotropic factor perpendicular to the main principal axis which
        is the axis of highest permeability. Factor between 0.0 and 1.0 (isotropic).
    angle : float or xr.DataArray of floats
        The angle along the main principal axis (highest permeability) measured
        in degrees from north (0), east (90), south (180) and west (270).
    """

    _pkg_id = "ani"

    _template = "[ani]\n" "    anifile = {anifile}\n"

    def __init__(
        self,
        factor,
        angle,
    ):
        super().__init__()
        self["factor"] = factor
        self["angle"] = angle

    def _render(self, directory, *args, **kwargs):
        # write ani file
        # in render function, as here we know where the model is run from
        # and how many layers: store this info for later use in save
        d = {"anifile": (directory / "horizontal_anistropy.ani").as_posix()}
        return self._template.format(**d)
    
    def _render_anifile(self, directory):
        """
        Unfortunately, the iMOD-WQ anisotropy works through an .ANI file, which
        then refers to .ARR files rather than a single indirection.
        
        So the runfile section points to the .ANI file, which in turn points to
        the .ARR files, or contains constants.
        """
        content = []
        for varname, nodata in zip(("factor", "angle"), (1.0, 0.0)):
            variable = self.dataset[varname]
            if "x" in da.coords and "y" in da.coords:
                for layer, da in variable.groupby("layer"):
                    path = (directory / f"{varname}_l{layer}.arr").as_posix()
                    content.append(
                        f"open/close {path} 1.0D0 (FREE) -1 {variable}_l{layer}"
                    )
            else:
                for layer, da in variable.groupby("layer"):
                    value = da.item()
                    if np.isnan(value):
                        value = nodata
                    content.append(
                        f"constant {value:FLOAT_FORMAT} {varname}_l{layer}"
                    )
        return "\n".join(content)
    
    def save(self, directory):
        ani_content = self._render_anifile(directory)
        path = (directory / "horizontal_anistropy.ani").as_posix()
        with open(path, "w") as f:
            f.write(ani_content)

        # Write .ARR files
        for varname, nodata in zip(("factor", "angle"), (1.0, 0.0)):
            da = self.dataset[varname]
            if "x" in da.coords and "y" in da.coords:
                path = directory / f"{varname}.arr"
                imod.array_io.writing._save(
                    path,
                    da,
                    nodata=nodata,
                    pattern=None,
                    dtype=np.float32,
                    write=_write_arr,
                )
        return
