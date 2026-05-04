import collections
import inspect
import pathlib
from pathlib import Path
from typing import Any, Optional

import tomli
import tomli_w

import imod.mf6
import imod.msw
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ipackage import IPackage
from imod.common.serializer import EngineType
from imod.logging.logging_decorators import standard_log_decorator
from imod.mf6.model import Modflow6Model
from imod.mf6.validation_settings import ValidationSettings
from imod.msw.model import MetaSwapModel
from imod.schemata import ValidationError


@standard_log_decorator()
def dump_modelpkgs(
    model: IModel,
    directory,
    modelname,
    validate: bool = True,
    mdal_compliant: bool = False,
    crs: Optional[Any] = None,
    engine: EngineType = "netcdf4",
) -> Path:
    """
    Dump  model to files. Writes a model definition as .TOML file, which
    points to data for each package. Each package is stored as a separate
    NetCDF. Structured grids are saved as regular NetCDFs, unstructured
    grids are saved as UGRID NetCDF. Structured grids are always made GDAL
    compliant, unstructured grids can be made MDAL compliant optionally.

    Parameters
    ----------
    directory: str or Path
        directory to dump simulation into.
    modelname: str
        modelname, will be used to create a subdirectory.
    validate: bool, optional
        Whether to validate simulation data. Defaults to True.
    mdal_compliant: bool, optional
        Convert data with
        :func:`imod.prepare.spatial.mdal_compliant_ugrid2d` to MDAL
        compliant unstructured grids. Defaults to False.
    crs: Any, optional
        Anything accepted by rasterio.crs.CRS.from_user_input
        Requires ``rioxarray`` installed.
    engine : str, optional
        File engine used to write packages. Options are ``'netcdf4'``,
        ``'zarr'``, and ``'zarr.zip'``. NetCDF4 is readable by many other
        softwares, for example QGIS. Zarr is optimized for big data, cloud
        storage and parallel access. The ``'zarr.zip'`` option is an
        experimental option which creates a zipped zarr store in a single
        file, which is easier to copy and automatically compresses data as
        well. Default is ``'netcdf4'``.

    """
    modeldirectory = Path(directory) / modelname
    modeldirectory.mkdir(exist_ok=True, parents=True)

    # validation currently only supports MF6, but we want to keep the option to turn it on for other
    validation_context = ValidationSettings(validate=validate)
    if validation_context.validate:
        statusinfo = model.validate(modelname, validation_context)
        if statusinfo.has_errors():
            raise ValidationError(statusinfo.to_string())

    toml_content: dict = collections.defaultdict(dict)

    for pkgname, pkg in model.items():
        pkg_path = pkg.to_file(
            modeldirectory,
            pkgname,
            mdal_compliant=mdal_compliant,
            crs=crs,
            engine=engine,
        )
        toml_content[type(pkg).__name__][pkgname] = pkg_path.name

    toml_path = modeldirectory / f"{modelname}.toml"
    with open(toml_path, "wb") as f:
        tomli_w.dump(toml_content, f)

    return toml_path


def from_file(instance, toml_path):
    if isinstance(instance, Modflow6Model):
        modref = imod.mf6
    if isinstance(instance, MetaSwapModel):
        modref = imod.msw
    pkg_classes = {
        name: pkg_cls
        for name, pkg_cls in inspect.getmembers(modref, inspect.isclass)
        if issubclass(pkg_cls, IPackage)
    }

    toml_path = pathlib.Path(toml_path)
    with open(toml_path, "rb") as f:
        toml_content = tomli.load(f)

    parentdir = toml_path.parent
    for key, entry in toml_content.items():
        for pkgname, path in entry.items():
            pkg_cls = pkg_classes[key]
            instance[pkgname] = pkg_cls.from_file(parentdir / path)

    return instance
