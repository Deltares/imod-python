import collections
from pathlib import Path
from typing import Any, Optional

import tomli_w

from imod.common.interfaces.idict import IDict
from imod.common.serializer import EngineType
from imod.logging.logging_decorators import standard_log_decorator
from imod.mf6.validation_settings import ValidationSettings
from imod.schemata import ValidationError


@standard_log_decorator()
def dump_model(
    model: IDict,
    directory,
    modelname,
    validate: Optional[bool] = True,
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
    if hasattr(model, "validate") and callable(getattr(model, "validate")):
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

    # simulation settings are only relevant/present for MetaSwap models (msw)
    if hasattr(model, "simulation_settings"):
        toml_content["simulation_settings"] = model.simulation_settings

    toml_path = modeldirectory / f"{modelname}.toml"
    with open(toml_path, "wb") as f:
        tomli_w.dump(toml_content, f)

    return toml_path
