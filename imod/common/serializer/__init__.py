from typing import Literal, TypeAlias

from imod.common.interfaces.ipackageserializer import IPackageSerializer
from imod.common.serializer.netcdfserializer import NetCDFSerializer
from imod.common.serializer.zarrserializer import ZarrSerializer

EngineType: TypeAlias = Literal["netcdf4", "zarr", "zarr.zip"]


def create_package_serializer(
    engine: EngineType, mdal_compliant: bool = False, crs: str | None = None
) -> IPackageSerializer:
    match engine:
        case "netcdf4":
            return NetCDFSerializer(mdal_compliant=mdal_compliant, crs=crs)
        case "zarr":
            return ZarrSerializer(use_zip=False)
        case "zarr.zip":
            return ZarrSerializer(use_zip=True)
        case _:
            raise ValueError(f"Unrecognized engine: {engine}")
