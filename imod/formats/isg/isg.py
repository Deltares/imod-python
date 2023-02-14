# %%
import struct
from os import PathLike
from typing import Union

import numpy as np

# %%

_WIDTHS = {
    2295: 8,
    4343: 16,
    5367: 20,
    9463: 36,
    11511: 44,
    12535: 48,
    16631: 64,
}


def _single_or_double(f, single_id, double_id):
    reclen_id = struct.unpack("i", f.read(4))[0]

    if reclen_id == single_id:
        dtype = "float32"
        f.read(_WIDTHS[single_id] - 4)  # padding bytes
    elif reclen_id == double_id:
        dtype = "float64"
        f.read(_WIDTHS[double_id] - 4)  # padding bytes
    else:
        raise ValueError(
            f"Expected record length identifier of {single_id} or {double_id}, "
            f"received: {reclen_id}"
        )
    return dtype


def read_isp(path):
    with open(path, "rb") as f:
        dtype = _single_or_double(f, 2295, 4343)
        xy = np.fromfile(f, dtype=dtype)
    return xy.reshape((-1, 2))


def read_is1(path):
    """
    Used for reading isd1, isc1, ist1.
    """
    with open(path, "rb") as f:
        dtype = _single_or_double(f, 11511, 12535)
        isd_dtype = np.dtype(
            [
                ("n", np.int32),
                ("reference", np.int32),
                ("distance", dtype),
                ("name", "S32"),
            ]
        )
        data = np.fromfile(f, dtype=isd_dtype)
    return data


def read_isd1(path):
    return read_is1(path)


def read_isc1(path):
    return read_is1(path)


def read_ist1(path):
    return read_is1(path)


def read_isq1(path):
    return read_is1(path)


def read_isd2(path):
    with open(path, "rb") as f:
        dtype = _single_or_double(f, 5367, 9463)
        isd_dtype = np.dtype(
            [
                ("date", np.int32),
                ("stage", dtype),
                ("bottom_elevation", dtype),
                ("resistance", dtype),
                ("infiltration_factor", dtype),
            ]
        )
        data = np.fromfile(f, dtype=isd_dtype)
    return data


def read_isd2_sfr(path):
    with open(path, "rb") as f:
        dtype = _single_or_double(f, 12535, 16631)
        isd_dtype = np.dtype(
            [
                ("date", np.int32),
                ("stage", dtype),
                ("bottom_elevation", dtype),
                ("width", dtype),
                ("thickness", dtype),
                ("conductivity", dtype),
                ("resistance", dtype),
                ("infiltration_factor", dtype),
                ("downstream_segment", np.int32),
                ("upstream_segment", np.int32),
                ("calculation_option", np.int32),
                ("diversion_option", np.int32),
                ("streamflow", dtype),
                ("runoff", dtype),
                ("precipitation", dtype),
                ("evaporation", dtype),
            ]
        )
        data = np.fromfile(f, dtype=isd_dtype)

    return data


def read_isc2(path):
    with open(path, "rb") as f:
        dtype = _single_or_double(f, 3319, 5367)
        isc_dtype = np.dtype(
            [
                ("distance", dtype),
                ("bottom_elevation", dtype),
                ("manning_roughness", dtype),
            ]
        )
        data = np.fromfile(f, dtype=isc_dtype)
    return data


def read_ist2(path):
    with open(path, "rb") as f:
        dtype = _single_or_double(f, 3319, 5367)
        ist_dtype = np.dtype(
            [
                ("date", np.int32),
                ("upstream_water_level", dtype),
                ("downstream_water_level", dtype),
            ]
        )
        data = np.fromfile(f, dtype=ist_dtype)
    return data


def read_isq2(path):
    with open(path, "rb") as f:
        dtype = _single_or_double(f, 3319, 5367)
        isq_dtype = np.dtype(
            [
                ("discharge", dtype),
                ("width", dtype),
                ("depth", dtype),
                ("factor", dtype),
            ]
        )
        data = np.fromfile(f, dtype=isq_dtype)
    return data


# %%


path = r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISD1"
a = read_isd1(path)

path = r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISD2"
b = read_isd2(path)
# %%

with open(path, "rb") as f:
    reclen_id = struct.unpack("i", f.read(4))[0]  # Lahey RecordLength Ident.

    if reclen_id == 11511:
        dtype = "float32"
        f.read(40)  # padding bytes
    elif reclen_id == 12535:
        dtype = "float64"
        f.read(44)  # padding bytes
    else:
        raise ValueError(
            f"Expected record length identifier of 2295 or 4343, received: reclen_id"
        )

    n = struct.unpack("i", f.read(4))[0]
    ref = struct.unpack("i", f.read(4))[0]
    dist = struct.unpack("f", f.read(4))[0]
    s = f.read(32).decode("utf-8")

    n2 = struct.unpack("i", f.read(4))[0]
    ref2 = struct.unpack("i", f.read(4))[0]
    dist2 = struct.unpack("f", f.read(4))[0]
    s2 = f.read(32).decode("utf-8")

# %%


a = np.full(5, s)
# %%
