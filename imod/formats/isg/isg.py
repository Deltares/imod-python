# %%
import struct

from os import PathLike
from typing import Union, NamedTuple

import numpy as np

# %%

_WIDTHS = {
    2295: 8,
    3319: 12,
    4343: 16,
    5367: 20,
    9463: 36,
    11511: 44,
    12535: 48,
    16631: 64,
}

IntArray = np.ndarray


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
                ("pointer", np.int32),
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


class Isc2Data(NamedTuple):
    manning: np.ndarray  # 1:1 with ISC1 points
    href_z: np.ndarray  # 1:1 with ISC1 points
    href_bed: np.ndarray  # 1:1 with ISC1 points
    z_index: np.ndarray  # which ISC1 points
    z: np.ndarray  # 1:N with ISC1 points
    bed_index: np.ndarray  # which ISC2 points
    bed: np.ndarray # 1:N with ISC1 points
    isc_type: IntArray


def alt_cumsum(a):
    """
    Alternative cumsum, always starts at 0 and omits the last value of the
    regular cumsum.
    """
    out = np.empty(a.size, a.dtype)
    out[0] = 0
    np.cumsum(a[:-1], out=out[1:])
    return out


def explicit_index(pointer: IntArray, n: IntArray):
    if len(pointer) == 0:
        return []
    increment = alt_cumsum(np.ones(n.sum(), dtype=int)) - np.repeat(alt_cumsum(n), n)
    return np.repeat(pointer, n) + increment


def cast_isc2(isc2: np.ndarray, dtype: str, isc1: np.ndarray):
    """
    Depending on value of the columns, the records in the ISC2 file have up to
    four (!) different meanings.
    
    Parameters
    ----------
    data: np.void
    dtype: str
        float32 or float64
    isc1: np.array of dtype isd_type
        
    Returns
    -------
    manning: np.void
    href_z: np.void
    href_bed: np.void
    z: np.void
    bed: np.void
    """
    if dtype == "float32":
        int1 = np.int8
        int2 = np.int16
    elif dtype == "float64":
        int1 = np.int16
        int2 = np.int32
    else:
        raise ValueError(f"Expected float32 or float64, got {dtype}")
    
    href_dtype = np.dtype(
        [
            ("dx", dtype),
            ("dy", dtype),
            ("href", dtype),
        ]
    )
    z_dtype = np.dtype(
        [
            ("x", dtype),
            ("y", dtype),
            ("z", dtype),
        ]
    )
    bed_dtype = np.dtype(
        [
            ("x", dtype),
            ("y", dtype),
            ("z_riverbed_m", int2),
            ("z_riverbed_cm", int1),
            ("inundation_area", int1),
        ]
    )
    
    n = isc1["n"]
    pointer = isc1["pointer"] - 1  # Compensate for Fortran 1-based indexing
    n_isc = n.size
    isc_id = np.arange(n_isc, dtype=int)
    isc_type = np.empty(n_isc, dtype=int)
    
    # Split based on the N value
    # Then, the z values must be split further.
    is_manning = n > 0
    is_other = n < 0
    if not (n[is_manning] == 1).all():
        raise ValueError("N > 1 detected for ISC Manning roughness coefficient")
    manning = isc2[pointer[is_manning]]
    href_index = pointer[is_other]
    href = isc2[href_index].view(href_dtype)
    
    # href values then have a number of points: either the riverbed elevation
    # (z) or a (bizarre) inundation area.
    pointer_z = pointer[is_other + 1]
    n_z = n[is_other] - 1  # We've already extracted href
    is_z = (href["dx"]) > 0 & (href["dy"] > 0)
    is_bed = ~is_z
    n_zz = abs(n_z[is_z])
    n_bed = abs(n_z[is_bed])
    z_index = explicit_index(pointer_z[is_z], n_zz) 
    bed_index = explicit_index(pointer_z[is_bed], n_bed)
    z = isc2[z_index].view(z_dtype)
    bed = isc2[bed_index].view(bed_dtype)
    
    isc_type[is_manning] = 0
    isc_type_z = isc_type[is_other]
    isc_type_z[is_z] = 1
    isc_type_z[is_bed] = 2
    isc_type[is_other] = isc_type_z
    return Isc2Data(
        manning,
        href[is_z],
        href[is_bed],
        np.repeat(isc_id[is_z], n_zz),
        z,
        np.repeat(isc_id[is_bed], n_bed),
        bed,
        isc_type,
    )


def read_isc2(path, isc1):
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
    return cast_isc2(data, dtype, isc1)


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
isc1 = read_isc1(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISC1")
# %%
isc2 = read_isc2(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISC2", isc1)
# %%
isd1 = read_isd1(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISD1")
isd2 = read_isd2(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISD2")
# %%
#isg = read_isc1(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISG")
isp = read_isp(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISP")
isq1 = read_isq1(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISQ1")
isq2 = read_isq2(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.ISQ2")
ist1 = read_ist1(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.IST1")
ist2 = read_ist2(r"c:\tmp\imodformats\isg\RIVIEREN_MORIA_AMIGO_AZURE_19900101-20200401_dag_mediaan.IST2")
# %%


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = isc2.bed["x"]
xcopy = x.copy()
y = isc2.bed["y"]
x = x[xcopy > 150_000]
y = y[xcopy > 150_000]
ii = isc2.bed_index[xcopy > 150_000]
ax.scatter(x, y, c=ii)

# %%

# %%
