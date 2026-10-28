# ISG

Reader for iMOD ISG-files (iMOD User Manual 5.6.1, section 9.9): a text file
of segments with eight associated binary files, related by record numbers.

Examples:
```python
import numpy as np
from imod.formats.isg import ISG

isg = ISG.read("DBASE/SFR.ISG")  # precision from the ISP-file
isg = ISG[np.float32]("DBASE/SFR.ISG")  # or explicit
isg.dataframe()  # segments
isg.geodataframe()  # segments as LineStrings
isg.isd2.dataframe()  # any associated file
isg.isc1.cross_sections(isg.isc2)  # ISC2 records per ISC1 cross-section
isg.cross_sections()  # the same, 1-D cross-sections located on their segment
isg.locate("isd1")  # calculation points as Points on their segment
isg.plot()  # segments with the located ISD1, ISC1, IST1, ISQ1 records
isg.cross_sections()[0].plot()  # a 1-D profile, or the "pixels" of a 2-D grid
isg.isd1.split(isg.isd2)[0].plot()  # levels of calculation point 1 over time
isg.isd1.plot(isg.isd2, "wlvl")  # WLVL of all calculation points
isg.isq1.split(isg.isq2)[0].plot()  # width and depth against Q
```

Models are frozen and their vectors read-only by design.

## Files

Every binary file starts with a header record that stores the record length as
`reclen * 256 + 247`. Records are numbered from 1 after that header. A
reference is a pair (first record, number of records); `0, 0` references
nothing. Reals are 4 bytes (single precision) or 8 bytes (double precision).

- **FK**: first record number of a range in the associated file.
- **index**: the \*1-files only point into their \*2-file, with a name and a
  distance along the segment.
- **geometry**: coordinates, exposed through `geometry()`, `geodataframe()`
  and `__geo_interface__`.

```mermaid
erDiagram
    ISG ||--o{ ISP : "ISEG, NSEG"
    ISG ||--o{ ISD1 : "ICLC, NCLC"
    ISG ||--o{ ISC1 : "ICRS, NCRS"
    ISG ||--o{ IST1 : "ISTW, NSTW"
    ISG ||--o{ ISQ1 : "IQHR, NQHR"
    ISD1 ||--o{ ISD2 : "IREF, N"
    ISC1 ||--o{ ISC2 : "IREF, abs(N)"
    IST1 ||--o{ IST2 : "IREF, N"
    ISQ1 ||--o{ ISQ2 : "IREF, N"

    ISG["ISG (text) · segments · geometry: LineString via ISP"] {
        int header_NSEG "header: number of segments"
        int header_ASFR "header: 0 RIV, 1 SFR; selects the ISD2 layout"
        str header_LABELS "header: labels of the ISD2 attributes"
        str LABEL "name of the segment, max 52 characters"
        int ISEG FK "first node in ISP"
        int NSEG "number of nodes"
        int ICLC FK "first calculation point in ISD1"
        int NCLC "number of calculation points (SFR: 2)"
        int ICRS FK "first cross-section in ISC1"
        int NCRS "number of cross-sections"
        int ISTW FK "first weir/structure in IST1"
        int NSTW "number of weirs/structures"
        int IQHR FK "first Q-width/depth relation in ISQ1"
        int NQHR "number of Q-width/depth relations"
    }
    ISP["ISP · nodes · geometry: Point"] {
        real X "x-coordinate of node (m)"
        real Y "y-coordinate of node (m)"
    }
    ISD1["ISD1 · index · calculation points · geometry: Point at DIST"] {
        int N "number of ISD2 records"
        int IREF FK "first ISD2 record"
        real DIST "distance from node 1 (m)"
        char32 CNAME "name of the calculation point"
    }
    ISD2["ISD2 · timeseries · RivISD2 (ASFR=0) | SfrISD2 (ASFR=1)"] {
        int IDATE "yyyymmdd"
        char8 CTIME "hh:mm:ss; SFR, and RIV double precision"
        real WLVL "water level (m+MSL)"
        real BTML "bottom level (m+MSL)"
        real RESIS "RIV: resistance (d)"
        real INFF "RIV: infiltration factor (-)"
        real WIDTH "SFR: stream width (m)"
        real THICKNESS "SFR: riverbed thickness (m)"
        real HCFACT "SFR: riverbed conductivity (m/d)"
        int IUPSEG "SFR: upstream segment, <0 lake id"
        int IDOWNSEG "SFR: downstream segment, <0 lake id"
        int CALCOPT "SFR: CalculationOption 1-5"
        int DIVOPT "SFR: DiversionOption 1-4, if IUPSEG gt 0"
        real QFLOW "SFR: inflow (m3/s)"
        real QRUNOFF "SFR: runoff (m3/s)"
        real PPTSW "SFR: precipitation (mm/d)"
        real ETSW "SFR: evaporation (mm/d)"
    }
    ISC1["ISC1 · index · cross-sections · geometry: Point at DIST"] {
        int N "gt 0: 1-D records; lt 0: 2-D records incl. header"
        int IREF FK "first ISC2 record"
        real DIST "distance from node 1 (m)"
        char32 CNAME "name of the cross-section"
    }
    ISC2["ISC2 · cross-section data · geometry: 1-D LineString or 2-D boxes"] {
        real A "DISTANCE (N gt 0), DX (first record) or X (2-D pixel)"
        real B "BOTTOM (N gt 0), DY (first record) or Y (2-D pixel)"
        real C "MRC (N gt 0), HREF (first record), Z or packed Zm, Zc, Zp (DX and DY lt 0)"
    }
    IST1["IST1 · index · weirs/structures · geometry: Point at DIST"] {
        int N "number of IST2 records"
        int IREF FK "first IST2 record"
        real DIST "distance from node 1 (m)"
        char32 CNAME "name of the weir/structure"
    }
    IST2["IST2 · timeseries"] {
        int IDATE "yyyymmdd"
        char8 CTIME "double precision only"
        real WLVL_UP "upstream water level (m+MSL)"
        real WLVL_DWN "downstream water level (m+MSL)"
    }
    ISQ1["ISQ1 · index · Q-width/depth relations · geometry: Point at DIST"] {
        int N "number of ISQ2 records"
        int IREF FK "first ISQ2 record"
        real DIST "distance from node 1 (m)"
        char32 CNAME "name of the relation"
    }
    ISQ2["ISQ2 · Q-width/depth table"] {
        real Q "discharge (m3/d)"
        real WIDTH "width (m)"
        real DEPTH "depth (m)"
        real FACTOR "obsolete factor (-)"
    }
```

Negative values are never record numbers: ISC1 `N < 0` marks a 2-D
cross-section of `abs(N)` records, `IUPSEG`/`IDOWNSEG < 0` are lake ids and
`Zp < 0` marks cells only inundated above `HREF`.

## Architecture

Each file is a Pydantic model with one numpy vector per attribute, generic over
the float type (`ISP[np.float32]`, `ISP[np.float64]`). A model given a path
reads its missing fields from the file with its own `extension()`, so the
associated files only need the path of the ISG-file.

```mermaid
classDiagram
    class FileModel {
        filepath: Path
        extension() str
        load(data) pre-validator
        parse(path) dict
        real() float type
    }
    class Table {
        columns() dict
        dataframe() DataFrame
        references() dict
        take(rows) Self
    }
    class SpatialTable {
        geometry() shapely array
        geodataframe() GeoDataFrame
        plot(ax) Axes
        #95;#95;geo_interface#95;#95;() dict
    }
    class BinaryFile {
        record(real) dtype
        parse(path) dict
        precision(path) float type
    }
    class IndexFile {
        split(data) list
        plot(data, y) Axes
    }
    class TimeSeries {
        levels
        time() DatetimeIndex
        plot() Axes
    }
    Table <|-- SpatialTable
    Table <|-- TimeSeries
    FileModel <|-- BinaryFile
    Table <|-- BinaryFile
    BinaryFile <|-- IndexFile
    IndexFile <|-- ISD1
    IndexFile <|-- ISC1
    IndexFile <|-- IST1
    IndexFile <|-- ISQ1
    BinaryFile <|-- ISP
    SpatialTable <|-- ISP
    BinaryFile <|-- RivISD2
    BinaryFile <|-- SfrISD2
    BinaryFile <|-- IST2
    TimeSeries <|-- RivISD2
    TimeSeries <|-- SfrISD2
    TimeSeries <|-- IST2
    BinaryFile <|-- ISQ2
    BinaryFile <|-- ISC2
    FileModel <|-- ISG
    SpatialTable <|-- ISG
    SpatialTable <|-- CrossSection1D
    SpatialTable <|-- BaseCrossSection2D
    BaseCrossSection2D <|-- CrossSection2D
    BaseCrossSection2D <|-- CrossSection2DReferenceHeight
    ISC1 ..> CrossSection1D : cross_sections(isc2)
    ISC1 ..> CrossSection2D : cross_sections(isc2)
    ISC1 ..> CrossSection2DReferenceHeight : cross_sections(isc2)
```

- `common.py`: base classes, `Vector` and the `References` marker.
- `isg.py`: the file models.
- `views.py`: the cross-sections, typed views of ISC2 records.
- `enums.py`, `validators.py`, `utils.py`: enums, reusable validators, file
  reading helpers.

Fields annotated with `References(file, count)` are checked in one validator
on `ISG`: each range must lie within the referenced file.
