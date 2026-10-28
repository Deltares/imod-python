import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shapely

import imod
from imod.formats import isg
from imod.formats.isg.utils import to_datetime

EXAMPLES = Path(__file__).parents[3] / "examples" / "isg"


SFR, RIV = isg.SurfaceFlowRouting.SFR, isg.SurfaceFlowRouting.RIV
CS1D, CS2DREF = isg.CrossSection1D, isg.CrossSection2DReferenceHeight


@pytest.mark.parametrize(
    "path, real, asfr, nseg, kind",
    [
        ("TUT_SFR/DBASE/SFR.ISG", np.float32, SFR, 3, CS1D),
        ("TUT_SFR/DBASE/SFR2.ISG", np.float32, SFR, 3, CS1D),
        ("TUT_LAK/DBASE/SFR.ISG", np.float32, SFR, 2, CS1D),
        ("isg/rivier_nstat_NW.ISG", np.float32, RIV, 8, CS2DREF),
        ("double/rivier.ISG", np.float64, RIV, 8, CS2DREF),
    ],
)
def test_read(path, real, asfr, nseg, kind):
    model = isg.ISG.read(EXAMPLES / path)
    assert isinstance(model, isg.ISG[real])
    assert model.asfr == model.isd2.asfr == asfr
    assert len(model) == nseg
    assert model.isp.x.dtype == real
    cross_sections = model.cross_sections()
    assert len(cross_sections) == len(model.isc1)
    assert all(isinstance(c, kind) for c in cross_sections)
    segments = model.geodataframe()
    assert segments.geom_type.eq("LineString").all()
    assert len(segments.geometry.iloc[0].coords) == model.nseg[0]
    assert model.isd2.dataframe().shape[0] == len(model.isd2)


def test_read_values():
    sfr = isg.ISG[np.float32](EXAMPLES / "TUT_SFR/DBASE/SFR.ISG")
    # Segment 3 is a diversion (DIVOPT=3) from segment 1, which feeds segment 2.
    assert sfr.isd2.iupseg.tolist() == [0, 0, 0, 1, 1, 1]
    assert sfr.isd2.idownseg.tolist() == [2, 2, 0, 0, 0, 0]
    assert sfr.isd2.divopt[4] == isg.DiversionOption.FRACTION
    assert sfr.isd2.qflow[0] == np.float32(13.9)
    assert sfr.isd1.cname[:2].tolist() == ["CalcPnt FROM", "CalcPnt TO"]
    assert (sfr.iqhr.tolist(), sfr.nqhr.tolist()) == ([1, 2, 3], [1, 1, 1])
    assert sfr.isq2.dataframe().loc[1].tolist() == [10.0, 10.0, 5.0, 1.0]
    profile = sfr.cross_sections()[0]
    assert (profile.distance[0], profile.bottom[0]) == (-20.0, 2.0)
    # The last calculation point of each segment lies at its end.
    length = sfr.geodataframe().length.to_numpy()
    assert np.allclose(length, sfr.isd1.dist[1::2], rtol=1e-5)

    lak = isg.ISG.read(EXAMPLES / "TUT_LAK/DBASE/SFR.ISG")
    # Segment 1 discharges into lake 1, segment 2 flows out of it.
    assert (lak.isd2.idownseg[1], lak.isd2.iupseg[2]) == (-1, -1)

    riv = isg.ISG.read(EXAMPLES / "isg/rivier_nstat_NW.ISG")
    assert riv.isd2.ctime is None
    assert riv.isd2.idate[0] == 19800101
    assert len(riv.isc2) == np.abs(riv.isc1.n).sum()
    cross_section = riv.cross_sections()[0]
    assert (cross_section.dx, cross_section.dy) == (-25.0, -25.0)
    assert len(cross_section) == 561
    assert (cross_section.zm[0], cross_section.zc[0], cross_section.zp[0]) == (
        3,
        88,
        15,
    )


def test_double_matches_single():
    single = isg.ISG.read(EXAMPLES / "isg/rivier_nstat_NW.ISG")
    double = isg.ISG.read(EXAMPLES / "double/rivier.ISG")
    assert np.array_equal(double.isp.x, single.isp.x)
    assert np.array_equal(double.isd2.wlvl, single.isd2.wlvl)
    assert double.isd2.ctime[0] == "00:00:00"
    # Zc was rounded differently when converting: up to 1 cm.
    for d, s in zip(double.cross_sections(), single.cross_sections()):
        assert np.array_equal(d.zm, s.zm) and np.array_equal(d.zp, s.zp)
        assert np.abs(d.zc.astype(int) - s.zc).max() <= 1


def test_cross_section_kinds():
    packed = np.array([(-1, -50, 5)], dtype=[("zm", "<i2"), ("zc", "i1"), ("zp", "i1")])
    isc1 = isg.ISC1[np.float32](
        n=[2, -3, -2, -2],
        iref=[1, 3, 6, 8],
        dist=[0.0, 0.0, 0.0, 0.0],
        cname=["p", "r", "rr", "mixed"],
    )
    isc2 = isg.ISC2[np.float32](
        a=[-1.0, 1.0, 25.0, 10.0, 35.0, -25.0, 10.0, -25.0, 10.0],
        b=[1.0, 1.0, 25.0, 20.0, 20.0, -25.0, 20.0, 25.0, 20.0],
        c=[0.03, 0.03, 9.5, 1.5, 2.5, 9.5, packed.view("<f4")[0], 9.5, 1.5],
    )
    profile, raster, reference, mixed = isc1.cross_sections(isc2)
    assert isinstance(profile, isg.CrossSection1D)
    assert isinstance(raster, isg.CrossSection2D)
    assert isinstance(reference, isg.CrossSection2DReferenceHeight)
    assert raster.bottom().tolist() == [1.5, 2.5]
    assert raster.geodataframe().area.tolist() == [625.0, 625.0]
    assert reference.bottom().tolist() == [-1.5]
    # Like iMOD, only DX < 0 and DY < 0 means packed Zm, Zc and Zp.
    assert isinstance(mixed, isg.CrossSection2D)
    # Without a segment, a 1-D cross-section is not located.
    assert np.isnan(profile.x)


def test_read_list_directed(tmp_path):
    # A header of only NSEG means ASFR=0; values may be separated by blanks;
    # rows after the NSEG rows are ignored.
    source = EXAMPLES / "isg/rivier_nstat_NW.ISG"
    for path in source.parent.glob("rivier_nstat_NW.IS*"):
        shutil.copy(path, tmp_path / path.name)
    rows = source.read_text().splitlines()[1:]
    text = [
        "8",
        *(row.replace(",", "  ") for row in rows),
        "'extra' 1 1 0 0 0 0 0 0 0 0",
    ]
    (tmp_path / source.name).write_text("\n".join(text))
    model = isg.ISG.read(tmp_path / source.name)
    expected = isg.ISG.read(source)
    assert (model.asfr, model.labels) == (RIV, ())
    assert model.label.tolist() == expected.label.tolist()
    assert np.array_equal(model.nseg, expected.nseg)


def test_invalid_reference():
    model = isg.ISG.read(EXAMPLES / "TUT_SFR/DBASE/SFR.ISG")
    data = {**dict(model), "iseg": np.array([1, 3, 10], dtype=np.int32)}
    with pytest.raises(ValueError, match="ISG.ISEG: records \\[3\\]"):
        isg.ISG[np.float32].model_validate(data)


def test_record_length_mismatch(tmp_path):
    path = tmp_path / "a.isp"
    np.array([16 * 256 + 247, 0, 0, 0], dtype="<i4").tofile(path)
    with pytest.raises(ValueError, match="record length of 16 bytes, expected 8"):
        isg.ISP[np.float32](path)


def test_geometry():
    sfr = isg.ISG.read(EXAMPLES / "TUT_SFR/DBASE/SFR2.ISG")
    # Calculation points lie on the first and last node of their segment.
    located = sfr.locate("isd1")
    ends = sfr.isp.geometry()[[0, 1, 2, 5, 6, 9]]
    assert shapely.distance(located.geometry.to_numpy(), ends).max() < 1e-3
    assert located["segment"].tolist() == [1, 1, 2, 2, 3, 3]
    profile = sfr.cross_sections()[0]
    assert sfr.__geo_interface__["type"] == "MultiLineString"
    assert profile.__geo_interface__["type"] == "LineString"
    # Segment 1 runs along the x-axis, so its profile runs along the y-axis.
    assert np.allclose(profile.coordinates()[0], profile.x)

    riv = isg.ISG.read(EXAMPLES / "isg/rivier_nstat_NW.ISG")
    raster = riv.cross_sections()[0]
    assert raster.__geo_interface__["type"] == "MultiPolygon"
    assert raster.geodataframe().area.iloc[0] == 625.0
    assert raster.bottom()[0] == pytest.approx(3.88)


def test_plot():
    sfr = isg.ISG.read(EXAMPLES / "TUT_SFR/DBASE/SFR2.ISG")
    riv = isg.ISG.read(EXAMPLES / "isg/rivier_nstat_NW.ISG")
    maps = (sfr, sfr.isp, sfr.cross_sections()[0], riv, riv.cross_sections()[0])
    for model in maps:
        model.plot()
    sfr.isd1.split(sfr.isd2)[0].plot()
    sfr.isq1.split(sfr.isq2)[0].plot()
    riv.isd1.split(riv.isd2)[0].plot()
    sfr.isc1.plot(sfr.isc2, "b")
    ax = riv.isd1.plot(riv.isd2, "wlvl")
    assert len(ax.get_lines()) == len(riv.isd1)
    plt.close("all")
    # A blank CTIME is midnight.
    time = to_datetime(np.array([20161112]), np.array(["        "]))
    assert time[0] == pd.Timestamp("2016-11-12")


def test_open_projectfile_data(tmp_path):
    entry = " 1,2, 001, 1.0, 0.0, -999.99, '{}'\n"
    paths = [EXAMPLES / "TUT_SFR/DBASE/SFR.ISG", EXAMPLES / "TUT_SFR/DBASE/SFR2.ISG"]
    prj = tmp_path / "isg.prj"
    prj.write_text(
        "0001,(ISG),1, iMOD segment rivers\nSTEADY-STATE\n001,002\n"
        + "".join(entry.format(path) for path in paths)
    )
    data, repeats = imod.formats.prj.open_projectfile_data(prj)
    assert set(data) == {"isg-1", "isg-2"}
    assert data["isg-2"]["isg"][0].filepath.name == "SFR2.ISG"
    assert data["isg-1"]["time"] == [None]
    assert repeats == {}
