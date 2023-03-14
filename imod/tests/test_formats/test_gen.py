import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg

import imod

TEST_GEOMETRIES = {
    "point": sg.Point(0.0, 0.0),
    "line": sg.LineString(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    ),
    "polygon": sg.Polygon(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    ),
    "circle": sg.Point(0.0, 0.0).buffer(1.0),
    "rectangle": sg.box(0.0, 0.0, 1.0, 1.0),
}


def approximately_equal(a: sg.Polygon, b: sg.Polygon) -> bool:
    """
    A circle or rectangle might not be exactly identical after writing to GEN.
    In case of a rectangle, the vertices might be in a different order.
    In case of a circle, the vertices might have shifted in general.
    """
    return (
        np.allclose(
            np.array(a.centroid.coords),
            np.array(b.centroid.coords),
        )
        and np.isclose(a.area, b.area)
        and np.allclose(a.bounds, b.bounds)
    )


def test_gen_invalid_feature_string():
    from imod.formats.gen.gen import vertices

    geom = TEST_GEOMETRIES["circle"]
    string = "cicle"
    with pytest.raises(ValueError):
        vertices(geom, string)

    # Or bad combination
    geom = TEST_GEOMETRIES["polygon"]
    string = "point"
    with pytest.raises(ValueError):
        vertices(geom, string)


def test_gen_invalid_feature_type():
    from imod.formats.gen.gen import vertices

    geom = sg.MultiPoint(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    string = ""
    with pytest.raises(TypeError):
        vertices(geom, string)


def test_gen_shapely_gen_conversion():
    from imod.formats.gen.gen import (
        from_circle,
        from_rectangle,
        to_circle,
        to_line,
        to_point,
        to_polygon,
        to_rectangle,
    )

    geom = TEST_GEOMETRIES["circle"]
    assert approximately_equal(geom, from_circle(to_circle(geom)[0]))

    geom = TEST_GEOMETRIES["rectangle"]
    assert approximately_equal(geom, from_rectangle(to_rectangle(geom)[0]))

    geom = TEST_GEOMETRIES["point"]
    assert geom.equals(sg.Point(to_point(geom)[0][0]))

    geom = TEST_GEOMETRIES["line"]
    assert geom.equals(sg.LineString(to_line(geom)[0]))

    geom = TEST_GEOMETRIES["polygon"]
    assert geom.equals(sg.Polygon(to_polygon(geom)[0]))


@pytest.mark.parametrize("ftype", ["point", "line", "polygon", "circle", "rectangle"])
def test_gen_single_feature(tmp_path, ftype):
    geom = TEST_GEOMETRIES[ftype]
    df = pd.DataFrame()
    gdf = gpd.GeoDataFrame(df, geometry=[geom])
    gdf["feature_type"] = ftype
    path = tmp_path / f"{ftype}.gen"
    imod.gen.write(path, gdf, feature_type="feature_type")
    back = imod.gen.read(path)
    assert (back["feature_type"] == ftype).all()
    if ftype in ("circle", "rectangle"):
        # Gotta do a different check, geometries won't be exactly the same
        geom_actual = back["geometry"].iloc[0]
        geom_expected = gdf["geometry"].iloc[0]
        expected = gdf.drop(columns="geometry").sort_index(axis=1)
        actual = back.drop(columns="geometry").sort_index(axis=1)
        assert expected.equals(actual)
        assert approximately_equal(geom_actual, geom_expected)
    else:
        assert gdf.sort_index(axis=1).equals(back.sort_index(axis=1))


def test_gen_multi_feature(tmp_path):
    df = pd.DataFrame()
    gdf = gpd.GeoDataFrame(df, geometry=list(TEST_GEOMETRIES.values()))
    gdf["feature_type"] = list(TEST_GEOMETRIES.keys())
    path = tmp_path / "multi_feature.gen"
    imod.gen.write(path, gdf, feature_type="feature_type")
    back = imod.gen.read(path)
    assert gdf.shape == back.shape
    assert back["feature_type"].equals(gdf["feature_type"])


@pytest.mark.parametrize("ftype", ["point", "line", "polygon", "circle", "rectangle"])
def test_gen_single_feature__infer_type(tmp_path, ftype):
    geom = TEST_GEOMETRIES[ftype]
    df = pd.DataFrame()
    gdf = gpd.GeoDataFrame(df, geometry=[geom])
    gdf["feature_type"] = ftype
    path = tmp_path / f"{ftype}.gen"
    imod.gen.write(path, gdf)
    back = imod.gen.read(path)
    if ftype in ("polygon", "circle", "rectangle"):
        assert (back["feature_type"] == "polygon").all()
    elif ftype == "line":
        assert (back["feature_type"] == "line").all()
    elif ftype == "point":
        assert (back["feature_type"] == "point").all()
    else:
        raise ValueError("wrong ftype")


def test_gen_column_truncation(tmp_path):
    MAX_NAME_WIDTH = 11
    ftype = "point"
    geom = TEST_GEOMETRIES[ftype]
    df = pd.DataFrame()
    gdf = gpd.GeoDataFrame(df, geometry=[geom, geom])
    a = "abc_def_ghi_jkl_mno_pqr_stu_vwx"
    b = a[:7]
    gdf[a] = [0, 0]
    gdf[b] = [1, 1]
    path = tmp_path / "name_truncation.gen"
    imod.gen.write(path, gdf)
    back = imod.gen.read(path)
    # Test for maximum length of 11
    assert sorted(back.columns) == [b, a[:MAX_NAME_WIDTH], "feature_type", "geometry"]

    # Make sure the column content isn't truncated
    df = pd.DataFrame()
    gdf = gpd.GeoDataFrame(df, geometry=[geom, geom])
    gdf[a] = [a, a]
    gdf["abc_def"] = [a, a]
    path = tmp_path / "column_truncation.gen"
    imod.gen.write(path, gdf)
    back = imod.gen.read(path)
    assert (back[b] == a).all()
    assert (back[a[:MAX_NAME_WIDTH]] == a).all()


def test_gen_empty_column(tmp_path):
    geom = TEST_GEOMETRIES["point"]
    df = pd.DataFrame()
    gdf = gpd.GeoDataFrame(df, geometry=[geom])
    gdf["empty"] = None
    path = tmp_path / "empty-column.gen"
    imod.gen.write(path, gdf)


def test_parse_ascii_points():
    from imod.formats.gen.gen import parse_ascii_points

    lines = [
        "1, 0.0, 0.0",
        "2, 1.0, 0.0",
        "3, 2.0, 0.0",
        "10, 0.0, 1.0",
        "11, 0.0, 2.0",
        "END",
    ]
    gdf = parse_ascii_points(lines)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert (gdf.geometry.geom_type == "Point").all()
    assert np.array_equal(gdf["id"], [1, 2, 3, 10, 11])


def test_parse_ascii_segment_linestrings():
    from imod.formats.gen.gen import parse_ascii_segments

    lines = [
        "1",
        "0.0, 0.0",
        "1.0, 1.0",
        "end",
        "2",
        "0.0, 0.0",
        "1.0, 1.0",
        "2.0, 3.0",
        "end",
        "end",
    ]
    gdf = parse_ascii_segments(lines)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert (gdf.geometry.geom_type == "LineString").all()
    assert not gdf.geometry.has_z.any()
    assert np.array_equal(gdf["id"], [1, 2])


def test_parse_ascii_segment_polygon():
    from imod.formats.gen.gen import parse_ascii_segments

    lines = [
        "1",
        "0.0, 0.0",
        "1.0, 1.0",
        "2.0, 2.0",
        "0.0, 0.0",
        "end",
        "2",
        "0.0, 0.0",
        "1.0, 1.0",
        "2.0, 3.0",
        "0.0, 0.0",
        "end",
        "end",
    ]
    gdf = parse_ascii_segments(lines)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert (gdf.geometry.geom_type == "Polygon").all()
    assert not gdf.geometry.has_z.any()
    assert np.array_equal(gdf["id"], [1, 2])


def test_parse_ascii_segment_linestrings3d():
    from imod.formats.gen.gen import parse_ascii_segments

    lines = [
        "1",
        "0.0, 0.0, 10.0",
        "1.0, 1.0, 10.0",
        "end",
        "2",
        "0.0, 0.0, 10.0",
        "1.0, 1.0, 10.0",
        "2.0, 3.0, 10.0",
        "end",
        "end",
    ]
    gdf = parse_ascii_segments(lines)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert (gdf.geometry.geom_type == "LineString").all()
    assert gdf.geometry.has_z.all()
    assert np.array_equal(gdf["id"], [1, 2])


def test_parse_ascii_segment_mixed_topology_error():
    from imod.formats.gen.gen import parse_ascii_segments

    with pytest.raises(ValueError, match="Some features are linestrings"):
        lines = [
            "1",
            "0.0, 0.0",
            "1.0, 1.0",
            "end",
            "2",
            "0.0, 0.0",
            "1.0, 1.0",
            "2.0, 3.0",
            "0.0, 0.0",
            "end",
            "end",
        ]
        parse_ascii_segments(lines)


def test_read_ascii(tmp_path):
    from imod.formats.gen.gen import read_ascii

    lines = [
        "1, 0.0, 0.0",
        "2, 1.0, 0.0",
        "3, 2.0, 0.0",
        "10, 0.0, 1.0",
        "11, 0.0, 2.0",
        "END",
    ]
    path = tmp_path / "ascii-points.gen"
    with open(path, "w") as f:
        f.write("\n".join(lines))

    gdf = read_ascii(path)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert (gdf.geometry.geom_type == "Point").all()
    assert np.array_equal(gdf["id"], [1, 2, 3, 10, 11])

    lines = [
        "1",
        "0.0, 0.0, 10.0",
        "1.0, 1.0, 10.0",
        "END",
        "2",
        "0.0, 0.0, 10.0",
        "1.0, 1.0, 10.0",
        "2.0, 3.0, 10.0",
        "end",
        "END",
    ]

    path = tmp_path / "ascii-segments.gen"
    with open(path, "w") as f:
        f.write("\n".join(lines))

    gdf = read_ascii(path)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert (gdf.geometry.geom_type == "LineString").all()
    assert gdf.geometry.has_z.all()
    assert np.array_equal(gdf["id"], [1, 2])
