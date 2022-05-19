import textwrap

import dask.array
import numpy as np

from imod.mf6.read_input import list_input as li

DIS_DTYPE = np.dtype(
    [
        ("layer", np.int32),
        ("row", np.int32),
        ("column", np.int32),
        ("head", np.float64),
        ("conductance", np.float64),
    ]
)


def test_recarr_to_dense__dis():
    dtype = DIS_DTYPE
    recarr = np.array(
        [
            (1, 1, 1, 1.0, 10.0),
            (2, 1, 1, 2.0, 10.0),
            (3, 1, 1, 3.0, 10.0),
        ],
        dtype=dtype,
    )

    variables = li.recarr_to_dense(
        recarr,
        index_columns=["layer", "row", "column"],
        fields=["head", "conductance"],
        shape=(3, 4, 5),
    )
    assert isinstance(variables, list)
    assert len(variables) == 2
    a, b = variables
    assert np.isfinite(a).sum() == 3
    assert np.isfinite(b).sum() == 3
    assert a[0, 0, 0] == 1.0
    assert a[1, 0, 0] == 2.0
    assert a[2, 0, 0] == 3.0


def test_recarr_to_dense__disv():
    dtype = np.dtype(
        [
            ("layer", np.int32),
            ("cell2d", np.int32),
            ("head", np.float64),
            ("conductance", np.float64),
        ]
    )
    recarr = np.array(
        [
            (1, 1, 1.0, 10.0),
            (2, 1, 2.0, 10.0),
            (3, 1, 3.0, 10.0),
        ],
        dtype=dtype,
    )
    variables = li.recarr_to_dense(
        recarr,
        index_columns=["layer", "cell2d"],
        fields=["head", "conductance"],
        shape=(3, 20),
    )
    assert isinstance(variables, list)
    assert len(variables) == 2
    a, b = variables
    assert np.isfinite(a).sum() == 3
    assert np.isfinite(b).sum() == 3
    assert a[0, 0] == 1.0
    assert a[1, 0] == 2.0
    assert a[2, 0] == 3.0


def test_recarr_to_dense__disu():
    dtype = np.dtype(
        [
            ("node", np.int32),
            ("head", np.float64),
            ("conductance", np.float64),
        ]
    )
    recarr = np.array(
        [
            (1, 1.0, 10.0),
            (21, 2.0, 10.0),
            (41, 3.0, 10.0),
        ],
        dtype=dtype,
    )
    variables = li.recarr_to_dense(
        recarr,
        index_columns=["node"],
        fields=["head", "conductance"],
        shape=(60,),
    )
    assert isinstance(variables, list)
    assert len(variables) == 2
    a, b = variables
    assert np.isfinite(a).sum() == 3
    assert np.isfinite(b).sum() == 3
    assert a[0] == 1.0
    assert a[20] == 2.0
    assert a[40] == 3.0


def test_read_text_listinput(tmp_path):
    dtype = DIS_DTYPE
    path = tmp_path / "listinput.dat"
    content = textwrap.dedent(
        """\
        # layer row column head conductance
        1 1 1 1.0 10.0
        2 1 1 2.0 10.0
        3 1 1 3.0 10.0
        """
    )

    with open(path, "w") as f:
        f.write(content)

    # Test for internal input as well, for an already opened file.
    with open(path) as f:
        variables = li.read_internal_listinput(
            f,
            dtype,
            index_columns=["layer", "row", "column"],
            fields=["head", "conductance"],
            max_rows=3,
            shape=(3, 4, 5),
        )
    assert isinstance(variables, list)
    assert len(variables) == 2

    variables = li.read_external_listinput(
        path,
        dtype,
        index_columns=["layer", "row", "column"],
        fields=["head", "conductance"],
        shape=(3, 4, 5),
        binary=False,
        max_rows=3,
    )
    assert isinstance(variables, list)
    assert len(variables) == 2


def test_read_binary_listinput(tmp_path):
    path = tmp_path / "listinput.bin"
    dtype = DIS_DTYPE
    recarr = np.array(
        [
            (1, 1, 1, 1.0, 10.0),
            (2, 1, 1, 2.0, 10.0),
            (3, 1, 1, 3.0, 10.0),
        ],
        dtype=dtype,
    )
    recarr.tofile(path)

    variables = li.read_external_listinput(
        path,
        dtype,
        index_columns=["layer", "row", "column"],
        fields=["head", "conductance"],
        shape=(3, 4, 5),
        binary=True,
        max_rows=3,
    )
    assert isinstance(variables, list)
    assert len(variables) == 2


def test_read_listinput(tmp_path):
    dtype = DIS_DTYPE
    path = "package-binary.txt"
    content = "\n".join(
        [
            "open/close listinput.bin (binary)",
        ]
    )
    with open(path, "w") as f:
        f.write(content)

    binpath = tmp_path / "listinput.bin"
    dtype = DIS_DTYPE
    recarr = np.array(
        [
            (1, 1, 1, 1.0, 10.0),
            (2, 1, 1, 2.0, 10.0),
            (3, 1, 1, 3.0, 10.0),
        ],
        dtype=dtype,
    )
    recarr.tofile(binpath)

    with open(path) as f:
        variables = li.read_listinput(
            f,
            tmp_path,
            dtype,
            index_columns=["layer", "row", "column"],
            fields=["head", "conductance"],
            shape=(3, 4, 5),
            max_rows=3,
        )

    assert len(variables) == 2
    for a in variables:
        assert isinstance(a, dask.array.Array)
        assert a.shape == (3, 4, 5)
        notnull = np.isfinite(a)
        assert notnull[0, 0, 0]
        assert notnull[1, 0, 0]
        assert notnull[2, 0, 0]
