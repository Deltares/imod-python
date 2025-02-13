import pandas as pd
import pytest

from imod.util.hfb import _prepare_index_names


def test_pandas_behavior_index_naming_expected():
    """
    Monitor whether pandas behaviour changes. Quite some logic in
    mf6/utilities/hfb depends on specific index naming and how pandas behaves.
    This tests if this behaviour goes unchanged.
    """

    df = pd.DataFrame(data={"col1": [1, 2], "col2": [1, 2]})
    df_reset = df.reset_index(drop=False)

    assert df.index.names == [None]
    assert df.index.name is None
    assert "index" in df_reset.columns

    df_roundtrip = df_reset.set_index("index")

    assert df_roundtrip.index.names == ["index"]
    assert df_roundtrip.index.name == "index"


def test_prepare_index_names():
    # Case 1: Single index, unnamed
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [1, 2]})
    df_prepared = _prepare_index_names(df.copy())

    assert df_prepared.index.names == ["index"]

    # Case 2: Single index, named
    df_index_named = df.copy()
    df_index_named.index = df.index.set_names(["index"])
    df_prepared = _prepare_index_names(df_index_named)

    assert df_prepared.index.names == ["index"]

    # Case 3: Multi index, unnamed
    df_renamed = df.rename(columns={"col1": "parts"})
    df_multi_unnamed = df_renamed.set_index([df.index, "parts"])
    assert df_multi_unnamed.index.names == [None, "parts"]
    df_prepared = _prepare_index_names(df_multi_unnamed)

    assert df_prepared.index.names == ["index", "parts"]

    # Case 4: Multi index, named
    df_renamed = df_index_named.rename(columns={"col1": "parts"})
    df_multi_unnamed = df_renamed.set_index([df_renamed.index, "parts"])
    assert df_multi_unnamed.index.names == ["index", "parts"]
    df_prepared = _prepare_index_names(df_multi_unnamed)

    assert df_prepared.index.names == ["index", "parts"]

    # Case 5: Wrong index name
    df_index_wrongname = df.copy()
    df_index_wrongname.index = df.index.set_names(["wrong_name"])
    with pytest.raises(IndexError):
        _prepare_index_names(df_index_wrongname)
