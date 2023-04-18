import numpy as np
import pandas as pd


def assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, **kwargs):
    """
    Dataframes are regularly created with "platform dependent" integer columns,
    such as int and np.intp; windows behaves differently from linux. This
    creates issues with local testing versus CI. Type checking can be disabled
    in assert_frame_equal, but we would like to check float versus int versus
    string -- we just convert any integer to 64-bit here.

    https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_frame_equal.html
    """

    def always_int64(df):
        df = df.copy()
        for column, dtype in df.dtypes.items():
            if np.issubdtype(dtype, np.integer):
                df[column] = df[column].astype(np.int64)
        return df

    left = always_int64(left)
    right = always_int64(right)
    pd.testing.assert_frame_equal(left, right, **kwargs)
