from datetime import datetime

import pandas as pd

from imod.util.expand_repetitions import resample_timeseries


def test_timeseries_resampling():
    # fmt: off
    data = [
        {  "time": datetime(1989, 1, 1), "rate": 100, "x": 0,"y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },  # output time 1
        {"time": datetime(1989, 1, 3), "rate": 200, "x": 0, "y": 0,  "id": "ID", "filt_top": 20, "filt_bot": 10,},
        { "time": datetime(1989, 1, 4), "rate": 300, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        {"time": datetime(1989, 1, 5), "rate": 400, "x": 0, "y": 0,"id": "ID", "filt_top": 20, "filt_bot": 10, },  # output time 2
        { "time": datetime(1989, 1, 6), "rate": 500, "x": 0,  "y": 0,"id": "ID","filt_top": 20, "filt_bot": 10,}, # output time 3
    ]  
    # fmt: on
    # initialize data of lists.

    # setup panda table
    timeseries = pd.DataFrame(
        data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 5), datetime(1989, 1, 6)]

    new_timeseries = resample_timeseries(timeseries, new_dates)
    # fmt: off
    expected_data = [
        { "time": datetime(1989, 1, 1), "rate": 175, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        { "time": datetime(1989, 1, 5), "rate": 400, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        { "time": datetime(1989, 1, 6), "rate": 500, "x": 0, "y": 0,  "id": "ID",  "filt_top": 20, "filt_bot": 10,},
    ]
    # fmt: on
    # setup panda table
    expected_timeseries = pd.DataFrame(
        expected_data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    assert new_timeseries.equals(expected_timeseries)


def test_timeseries_resampling_2():
    # fmt: off
    data = [
        { "time": datetime(1989, 1, 1, 14, 0, 0), "rate": 100, "x": 0, "y": 0, "id": "ID",  "filt_top": 20,  "filt_bot": 10, },  # after output time 1
        {"time": datetime(1989, 1, 3),  "rate": 200, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        { "time": datetime(1989, 1, 4), "rate": 300, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        {"time": datetime(1989, 1, 5),  "rate": 400, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },  # output time 2
        {"time": datetime(1989, 1, 6),  "rate": 500, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, }, # output time 3 is
    ]  
    # fmt: on
    # initialize data of lists.

    # setup panda table
    timeseries = pd.DataFrame(
        data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 5), datetime(1989, 1, 6)]

    new_timeseries = resample_timeseries(timeseries, new_dates)
    # fmt: off
    expected_data = [
        { "time": datetime(1989, 1, 1), "rate": 160.416667, "x": 0,"y": 0, "id": "ID", "filt_top": 20,"filt_bot": 10, },
        { "time": datetime(1989, 1, 5), "rate": 400, "x": 0,"y": 0, "id": "ID", "filt_top": 20,"filt_bot": 10,},
        { "time": datetime(1989, 1, 6), "rate": 500, "x": 0, "y": 0,"id": "ID","filt_top": 20, "filt_bot": 10,},
    ]
    # fmt: on
    # setup panda table
    expected_timeseries = pd.DataFrame(
        expected_data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    pd.testing.assert_frame_equal(
        new_timeseries, expected_timeseries, check_dtype=False
    )


def test_timeseries_resampling_3():
    # fmt: off
    data = [
        { "time": datetime(1989, 1, 1), "rate": 100, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10,},  # output time 1
        { "time": datetime(1989, 1, 3), "rate": 200, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10,  },
        { "time": datetime(1989, 1, 4), "rate": 300, "x": 0, "y": 0, "id": "ID", "filt_top": 20,"filt_bot": 10,},
        {  "time": datetime(1989, 1, 5),  "rate": 400, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },  # output time 2
        {  "time": datetime(1999, 1, 6), "rate": 500,  "x": 0,"y": 0, "id": "ID","filt_top": 20, "filt_bot": 10,},
    ]  # 10 years later than output time 3
    # fmt: on
    # initialize data of lists.

    # setup panda table
    timeseries = pd.DataFrame(
        data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 5), datetime(1989, 1, 6)]

    new_timeseries = resample_timeseries(timeseries, new_dates)

    # fmt: off
    expected_data = [
        {"time": datetime(1989, 1, 1), "rate": 175,"x": 0,"y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10},
        {"time": datetime(1989, 1, 5), "rate": 400,"x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        {"time": datetime(1989, 1, 6), "rate": 400,  "x": 0, "y": 0, "id": "ID","filt_top": 20, "filt_bot": 10, },
    ]
    # fmt: on
    # setup panda table
    expected_timeseries = pd.DataFrame(
        expected_data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    pd.testing.assert_frame_equal(
        new_timeseries, expected_timeseries, check_dtype=False
    )
