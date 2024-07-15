from datetime import datetime

import pandas as pd

from imod.util.expand_repetitions import resample_timeseries


def test_timeseries_resampling():
    # In this test, we resample a timeseries for a coarser output discretization.
    # The output times are a subset of the input times.
    timeseries = pd.DataFrame(
        {}, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    timeseries["time"] = [datetime(1989, 1, i) for i in [1, 3, 4, 5, 6]]
    timeseries["rate"] = [i * 100 for i in range(1, 6)]
    timeseries["x"] = 0
    timeseries["y"] = 0
    timeseries["id"] = "ID"
    timeseries["filt_top"] = 20
    timeseries["filt_bot"] = 10

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 5), datetime(1989, 1, 6)]

    new_timeseries = resample_timeseries(timeseries, new_dates)
    # fmt: off
    expected_data = [
        { "time": datetime(1989, 1, 1), "rate": 175.0, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        { "time": datetime(1989, 1, 5), "rate": 400.0, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        { "time": datetime(1989, 1, 6), "rate": 500.0, "x": 0, "y": 0,  "id": "ID",  "filt_top": 20, "filt_bot": 10,},
    ]
    # fmt: on
    # setup panda table
    expected_timeseries = pd.DataFrame(
        expected_data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    assert new_timeseries.equals(expected_timeseries)


def test_timeseries_resampling_2():
    # In this test, we resample a timeseries for a coarser output discretization.
    # The output times are a not a subset of the input times, and they begin earlier.
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
    # In this test, we resample a timeseries for a coarser output discretization.
    # The output times are a subset of the input times.

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


def test_timeseries_resampling_4():
    # In this test, we resample a timeseries for a coarser output discretization.
    # The output times are a not a subset of the input times, and they end later.
    #
    # fmt: off
    data = [
        { "time": datetime(1989, 1, 1, 11,0,0), "rate": 100, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10,},  # output time 1
        { "time": datetime(1989, 1, 2, 11,0,0), "rate": 100, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10,  },
        { "time": datetime(1989, 1, 3, 11,0,0), "rate": 200, "x": 0, "y": 0, "id": "ID", "filt_top": 20,"filt_bot": 10,},
        {  "time": datetime(1989, 1, 4, 11,0,0),  "rate": 300, "x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },  # output time 2
        {  "time": datetime(1989, 1, 5, 11,0,0), "rate": 300,  "x": 0,"y": 0, "id": "ID","filt_top": 20, "filt_bot": 10,},
    ]  # 10 years later than output time 3
    # fmt: on
    # initialize data of lists.

    # setup panda table
    timeseries = pd.DataFrame(
        data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 3), datetime(1999, 1, 4)]

    new_timeseries = resample_timeseries(timeseries, new_dates)

    # fmt: off
    expected_data = [
        {"time": datetime(1989, 1, 1), "rate": 77.083333,"x": 0,"y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10},
        {"time": datetime(1989, 1, 3), "rate":299.947532,"x": 0, "y": 0, "id": "ID", "filt_top": 20, "filt_bot": 10, },
        {"time": datetime(1999, 1, 4), "rate": 300,  "x": 0, "y": 0, "id": "ID","filt_top": 20, "filt_bot": 10, },
    ]
    # fmt: on
    # setup panda table
    expected_timeseries = pd.DataFrame(
        expected_data, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )

    pd.testing.assert_frame_equal(
        new_timeseries, expected_timeseries, check_dtype=False
    )
