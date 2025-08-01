from datetime import datetime

import numpy as np
import pandas as pd

from imod.util.expand_repetitions import average_timeseries, resample_timeseries


def initialize_timeseries(times: list[datetime], rates: list[float]) -> pd.DataFrame:
    timeseries = pd.DataFrame(
        {}, columns=["time", "rate", "x", "y", "id", "filt_top", "filt_bot"]
    )
    timeseries["time"] = times
    timeseries["rate"] = rates
    timeseries["x"] = 0
    timeseries["y"] = 0
    timeseries["id"] = "ID"
    timeseries["filt_top"] = 20
    timeseries["filt_bot"] = 10
    timeseries["index"] = 0

    return timeseries


def test_timeseries_resampling():
    # In this test, we resample a timeseries for a coarser output discretization.
    # The output times are a subset of the input times.
    times = [datetime(1989, 1, i) for i in [1, 3, 4, 5, 6]]
    rates = [i * 100 for i in range(1, 6)]
    timeseries = initialize_timeseries(times, rates)

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 5), datetime(1989, 1, 6)]
    new_timeseries = resample_timeseries(timeseries, new_dates)

    expected_times = [datetime(1989, 1, i) for i in [1, 5, 6]]
    expected_rates = [175.0, 400.0, 500.0]
    expected_timeseries = initialize_timeseries(expected_times, expected_rates)

    assert new_timeseries.equals(expected_timeseries)


def test_timeseries_resampling__index_nonzero_start():
    # In this test, we resample a timeseries for a coarser output
    # discretization. The output times are a subset of the input times. The
    # index of the timeseries has a non-zero start.
    times = [datetime(1989, 1, i) for i in [1, 3, 4, 5, 6]]
    rates = [i * 100 for i in range(1, 6)]
    timeseries = initialize_timeseries(times, rates)
    timeseries = timeseries.set_index(timeseries.index + 4)

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 5), datetime(1989, 1, 6)]
    new_timeseries = resample_timeseries(timeseries, new_dates)

    expected_times = [datetime(1989, 1, i) for i in [1, 5, 6]]
    expected_rates = [175.0, 400.0, 500.0]
    expected_timeseries = initialize_timeseries(expected_times, expected_rates)

    assert new_timeseries.equals(expected_timeseries)


def test_timeseries_resampling_2():
    # In this test, we resample a timeseries for a coarser output discretization.
    # The output times are a not a subset of the input times, and they begin earlier.
    times = [datetime(1989, 1, 1, 14, 0, 0)]
    times += [datetime(1989, 1, i) for i in [3, 4, 5, 6]]
    rates = [i * 100 for i in range(1, 6)]
    timeseries = initialize_timeseries(times, rates)

    # initialize data of lists.

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 5), datetime(1989, 1, 6)]
    new_timeseries = resample_timeseries(timeseries, new_dates)

    expected_times = [datetime(1989, 1, i) for i in [1, 5, 6]]
    expected_rates = [160.416667, 400.0, 500.0]
    expected_timeseries = initialize_timeseries(expected_times, expected_rates)

    pd.testing.assert_frame_equal(
        new_timeseries, expected_timeseries, check_dtype=False
    )


def test_timeseries_resampling_3():
    # In this test, we resample a timeseries for a coarser output discretization.
    # The output times are a subset of the input times.

    times = [datetime(1989, 1, i) for i in [1, 3, 4, 5]]
    times += [datetime(1999, 1, 6)]  # Ten years after last entry.
    rates = [i * 100 for i in range(1, 6)]
    timeseries = initialize_timeseries(times, rates)

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 5), datetime(1989, 1, 6)]
    new_timeseries = resample_timeseries(timeseries, new_dates)

    expected_times = [datetime(1989, 1, i) for i in [1, 5, 6]]
    expected_rates = [175.0, 400.0, 400.0]
    expected_timeseries = initialize_timeseries(expected_times, expected_rates)

    pd.testing.assert_frame_equal(
        new_timeseries, expected_timeseries, check_dtype=False
    )


def test_timeseries_resampling_4():
    # In this test, we resample a timeseries for a coarser output discretization.
    # The output times are a not a subset of the input times, and they end later.
    # initialize data of lists.

    times = [datetime(1989, 1, i, 11, 0, 0) for i in [1, 2, 3, 4, 5]]
    rates = [100, 100, 200, 300, 300]
    timeseries = initialize_timeseries(times, rates)

    new_dates = [datetime(1989, 1, 1), datetime(1989, 1, 3), datetime(1999, 1, 4)]
    new_timeseries = resample_timeseries(timeseries, new_dates)

    expected_times = [datetime(1989, 1, 1), datetime(1989, 1, 3), datetime(1999, 1, 4)]
    expected_rates = [77.083333, 299.947532, 300]
    expected_timeseries = initialize_timeseries(expected_times, expected_rates)

    pd.testing.assert_frame_equal(
        new_timeseries, expected_timeseries, check_dtype=False
    )


def test_timeseries_resampling_5():
    # In this test, we resample a timeseries to a finer output discretization.
    # The original times have several preceding timesteps, which need to be
    # clipped off.
    original_times = [datetime(1899, 1, 1), datetime(1909, 1, 1)] + [
        datetime(1989, 1, i) for i in [1, 3, 5, 7, 9]
    ]
    original_rates = [0.0, 0.0, 200.0, 200.0, 300.0, 400.0, 500.0]
    original_timeseries = initialize_timeseries(original_times, original_rates)

    new_dates = [datetime(1989, 1, 2), datetime(1989, 1, 3), datetime(1989, 1, 4)]
    new_timeseries = resample_timeseries(original_timeseries, new_dates)

    expected_times = [datetime(1989, 1, 2), datetime(1989, 1, 3), datetime(1989, 1, 4)]
    expected_rates = [200.0, 200.0, 200.0]
    expected_timeseries = initialize_timeseries(expected_times, expected_rates)

    pd.testing.assert_frame_equal(
        new_timeseries, expected_timeseries, check_dtype=False
    )


def test_timeseries_resampling_6():
    # In this test, we resample a timeseries after the last timestep of
    # timeseries, should be forward filled.
    original_times = [datetime(1899, 1, 1), datetime(1909, 1, 1)] + [
        datetime(1989, 1, i) for i in [1, 3, 5, 7, 9]
    ]
    original_rates = [0.0, 0.0, 200.0, 200.0, 300.0, 400.0, 500.0]
    original_timeseries = initialize_timeseries(original_times, original_rates)

    new_dates = [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 4)]
    new_timeseries = resample_timeseries(original_timeseries, new_dates)

    expected_times = [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 4)]
    expected_rates = [500.0, 500.0, 500.0]
    expected_timeseries = initialize_timeseries(expected_times, expected_rates)

    pd.testing.assert_frame_equal(
        new_timeseries, expected_timeseries, check_dtype=False
    )


def test_timeseries_resampling_coarsen_and_refine():
    # In this test, we resample a timeseries for a coarser output discretization.
    # Then we refine it again to the original discretization.
    # The coarsening was chosen so that after this the original timeseries should be obtained

    original_times = [datetime(1989, 1, i) for i in [1, 2, 3, 4, 5, 6]]
    original_rates = [100, 100, 200, 200, 300, 300]
    original_timeseries = initialize_timeseries(original_times, original_rates)

    coarse_times = [datetime(1989, 1, 1), datetime(1989, 1, 3), datetime(1989, 1, 5)]
    coarse_timeseries = resample_timeseries(original_timeseries, coarse_times)

    re_refined_timeseries = resample_timeseries(coarse_timeseries, original_times)

    pd.testing.assert_frame_equal(
        original_timeseries, re_refined_timeseries, check_dtype=False
    )


def test_timeseries_resampling_refine_and_coarsen():
    # In this test, we resample a timeseries for a finer output discretization.
    # Then we coarsen it again to the original discretization.
    # The refinement was chosen so that after this the original timeseries should be obtained
    original_times = [datetime(1989, 1, i) for i in [1, 2, 3, 4, 5, 6]]
    original_rates = [100, 100, 200, 200, 300, 300]
    original_timeseries = initialize_timeseries(original_times, original_rates)

    refined_times = pd.date_range(
        datetime(1989, 1, 1), datetime(1989, 1, 6), periods=121
    )
    refined_timeseries = resample_timeseries(original_timeseries, refined_times)

    re_coarsened_timeseries = resample_timeseries(refined_timeseries, original_times)

    pd.testing.assert_frame_equal(
        original_timeseries, re_coarsened_timeseries, check_dtype=False
    )


def test_mean_timeseries():
    # In this test, we compute the mean of a timeseries.
    times = [datetime(1989, 1, i) for i in [1, 3, 4, 5, 6]]
    rates = [i * 100 for i in range(1, 6)]
    timeseries = initialize_timeseries(times, rates)

    mean_timeseries = average_timeseries(timeseries)

    dummy_times = [datetime(1989, 1, 1)]
    expected_rates = np.mean(rates)
    expected_timeseries = initialize_timeseries(dummy_times, expected_rates)
    col_order = ["x", "y", "id", "filt_top", "filt_bot", "index", "rate"]
    expected_timeseries = expected_timeseries[col_order]

    pd.testing.assert_frame_equal(
        mean_timeseries, expected_timeseries, check_dtype=False
    )
