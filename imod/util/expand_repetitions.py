import itertools
from datetime import datetime

import numpy as np
import pandas as pd

LOCATION_COLUMNS = ["x", "y", "id", "filt_top", "filt_bot", "index"]


def expand_repetitions(
    repeat_stress: list[datetime], time_min: datetime, time_max: datetime
) -> dict[np.datetime64, np.datetime64]:
    """
    Given a list of repeat stresses,  and the start and end time of the simulation,
    this function returns a dictionary indicating what repeat stress should be used
    at what time in the simulation.

    Parameters
    ----------
    repeat_stress:  list of datetime, optional
        This dict list contains contains, per topic, the period alias (a string) to
        its datetime.
    time_min: datetime
        starting time  of the simulation
    time_max: datetime
        ending time  of the simulation


    returns:
    --------
    A dictionary that can be used to repeat data for e.g.repeating stress periods such as
    seasonality without duplicating the values. The ``repeat_items``
    dimension should have size 2: the first value is the "key", the second
    value is the "value". For the "key" datetime, the data of the "value"
    datetime will be used.
    """
    expanded: dict[np.datetime64, np.datetime64] = {}
    for year, date in itertools.product(
        range(time_min.year, time_max.year + 1),
        repeat_stress,
    ):
        newdate = np.datetime64(date.replace(year=year))
        if newdate < time_max:
            expanded[newdate] = np.datetime64(date)
    return expanded


def average_timeseries(well_rate: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean value of the timeseries for a single well. Time column is
    dropped.

    Parameters
    ----------
    well_rate:  pd.DataFrame
        input timeseries for a single well
    """
    # Take first item
    output_frame = well_rate.iloc[[0]].drop(columns=["time", "rate"])
    # Compute mean
    col_index = output_frame.shape[1]
    output_frame.insert(col_index, "rate", well_rate["rate"].mean())
    return output_frame


def _forward_fill_rates_to_output_times(
    well_rate: pd.DataFrame, output_frame: pd.DataFrame, times: pd.Index | list
) -> pd.DataFrame:
    """
    Merge well records and output times, forward fill the rate to the output
    times and clip records before the first output time.
    """
    intermediate_df = pd.merge(
        output_frame,
        well_rate,
        how="outer",
        on="time",
        validate="m:m",
    ).ffill()
    # Catch case where multiple well rates precede first timestep on multiple
    # occasions, these have to be clipped to avoid including them in the
    # computation of weighted averages later in the function. We only want to
    # include the last well timestep BEFORE the first output timestep.
    n_before_first_timestep = (intermediate_df["time"] < times[0]).cumsum()
    to_preserve = n_before_first_timestep == n_before_first_timestep.max()
    return intermediate_df[to_preserve]


def _zero_fill_before_first_record(
    intermediate_df: pd.DataFrame, well_rate: pd.DataFrame
) -> None:
    """
    The entries before the start of the well timeseries do not have data yet,
    so we fill them in here. Keep rate to zero and pad the location columns with
    the first entry. Modifies intermediate_df in place.
    """
    time_before_start_input = (
        intermediate_df["time"].values < well_rate["time"].values[0]
    )
    if time_before_start_input[0]:
        intermediate_df.loc[time_before_start_input, "rate"] = 0.0
        first_well_location_values = well_rate.iloc[0][LOCATION_COLUMNS].values
        intermediate_df.loc[time_before_start_input, LOCATION_COLUMNS] = (
            first_well_location_values
        )


def _add_durations_and_volumes(intermediate_df: pd.DataFrame) -> None:
    """Add the duration until the next row and the rate volume. In place."""
    # compute time difference from perious to current row
    time_diff_col = intermediate_df["time"].diff()
    col_index = intermediate_df.shape[1] - 1
    intermediate_df.insert(col_index, "time_to_next", time_diff_col.values)

    # shift the new column 1 place down so that they become the time to the next row
    intermediate_df["time_to_next"] = intermediate_df["time_to_next"].shift(-1)

    intermediate_df["duration_sec"] = intermediate_df["time_to_next"].dt.total_seconds()
    intermediate_df["volume"] = (
        intermediate_df["rate"] * intermediate_df["duration_sec"]
    )


def resample_timeseries(
    well_rate: pd.DataFrame, times: list[datetime] | pd.DatetimeIndex
) -> pd.DataFrame:
    """
    On input, well_rate is a dataframe containing a timeseries for rate for one well
    while "times" is a list of datetimes.
    This function creates a new dataframe containing a timeseries defined on the
    times in "times".

    Parameters
    ----------
    well_rate:  pd.DataFrame
        input timeseries for a single well
    times: datetime
        list of times on which the output dataframe should have entries.

    returns:
    --------
    a new dataframe containing a timeseries defined on the times in "times".
    """
    is_steady_state = len(times) == 0

    output_frame = pd.DataFrame(times)
    output_frame = output_frame.rename(columns={0: "time"})
    intermediate_df = _forward_fill_rates_to_output_times(
        well_rate, output_frame, times
    )
    _zero_fill_before_first_record(intermediate_df, well_rate)
    _add_durations_and_volumes(intermediate_df)

    # Integrate by grouping by the period number
    intermediate_df["period_nr"] = intermediate_df["time"].isin(times).cumsum()
    # Drop the kept record before the first output time (period_nr 0). Its rate
    # is already forward filled to the first output time. If not dropped, the
    # results below shift one row, as the first group is not the first period.
    intermediate_df = intermediate_df[intermediate_df["period_nr"] > 0]
    gb = intermediate_df.groupby("period_nr")

    output_frame["rate"] = (gb["volume"].sum() / gb["duration_sec"].sum()).reset_index(
        drop=True
    )
    # If last value is nan (fell outside range), pad with last well rate.
    last_index = output_frame.index[-1]
    if output_frame.isna().loc[last_index, "rate"]:
        output_frame.loc[last_index, "rate"] = well_rate["rate"].iloc[-1]

    if is_steady_state:
        # Take first element, the slice is to force pandas to return it as
        # dataframe instead of series.
        location_dataframe = intermediate_df[LOCATION_COLUMNS].iloc[slice(0, 1), :]
        # Concat along columns and drop time column
        return pd.concat([output_frame, location_dataframe], axis=1).drop(
            columns="time"
        )
    else:
        columns_to_merge = ["time"] + LOCATION_COLUMNS
        return pd.merge(
            output_frame,
            intermediate_df[columns_to_merge],
            on="time",
            how="left",
            validate="1:m",
        )
