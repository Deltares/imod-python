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
    dropped. If no records are valid, the rate is set to 0.0.

    Parameters
    ----------
    well_rate:  pd.DataFrame
        input timeseries for a single well

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with the mean rate and the location columns
        from the first record.
    """
    # Take first item
    output_frame = well_rate.iloc[[0]].drop(columns=["time", "rate"])
    # Compute mean.
    rate = well_rate["rate"].mean()
    col_index = output_frame.shape[1]
    output_frame.insert(col_index, "rate", 0.0 if pd.isna(rate) else rate)
    return output_frame


def _align_records_to_output_times(
    well_rate: pd.DataFrame, output_frame: pd.DataFrame, times: pd.Index | list
) -> pd.DataFrame:
    """
    Merge well records and output times, so that every row holds the record in
    effect at that time, and clip records before the first output time. NaN
    rates in the records are kept, except those before the first valid rate,
    which become zero.
    """
    # Look up the rate of each row from the record in effect, so a NaN rate stays
    # a property of its record instead of being filled by the forward fill.
    well_rate_numbered = well_rate.assign(record_nr=np.arange(len(well_rate)))
    intermediate_df = pd.merge(
        output_frame,
        well_rate_numbered,
        how="outer",
        on="time",
        validate="m:m",
    ).ffill()
    rates = pd.Series(well_rate["rate"].to_numpy(dtype=float, na_value=np.nan))
    rate = intermediate_df["record_nr"].map(rates)
    is_leading = rate.notna().cumsum() == 0
    intermediate_df["rate"] = rate.mask(is_leading, 0.0)
    intermediate_df = intermediate_df.drop(columns="record_nr")
    # Only keep the last well record BEFORE the first output time, earlier
    # records would distort the weighted averages.
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
    """Add the duration until the next row and the rate volume"""
    # Compute time difference from previous to current row
    time_diff_col = intermediate_df["time"].diff()
    col_index = intermediate_df.shape[1] - 1
    intermediate_df.insert(col_index, "time_to_next", time_diff_col.values)

    # Shift the column up so each row holds the time until the next row
    intermediate_df["time_to_next"] = intermediate_df["time_to_next"].shift(-1)

    intermediate_df["duration_sec"] = intermediate_df["time_to_next"].dt.total_seconds()
    # A NaN duration drops the row from both the volume and duration sums.
    intermediate_df.loc[intermediate_df["rate"].isna(), "duration_sec"] = np.nan
    intermediate_df["volume"] = (
        intermediate_df["rate"] * intermediate_df["duration_sec"]
    )


def resample_timeseries(
    well_rate: pd.DataFrame,
    times: list[datetime] | pd.DatetimeIndex,
    skipna: bool = False,
) -> pd.DataFrame:
    """
    Resample the rate timeseries of a single well to the given output times.

    The rate at each output time is the average rate over the period until the
    next output time, weighted by duration. The last record before the first 
    output time sets the rate at the start, the rate is zero before the first 
    record, and the last output time gets the last well rate.

    NaN rates are forward filled by default. With ``skipna=True`` they are left
    out of the weights instead. NaN rates before the first valid rate and
    periods with only NaN rates count as zero, and the last output time gets
    the last valid rate.

    Parameters
    ----------
    well_rate: pd.DataFrame
        Timeseries of a single well, with columns "time", "rate", "x", "y",
        "id", "filt_top", "filt_bot" and "index".
    times: list of datetime or pd.DatetimeIndex
        Times on which the output dataframe should have entries.
    skipna: bool, default False
        Leave NaN rates out of the weights instead of forward filling them.

    Returns
    -------
    pd.DataFrame
        Timeseries with the same columns as ``well_rate``, with one row per
        entry in ``times``.
    """
    is_steady_state = len(times) == 0
    if not skipna:
        well_rate = well_rate.assign(rate=well_rate["rate"].ffill())

    output_frame = pd.DataFrame(times)
    output_frame = output_frame.rename(columns={0: "time"})
    intermediate_df = _align_records_to_output_times(well_rate, output_frame, times)
    _zero_fill_before_first_record(intermediate_df, well_rate)
    _add_durations_and_volumes(intermediate_df)

    # Integrate by grouping by the period number
    intermediate_df["period_nr"] = intermediate_df["time"].isin(times).cumsum()
    # Drop the record before the first output time (period_nr 0), as it is
    # already forward filled onto the first output time. Keeping it would shift
    # all averages one row.
    intermediate_df = intermediate_df[intermediate_df["period_nr"] > 0]
    gb = intermediate_df.groupby("period_nr")

    output_frame["rate"] = (gb["volume"].sum() / gb["duration_sec"].sum()).reset_index(
        drop=True
    )
    # If last value is nan (fell outside range), pad with last valid well rate.
    last_index = output_frame.index[-1]
    if output_frame.isna().loc[last_index, "rate"]:
        valid_rates = well_rate["rate"].dropna()
        output_frame.loc[last_index, "rate"] = (
            valid_rates.iloc[-1] if len(valid_rates) > 0 else 0.0
        )
    # Periods where all rates are NaN.
    output_frame["rate"] = output_frame["rate"].fillna(0.0)

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
