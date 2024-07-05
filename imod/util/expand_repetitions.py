import itertools
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


def expand_repetitions(
    repeat_stress: List[datetime], time_min: datetime, time_max: datetime
) -> Dict[datetime, datetime]:
    expanded = {}
    for year, date in itertools.product(
        range(time_min.year, time_max.year + 1),
        repeat_stress,
    ):
        newdate = date.replace(year=year)
        if newdate < time_max:
            expanded[newdate] = date
    return expanded


def resample_timeseries(well_rate, times):
    output_frame = pd.DataFrame(times)
    output_frame = output_frame.rename(columns={0: "time"})
    intermediate_df = pd.merge(
        output_frame,
        well_rate,
        how="outer",
        on="time",
    ).fillna(method="ffill")

    # the entries before the start of the well timeseries do not have data yet, so we fill them in here
    if intermediate_df["time"].values[0] < well_rate["time"].values[0]:
        intermediate_df.loc[intermediate_df["time"] < well_rate["time"][0], "rate"] = (
            0.0
        )
        intermediate_df.loc[intermediate_df["time"] < well_rate["time"][0], "x"] = (
            well_rate["x"][0]
        )
        intermediate_df.loc[intermediate_df["time"] < well_rate["time"][0], "y"] = (
            well_rate["y"][0]
        )
        intermediate_df.loc[intermediate_df["time"] < well_rate["time"][0], "id"] = (
            well_rate["id"][0]
        )
        intermediate_df.loc[
            intermediate_df["time"] < well_rate["time"][0], "filt_top"
        ] = well_rate["filt_top"][0]
        intermediate_df.loc[
            intermediate_df["time"] < well_rate["time"][0], "filt_bot"
        ] = well_rate["filt_bot"][0]

    # compute time difference from perious to current row
    time_diff_col = intermediate_df["time"].diff()
    intermediate_df.insert(7, "time_to_next", time_diff_col.values)

    # shift the new column 1 place down so that they become the time to the next row
    intermediate_df["time_to_next"][0:-1] = intermediate_df["time_to_next"][1:]
    intermediate_df["time_to_next"][-1] = (
        np.nan
    )  # the last one isn't used for anything but set it to NaN anyway

    output_frame = pd.merge(output_frame, intermediate_df)
    for i in range(len(times) - 1):
        output_frame["rate"][i] = integrate_timestep_rate(
            intermediate_df, times[i], times[i + 1]
        )

    return output_frame.drop("time_to_next", axis=1)


def integrate_timestep_rate(well_rate, time_0, time_1):
    delta_time = time_1 - time_0
    timestep_data = well_rate.loc[
        (well_rate["time"] >= time_0) & (well_rate["time"] < time_1)
    ]
    if len(timestep_data) == 1:
        return timestep_data["rate"].values[0]
    else:
        rate = 0
        for row in range(len(timestep_data)):
            rate += (
                timestep_data["rate"].iloc[row]
                * timestep_data["time_to_next"].iloc[row].total_seconds()
            )
        return rate / delta_time.total_seconds()
