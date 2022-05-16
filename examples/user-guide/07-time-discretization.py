"""
Model time discretization
=========================

iMOD Python provides nice functionality to discretize your models into stress
periods, depending on the timesteps you assigned your boundary conditions. This
functionality is activated with the ``create_time_discretization()`` method.

"""

# %%
# Basics
# ------
#
# To demonstrate the ``create_time_discretization()`` method, we first have to
# create a Model object. In this case we'll use a Modflow 6 simulation, but the
# ``imod.wq.SeawatModel`` and imod.flow.ImodflowModel also support this. Wel'll
# start off with the usual imports:

import numpy as np
import pandas as pd
import xarray as xr

import imod

# %%
# We can discretize a simulation as follows, this creates a
# TimeDiscretization object under the key ``"time_discretization"``.

simulation = imod.mf6.Modflow6Simulation("example")
simulation.create_time_discretization(
    additional_times=["2000-01-01", "2000-01-02", "2000-01-04"]
)

simulation["time_discretization"]

# %%
# To view the data inside TimeDiscretization object:

simulation["time_discretization"].dataset

# %%
# Notice that even though we specified three points in time, only two
# timesteps are included in the ``time`` coordinate, this is because Modflow
# requires a start time and a duration of each stress period. iMOD Python
# therefore uses three points in time to compute two stress periods with a
# duration.

simulation["time_discretization"].dataset["timestep_duration"]

# %%
# These two stress periods use their respective start time in their ``time``
# coordinate.
#
# Boundary Conditions
# -------------------
#
# The ``create_time_discretization`` method becomes especially useful if we add boundary
# conditions to our groundwater model. We'll first still have to initialize a
# groundwater flow model though:

gwf_model = imod.mf6.GroundwaterFlowModel()

# %%
# Next, we can assign a Constant Head boundary with two timesteps:

chd_times = pd.to_datetime(["2000-01-01", "2000-01-04"])
chd_data = xr.DataArray(data=np.ones((2,)), dims=("time",), coords={"time": chd_times})

gwf_model["chd"] = imod.mf6.ConstantHead(head=chd_data)

gwf_model["chd"].dataset

# %%
# We'll also assign a Recharge boundary with two timesteps, which differ from
# the ConstantHead boundary:

rch_times = pd.to_datetime(["2000-01-01", "2000-01-02"])
rch_data = xr.DataArray(data=np.ones((2,)), dims=("time",), coords={"time": rch_times})

gwf_model["rch"] = imod.mf6.Recharge(rate=rch_data)

gwf_model["rch"].dataset

# %%
# We can now let iMOD Python figure out how the simulation's time should be
# discretized. It is important that we provide an endtime, otherwise the
# duration of the last stress period cannot be determined:

endtime = pd.to_datetime(["2000-01-06"])

simulation_bc = imod.mf6.Modflow6Simulation("example_bc")
simulation_bc["gwf_1"] = gwf_model

simulation_bc.create_time_discretization(additional_times=endtime)

simulation_bc["time_discretization"].dataset

# %%
# Notice that iMOD Python figured out that the two boundary conditions, both
# with two timesteps, should lead to three stress periods!
#
# Specifying extra settings
# -------------------------
#
# The ``TimeDiscretization`` package also supports other settings, like the
# amount of timesteps which Modflow should use within a stress period, as well
# as a timestep multiplier, to gradually increase the timesteps modflow uses
# within a stress period. This can be useful when boundary conditions change
# very abruptly between stress periods. These settings are set by accessing
# their respective variables in the ``dataset``.

times = simulation_bc["time_discretization"].dataset.coords["time"]

simulation_bc["time_discretization"].dataset["timestep_multiplier"] = 1.5
simulation_bc["time_discretization"].dataset["n_timesteps"] = xr.DataArray(
    data=[2, 4, 4], dims=("time",), coords={"time": times}
)

simulation_bc["time_discretization"].dataset

# %%
