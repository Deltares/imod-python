"""
Working with iMOD5 models in MODFLOW 6
======================================

This example shows how to work with iMOD5 models in MODFLOW 6. It demonstrates
how to convert an iMOD5 model to a MODFLOW 6 model using the `imod` package. The
example fetches an iMOD5 model, converts it to a MODFLOW 6 model, and saves it
to a temporary directory.

"""

import imod

tmpdir = imod.util.temporary_directory()

# %%
# 
#
# Fetching an iMOD5 model
# -----------------------
#
# Let's start by fetching the example data
# from the `imod.data` module. This will download a project file and
# accompanying data files to a temporary directory.

prj_dir = tmpdir / "prj"
prj_dir.mkdir(exist_ok=True, parents=True)

model_dir = imod.data.fetch_imod5_model(prj_dir)

# %% 
# 
# Let's view the model directory. It contains the project file and
# accompanying model contents.

from pprint import pprint

imod_dir_contents = list(model_dir.glob("*"))
pprint(imod_dir_contents)

# %% 
# 
# The directory contains a project file and a database folder.
# This database contains all the IDF, IPF, and GEN files that make up the
# spatial model input.
# 
# Let's look at the projectfile contents. Read the projectfile as follows:

prj_path = model_dir / "iMOD5_model.prj"
prj_content = imod.prj.read_projectfile(prj_path)
pprint(prj_content)

# %% 
# 
# This contains all the projectfile contents in a dictionary, which is quite a
# lot of information. This is too much to go through in detail. We can also open
# all data that the projectfile points to, using the
# :doc:`/api/generated/io/imod.formats.prj.open_projectfile` function.

imod5_data, period_data = imod.prj.open_projectfile_data(prj_path)
imod5_data

# %%
#
# This groups all data per package in the projectfile into a dictionary with
# DataArrays per variable.

imod5_data["riv-1"]["stage"]

# %%
#
# Let's plot the stage data of the first river package.

imod5_data["riv-1"]["stage"].isel(layer=0, drop=True).plot.imshow()

# %%
#
# This is nice enough, but we want to convert this iMOD5 model to a MODFLOW 6
# model. We can do this using
# :doc:`/api/generated/imod.mf6.simulation.Modflow6Simulation.from_imod5_data`
# method. Next to the iMOD5 data and period data, we also need to provide the
# times. These will be used to resample the asynchronous well timeseries data to
# these times. Let's therefore first create a list of times, we can use pandas
# for this:

import pandas as pd
times = pd.date_range(start="2020-01-01", periods=10, freq="MS")
times

# %%
#
# Now that we have a list of times, we can import the iMOD5 data into a MODFLOW
# 6 simulation. This might require some time, as it will convert all the iMOD5
# data to be compatible with MODFLOW . For example, the river systems with
# infiltration factors are transformed into a separate
# Drain and River package (if necessary) to get the same behavior as iMOD5's
# infiltration factors.

mf6_sim = imod.mf6.Modflow6Simulation.from_imod5_data(imod5_data, period_data, times)
mf6_sim

# %%
#
# At the moment the MODFLOW 6 simulation has quite loose solver settings:

mf6_sim["ims"]

# %%
#
# This is because by default an iMOD5 model is imported with a
# SolutionPresetModerate, which is quite loose. Let's set a stricter solver
# setting preset, by setting it to SolutionPresetSimple.

mf6_sim["ims"] = imod.mf6.SolutionPresetSimple(["imported_model"])
mf6_sim["ims"]

# %%
# 
# This has a inner_dvclose of 0.001 instead of 0.01, which is a lot stricter: a
# numerical error of 1 mm is only allowed by the solver, instead of 1 cm.
#
# Now that we have a MODFLOW 6 simulation, we can regrid it to an unstructured
# grid.

# %%
