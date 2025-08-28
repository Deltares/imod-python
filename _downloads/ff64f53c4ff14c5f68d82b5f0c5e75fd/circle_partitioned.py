"""
Circle partitioned
==================

This example illustrates a circular model that is split into 3 submodels.
The split method returns a simulation object that can be run as is. In this
case the 3 submodels are roughly equal sized partitions that have the shape
of pie pieces.
"""

import matplotlib.pyplot as plt
from example_models import create_circle_simulation

import imod

simulation = create_circle_simulation()
tmp_path = imod.util.temporary_directory()
simulation.write(tmp_path / "original", False)

number_partitions = 5
submodel_labels = simulation.create_partition_labels(number_partitions)

# Create a simulation that is split in subdomains according to the label array.
new_sim = simulation.split(submodel_labels)
# %%
# Write the simulation input files for the new simulation.
new_sim.write(tmp_path, False)

# run the split simulation
new_sim.run()
# %%
# Visualize the computed heads in the top layer.
fig, ax = plt.subplots()
head = new_sim.open_head()

head["head"].isel(layer=0, time=-1).ugrid.plot.contourf(ax=ax)
# %%
# Visualize the flow-horizontal-face-x componenty of the balances.
fig, ax = plt.subplots()
balances = new_sim.open_flow_budget()

balances["flow-horizontal-face-x"].isel(layer=0, time=-1).ugrid.plot()
pass

# %%
