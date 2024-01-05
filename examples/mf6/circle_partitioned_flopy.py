"""
This example illustrates how imod-python tooling and flopy tooling can be 
used sequentially in a workflow. A simulations set up in imod-python can be
converted to flopy simulations. It can then be manipulated using 
flopy features. The output can be loaded using imod-python tooling. 

In this example we do all this to show how model splitting works in flopy and imod python,
and how the resulting output matches.  This may be of interest to the user when 
flopy supports splitting something that imod-pythone cannot (such as transport models)
"""
import os
import shutil

import flopy
import matplotlib.pyplot as plt
import numpy as np
import xugrid as xu
from example_models import create_circle_simulation
from flopy.mf6.utils import Mf6Splitter

import imod
from imod.mf6.multimodel.partition_generator import get_label_array
from imod.typing.grid import merge

# Set up the simulation and write the MF6 inputfiles 
simulation = create_circle_simulation()
tmp_path = imod.util.temporary_directory()
ip_unsplit_dir = tmp_path / "original"
simulation.write(ip_unsplit_dir, binary=False, use_absolute_paths=True)


# Load the simulation into flopy and run it
flopy_sim = flopy.mf6.MFSimulation.load(
    sim_ws=ip_unsplit_dir,
    verbosity_level=1,
)
flopy_dir = tmp_path / "flopy"
flopy_sim.set_sim_path(flopy_dir)
flopy_sim.write_simulation(silent=False)
flopy_sim.run_simulation(silent=True)

# Read the results of the unsplit simulation
cbc = imod.mf6.open_cbc(
    ip_unsplit_dir / "GWF_1/GWF_1.cbc", ip_unsplit_dir / "GWF_1/disv.disv.grb"
)

# Create a label array for splitting the simulation
two_parts = np.zeros_like(simulation["GWF_1"].domain.isel(layer=0).values)
two_parts[:97] = 0
two_parts[97:] = 1
label_array = simulation["GWF_1"].domain.isel(layer=0)
label_array.values = two_parts


# Split the flopy simulation, write it and run it
mf_splitter = Mf6Splitter(flopy_sim)
flopy_split_sim = mf_splitter.split_model(two_parts)
flopy_split_dir = tmp_path / "flopy_split"
flopy_split_sim.set_sim_path(flopy_split_dir)
flopy_split_sim.write_simulation(silent=False)
flopy_split_sim.run_simulation(silent=False)


# Load the results. Note that flopy has written some of its output to the folder where 
# the output of the model before splitting was supposed to go. That is because loading and splitting  
# a simulation in flopy does not affect the folders specified in the oc file ( which are absolute paths in this case). 
q1 = imod.mf6.open_cbc(
    ip_unsplit_dir / "GWF_1/GWF_1_0.cbc", flopy_split_dir / "gwf_1_0.disv.grb"
)
q2 = imod.mf6.open_cbc(
    ip_unsplit_dir / "GWF_1/GWF_1_1.cbc", flopy_split_dir / "gwf_1_1.disv.grb"
)

# Merge the results
orig_domain = simulation["GWF_1"].domain
q_merged = xu.zeros_like(orig_domain)
array_dict = {0: q1["npf-qx"].values, 1: q2["npf-qx"].values}
new_vel_array = mf_splitter.reconstruct_array(array_dict)
q_merged.values = new_vel_array

# Run the original imod-python simulation, load the results 
simulation.run()
original_balances = simulation.open_flow_budget()

# Compute the absolute  difference between the unsplit output and the flopy-split output
diff = original_balances["npf-qx"] - q_merged
fig, ax = plt.subplots()
diff.isel(layer=0, time=0).ugrid.plot.contourf(ax=ax)


# in areas with very low absolute values, large relative differences can correspond to a
# negligible absolute error, so we throw these out
reldif = abs(diff) / abs(original_balances["npf-qx"])
fig, ax = plt.subplots()
reldif.values = np.where(original_balances["npf-qx"].values < 1e-8 , 0, reldif.values) #
reldif.isel(layer=0, time=0).ugrid.plot.contourf(ax=ax)


# now also split the simulation with imod python, and run the simulatiohn
ip_dir = tmp_path / "ip_split"
new_sim = simulation.split(label_array)
new_sim.write(ip_dir, False)
new_sim.run()

# %%
# Visualize the flow-horizontal-face-x componenty of the balances.
balances = new_sim.open_flow_budget()

# Open the output and compute the difference between the imod-python split model and the unsplit model
fig, ax = plt.subplots()
veldif_x = original_balances["npf-qx"] - balances["npf-qx"]
veldif_y = original_balances["npf-qy"] - balances["npf-qy"]
veldif_z = original_balances["npf-qz"] - balances["npf-qz"]
veldif_x.isel(layer=0, time=-1).ugrid.plot()

# Compute and plot the relative differences
fig, ax = plt.subplots()
reldif_x = abs(veldif_x) / abs(balances["npf-qx"])
reldif_x.values = np.where(original_balances["npf-qx"].values < 1e-8, 0 , reldif_x.values)
reldif_x.isel(layer=0, time=-1).ugrid.plot()


reldif_y = veldif_y / balances["npf-qy"]
reldif_y.values = np.where(original_balances["npf-qx"].values < 1e-8, 0, reldif_y.values)
reldif_y.isel(layer=0, time=-1).ugrid.plot()

# %%
