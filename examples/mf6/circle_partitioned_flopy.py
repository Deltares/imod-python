"""
This example illustrates a circular model that is split into 3 submodels.
The split method returns a simulation object that can be run as is. In this
case the 3 submodels are roughly equal sized partitions that have the shape
of pie pieces.
"""
import matplotlib.pyplot as plt
from example_models import create_circle_simulation
import numpy as np
import imod
from imod.mf6.multimodel.partition_generator import get_label_array
import os
import flopy
from flopy.mf6.utils import Mf6Splitter
import xugrid as xu
from imod.typing.grid import merge
import shutil

simulation = create_circle_simulation()

tmp_path = imod.util.temporary_directory()
ip_unsplit_dir = tmp_path / "original"

simulation.write(ip_unsplit_dir, binary=False, use_absolute_paths=True)


flopy_sim = flopy.mf6.MFSimulation.load(
    sim_ws=ip_unsplit_dir,
    verbosity_level=1,
)
flopy_dir = tmp_path / "flopy"
flopy_sim.set_sim_path(flopy_dir)
flopy_sim.write_simulation(silent=False)
flopy_sim.run_simulation(silent=True)
cbc = imod.mf6.open_cbc(
    ip_unsplit_dir / "GWF_1/GWF_1.cbc", ip_unsplit_dir / "GWF_1/disv.disv.grb"
)


two_parts = np.zeros_like(simulation["GWF_1"].domain.isel(layer=0).values)

two_parts[:97] = 0
two_parts[97:] = 1
label_array = simulation["GWF_1"].domain.isel(layer=0)
label_array.values = two_parts

mf_splitter = Mf6Splitter(flopy_sim)

orig_domain = simulation["GWF_1"].domain
q_merged = xu.zeros_like(orig_domain)


flopy_split_sim = mf_splitter.split_model(two_parts)
flopy_split_dir = tmp_path / "flopy_split"
flopy_split_sim.set_sim_path(flopy_split_dir)
flopy_split_sim.write_simulation(silent=False)
flopy_split_sim.run_simulation(silent=False)
q1 = imod.mf6.open_cbc(
    ip_unsplit_dir / "GWF_1/GWF_1_0.cbc", flopy_split_dir / "gwf_1_0.disv.grb"
)
q2 = imod.mf6.open_cbc(
    ip_unsplit_dir / "GWF_1/GWF_1_1.cbc", flopy_split_dir / "gwf_1_1.disv.grb"
)
array_dict = {0: q1["npf-qx"].values, 1: q2["npf-qx"].values}
new_vel_array = mf_splitter.reconstruct_array(array_dict)
q_merged.values = new_vel_array


simulation.run()
original_balances = simulation.open_flow_budget()
diff = original_balances["npf-qx"] - q_merged

reldif = abs(diff) / abs(original_balances["npf-qx"])


fig, ax = plt.subplots()
diff.isel(layer=0, time=0).ugrid.plot.contourf(ax=ax)

fig, ax = plt.subplots()
reldif.values = np.where(reldif.values > 0.8, 0, reldif.values)
reldif.isel(layer=0, time=0).ugrid.plot.contourf(ax=ax)
idomain = simulation["GWF_1"]["disv"].dataset["idomain"]

# now do the same for imod-python
ip_dir = tmp_path / "ip_split"
number_partitions = 2
# submodel_labels = get_label_array(simulation, number_partitions)
new_sim = simulation.split(label_array)
new_sim.write(ip_dir, False)
# shutil.copy(flopy_split_dir / "sim_0_1.gwfgwf",ip_dir/"GWF_1_0_GWF_1_1.gwfgwf")


# run the split simulation
new_sim.run()

# %%
# Visualize the flow-horizontal-face-x componenty of the balances.

balances = new_sim.open_flow_budget()

# balances["flow-horizontal-face-x"].isel(layer=0, time=-1).ugrid.plot()
fig, ax = plt.subplots()
original_balances["npf-qx"].isel(layer=0, time=-1).ugrid.plot()
veldif_x = original_balances["npf-qx"] - balances["npf-qx"]
veldif_y = original_balances["npf-qy"] - balances["npf-qy"]
veldif_z = original_balances["npf-qz"] - balances["npf-qz"]
veldif_x.isel(layer=0, time=-1).ugrid.plot()

fig, ax = plt.subplots()
reldif_x = abs(veldif_x) / abs(balances["npf-qx"])
# reldif_x.values = np.where(np.abs(reldif_x.values) > 0.8, 0 , reldif_x.values)
reldif_x.isel(layer=0, time=-1).ugrid.plot()


reldif_y = veldif_y / balances["npf-qy"]
reldif_y.values = np.where(np.abs(reldif_y.values) > 100, 0, reldif_y.values)
reldif_y.isel(layer=0, time=-1).ugrid.plot()


# Note: xr.identical() fails, as the regridder as a dx coord for cellsizes.
assert np.testing.assert_allclose(
    original_balances["npf-qx"].values, balances["npf-qx"].values, equal_nan=True
)
pass

# %%
