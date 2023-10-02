import copy

import matplotlib.pyplot as plt
from example_models import create_circle_simulation

import imod

simulation = create_circle_simulation()
tmp_path = imod.util.temporary_directory()
simulation.write(tmp_path / "original", False)

idomain = simulation["GWF_1"]["disv"].dataset["idomain"]
submodel_labels = copy.deepcopy(idomain.sel({"layer": 1}))

submodel_labels.values[:67] = 0
submodel_labels.values[67:118] = 1
submodel_labels.values[118:] = 2

new_sim = simulation.split(submodel_labels)


new_sim.write(tmp_path, False)

new_sim.run()

sim_head_sub1 = imod.mf6.out.open_hds(
    tmp_path / "GWF_1_0/GWF_1_0.hds",
    tmp_path / "GWF_1_0/disv.disv.grb",
).compute()


sim_head_sub2 = imod.mf6.out.open_hds(
    tmp_path / "GWF_1_1/GWF_1_1.hds",
    tmp_path / "GWF_1_1/disv.disv.grb",
).compute()


fig, ax = plt.subplots()
sim_head_sub1.isel(time=-1, layer=0).ugrid.plot(ax=ax)
ax.set_aspect(1)


fig, ax = plt.subplots()
sim_head_sub2.isel(time=-1, layer=0).ugrid.plot(ax=ax)
ax.set_aspect(1)
pass
