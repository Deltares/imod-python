"""
This example illustrates a circular model that is split into 3 subnodels.
The split method retturns a simulation object that can be run as is. In this
case the 3 submodels are 3 roughly equal sized partitions that have the shape
of pie pieces.
"""
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

for iplot in range(3):
    sim_head_sub = imod.mf6.out.open_hds(
        tmp_path / f"GWF_1_{iplot}/GWF_1_{iplot}.hds",
        tmp_path / f"GWF_1_{iplot}/disv.disv.grb",
    ).compute()
    fig, ax = plt.subplots()
    sim_head_sub.isel(time=-1, layer=0).ugrid.plot(ax=ax)
    ax.set_aspect(1)
