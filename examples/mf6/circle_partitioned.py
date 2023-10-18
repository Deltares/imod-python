"""
This example illustrates a circular model that is split into 3 submodels.
The split method returns a simulation object that can be run as is. In this
case the 3 submodels are roughly equal sized partitions that have the shape
of pie pieces.
"""
import copy

import matplotlib.pyplot as plt
from example_models import create_circle_simulation

import imod
from imod.mf6.partitioned_simulation_postprocessing import merge_heads
from imod.mf6.simulation import get_models

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


fig, ax = plt.subplots()
submodel_names = list(get_models(new_sim).keys())
head = merge_heads(tmp_path, submodel_names)
head["head"].isel(layer=0, time=-1).ugrid.plot.contourf(ax=ax)
# %%
