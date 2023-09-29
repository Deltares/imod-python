import numpy as np
import xarray as xr
from example_models import create_circle_simulation

import imod
from imod.typing.grid import zeros_like
import copy

simulation = create_circle_simulation()
tmp_path  = imod.util.temporary_directory()
simulation.write(tmp_path / "original", False)

idomain = simulation["GWF_1"]["disv"].dataset["idomain"]
submodel_labels = copy.deepcopy(idomain.sel({"layer": 1}))

submodel_labels.values[:101] = 0
submodel_labels.values[101:] = 1

new_sim = simulation.split(submodel_labels)



new_sim.write(tmp_path, False)
pass