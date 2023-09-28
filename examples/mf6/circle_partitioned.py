import numpy as np
import xarray as xr
from example_models import create_circle_simulation

import imod
from imod.typing.grid import zeros_like

simulation = create_circle_simulation()

idomain = simulation["GWF_1"]["disv"].dataset["idomain"]
submodel_labels = idomain.sel({"layer": 1})

submodel_labels.values[:101] = 0
submodel_labels.values[101:] = 1

new_sim = simulation.split(submodel_labels)

tmp_path  = imod.util.temporary_directory()

new_sim.write(tmp_path, False)
pass