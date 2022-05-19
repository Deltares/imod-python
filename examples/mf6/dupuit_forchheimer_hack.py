"""
Dupuit-Forchheimer "hack"
=========================

In contrast to earlier versions of MODFLOW, MODFLOW6 only officially supports
fully 3D simulations. This means that the number of layers doubles minus one:
before, heads were only computed for aquifers while in MODFLOW6 heads are
computed for both aquifers and aquitards.

This has some downsides: the required amount of RAM increases, and computation
times mildly increase (by roughly 70%). However, MODFLOW6's logic can be
modified relatively easy through its API, provided by ``xmipy`` in Python.
This requires a script utilizing ``xmipy``, and customized MODFLOW6 input.
This example demonstrates both.

XMI script
==========

In a nutshell, this hack consists of:

* Write MODFLOW6 input with reduced layers, representing the aquifers
  exclusively.
* As we cannot rely on MODFLOW6 to use the aquitard thickness to compute
  aquitard resistance, we assign it ourselves to the K33 entry of the
  NodePropertyFlow package.
* Initialize MODFLOW6, fetch the K33 values from memory, and assign these
  to the saturated conductance (CONDSAT).
* Run the MODFLOW6 simulation to completion.

We'll start by implementing a DupuitForchheimerSimulation class.
"""
#%%

import datetime

import numpy as np
import pandas as pd
import xarray as xr
from xmipy import XmiWrapper

import imod

# %%
# Some type hinting.

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray

# %%
# Simulation class
# ----------------
#
# We'll start by defining an ordinary Simulation class. This class behaves the
# same as a regular MODFLOW6 simulation (i.e. when running ``mf6.exe```).


class Simulation:
    """
    Run all stress periods in a simulation.
    """

    def __init__(self, wdir: str, name: str):
        self.modelname = name
        self.mf6 = XmiWrapper(lib_path="libmf6.dll", working_directory=wdir)
        self.mf6.initialize()
        self.max_iter = self.mf6.get_value_ptr("SLN_1/MXITER")[0]
        shape = np.zeros(1, dtype=np.int32)
        self.ncell = self.mf6.get_grid_shape(1, shape)[0]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.finalize()

    def do_iter(self, sol_id: int) -> bool:
        """Execute a single iteration"""
        has_converged = self.mf6.solve(sol_id)
        return has_converged

    def update(self):
        # We cannot set the timestep (yet) in Modflow
        # -> set to the (dummy) value 0.0 for now
        self.mf6.prepare_time_step(0.0)
        self.mf6.prepare_solve(1)
        # Convergence loop
        self.mf6.prepare_solve(1)
        for kiter in range(1, self.max_iter + 1):
            has_converged = self.do_iter(1)
            if has_converged:
                print(f"MF6 converged in {kiter} iterations")
                break
        self.mf6.finalize_solve(1)
        # Finish timestep
        self.mf6.finalize_time_step()
        current_time = self.mf6.get_current_time()
        return current_time

    def get_times(self):
        """Return times"""
        return (
            self.mf6.get_start_time(),
            self.mf6.get_current_time(),
            self.mf6.get_end_time(),
        )

    def run(self):
        start = datetime.datetime.now()

        _, current_time, end_time = self.get_times()
        while current_time < end_time:
            current_time = self.update()

        stop = datetime.datetime.now()
        print(
            f"Elapsed run time: {stop - start} (hours: minutes: seconds)."
            f"Simulation terminated normally."
        )

    def finalize(self):
        self.mf6.finalize()


# %%
# Dupuit Forchheimer Simulation
# -----------------------------
#
# Next, we'll inherit from the basic simulation and add some additional methods
# to extract the K33 values, and use it to set the CONDSAT array of the model.


class DupuitForchheimerSimulation(Simulation):
    def __init__(self, wdir: str, name: str):
        super().__init__(wdir, name)
        self.set_resistance()

    def conductance_index(
        self,
        vertical: BoolArray,
    ):
        """
        We've looked at the upper diagonal half of the coefficient matrix.

        While this means that j is always larger than i, it does not mean that cell
        i is always overlying cell j; the cell numbering may be arbitrary.

        In our convention, the k33 values is interpreted as the resistance to the
        cell BELOW it. Should now a case arise when j > i, but with cell j overlying
        cell i, we should use the k33 value of cell j to set the conductance.

        # TODO: reduced numbers?
        """
        mf6 = self.mf6
        modelname = self.modelname
        top = mf6.get_value_ptr(f"{modelname}/DIS/TOP")
        bottom = mf6.get_value_ptr(f"{modelname}/DIS/BOT")
        # Collect cell-to-cell connections
        # Python is 0-based, Fortran is 1-based
        # TODO: grab IAUSR and JAUSR if NODEREDUCED?
        ia = mf6.get_value_ptr(f"{modelname}/CON/IA") - 1
        ja = mf6.get_value_ptr(f"{modelname}/CON/JA") - 1
        # Convert compressed sparse row (CSR) to row(i) and column(j) numbers:
        n = np.diff(ia)
        i = np.repeat(np.arange(self.ncell), n)
        j = ja
        # Get the upper diagonal, and only the vertical entries.
        upper = j > i
        i = i[upper][vertical]
        j = j[upper][vertical]
        # Now find out which cell is on top, i or j.
        top_i = top[i]
        bottom_j = bottom[j]
        take_i = top_i >= bottom_j
        take_j = ~take_i
        # Create the index with the appropriate values of i and j.
        index = np.empty(i.size, dtype=np.int32)
        index[take_i] = i[take_i]
        index[take_j] = j[take_j]
        return index

    def set_resistance(self):
        mf6 = self.mf6
        modelname = self.modelname
        # Grab views on the MODFLOW6 memory:
        area = mf6.get_value_ptr(f"{modelname}/DIS/AREA")
        k33 = mf6.get_value_ptr(f"{modelname}/NPF/K33")
        condsat = mf6.get_value_ptr(f"{modelname}/NPF/CONDSAT")
        ihc = mf6.get_value_ptr(f"{modelname}/CON/IHC")
        vertical = ihc == 0
        new_cond = area / k33
        cell_i = self.conductance_index(vertical)
        condsat[vertical] = new_cond[cell_i]


# %%
# Creating input
# ==============
#
# As mentioned, the primary thing is to write the resistance of the aquitard
# layers into K33. Unfortunately, there is one more hurdle: in MODFLOW6, all
# layers are assumed to be contiguous in depth for DIS and DISV discretization
# packages. This means that for a layer, the bottom of the overlying layer is
# equal its top. DISU is an exception, and allows separate top and bottom
# values. Consequently, we will write all the model as if it is a DISU model.
#
# We can do with relative easy, by creating a
# ``LowLevelUnstructuredDiscretization`` instance via its ``from_dis`` method.
# Additionally, all packages features a ``to_disu`` method. Otherwise, we
# assemble and write the model as usual.
#
# However, let's start with a small fully 3D benchmark. The model features:
#
# * recharge across the entire top layer;
# * two aquifers of 50 m separated by an aquitard of 10 m;
# * a drain in the center.

nlay = 3
nrow = 150
ncol = 151
shape = (nlay, nrow, ncol)

dx = 10.0
dy = -10.0
xmin = 0.0
xmax = dx * ncol
ymin = 0.0
ymax = abs(dy) * nrow
dims = ("layer", "y", "x")

layer = np.array([1, 2, 3])
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x}

idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
bottom = xr.DataArray([40.0, 30.0, 0.0], {"layer": layer}, ("layer",))
top = xr.DataArray([50.0])
icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
recharge = xr.full_like(idomain.sel(layer=1), 0.001)

# We'll assume a resistance of 10 days, for a ditch of 2 m wide.
conductance = xr.full_like(idomain.sel(layer=1), np.nan)
elevation = xr.full_like(idomain.sel(layer=1), np.nan)
conductance[:, 75] = 20.0
elevation[:, 75] = 42.0

# %%
# For comparison, we set horizontal conductivity of the aquitard to a tiny value.

k = xr.DataArray([10.0, 1.0e-6, 10.0], {"layer": layer}, ("layer",))
k33 = xr.DataArray([10.0, 0.1, 10.0], {"layer": layer}, ("layer",))

gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=top, bottom=bottom, idomain=idomain
)
gwf_model["drn"] = imod.mf6.Drainage(
    elevation=elevation,
    conductance=conductance,
    print_input=True,
    print_flows=True,
    save_flows=True,
)
gwf_model["ic"] = imod.mf6.InitialConditions(head=45.0)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=icelltype,
    k=k,
    k33=k33,
)
gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
gwf_model["rch"] = imod.mf6.Recharge(recharge)

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("ex01-twri")
simulation["GWF_1"] = gwf_model
# Define solver settings
simulation["solver"] = imod.mf6.Solution(
    print_option="summary",
    csv_output=False,
    no_ptc=True,
    outer_dvclose=1.0e-4,
    outer_maximum=500,
    under_relaxation=None,
    inner_dvclose=1.0e-4,
    inner_rclose=0.001,
    inner_maximum=100,
    linear_acceleration="cg",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
simulation.time_discretization(["2020-01-01", "2020-01-02"])

# %%

simulation.write("model3d")
# %%
# Let's run the simulation with our class defined above.

with Simulation("model3d", "model3d") as xmi_simulation:
    xmi_simulation.run()

# %%

head3d = imod.mf6.open_hds_like("model3d/GWF_1/GWF_1.hds", idomain)

# %%
head3d.isel(time=0, layer=0).plot.contourf()

# %%

headdiff = head3d.isel(time=0, layer=0) - head3d.isel(time=0, layer=1)
headdiff.plot.imshow()

# %%
# We can now create the reduced model.

nlay = 2
nrow = 150
ncol = 151

layer = np.array([1, 2])
y = np.arange(ymax, ymin, dy) + 0.5 * dy
x = np.arange(xmin, xmax, dx) + 0.5 * dx
coords = {"layer": layer, "y": y, "x": x}

shape = (nlay, nrow, ncol)
idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
bottom = idomain * xr.DataArray([40.0, 0.0], {"layer": layer}, ("layer",))
top = idomain * xr.DataArray([50.0, 30.0], {"layer": layer}, ("layer",))
icelltype = idomain * xr.DataArray([1, 0], {"layer": layer}, ("layer",))
recharge = xr.full_like(idomain.sel(layer=1), 0.001)

# We'll assume a resistance of 10 days, for a ditch of 2 m wide.
conductance = xr.full_like(idomain.sel(layer=1), np.nan)
elevation = xr.full_like(idomain.sel(layer=1), np.nan)
conductance[:, 75] = 20.0
elevation[:, 75] = 42.0

# %%
# For comparison, we set horizontal conductivity of the aquitard to a tiny value.

k = idomain * xr.DataArray([10.0, 10.0], {"layer": layer}, ("layer",))
k33 = idomain * xr.DataArray([100.0, 100.0], {"layer": layer}, ("layer",))

gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["disu"] = imod.mf6.LowLevelUnstructuredDiscretization.from_dis(
    top=top, bottom=bottom, idomain=idomain
)
# %%
gwf_model["drn"] = imod.mf6.Drainage(
    elevation=elevation,
    conductance=conductance,
    print_input=True,
    print_flows=True,
    save_flows=True,
).to_disu()
gwf_model["ic"] = imod.mf6.InitialConditions(head=45.0)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=icelltype,
    k=k,
    k33=k33,
).to_disu()
gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
gwf_model["rch"] = imod.mf6.Recharge(recharge).to_disu()

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("dupuit")
simulation["GWF_1"] = gwf_model
# Define solver settings
simulation["solver"] = imod.mf6.Solution(
    print_option="summary",
    csv_output=False,
    no_ptc=True,
    outer_dvclose=1.0e-4,
    outer_maximum=500,
    under_relaxation=None,
    inner_dvclose=1.0e-4,
    inner_rclose=0.001,
    inner_maximum=100,
    linear_acceleration="cg",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
simulation.time_discretization(["2020-01-01", "2020-01-02"])
# %%

simulation.write("dupuit")

# %%

with DupuitForchheimerSimulation("dupuit", "GWF_1") as xmi_sim:
    xmi_sim.run()

# %%

ncell = 45300
path = "dupuit/GWF_1/GWF_1.hds"
disu_coords = {"node": np.arange(1, ncell + 1)}
d = {"ncell": ncell, "name": "head", "coords": disu_coords}
head = imod.mf6.out.disu.open_hds(path, d, True)

# %%
from typing import Any, Dict


def unstack(da: xr.DataArray, dim: str, coords: Dict[str, Any]):
    """
    Unstack existing dimension into multiple new dimensions with new
    coordinates.
    """
    new = da.copy()
    new.coords[dim] = pd.MultiIndex.from_product(
        [v.values for v in coords.values()], names=list(coords.keys())
    )
    return new.unstack(dim)


head_df = unstack(head, "node", dict(idomain.coords))

# %%

delta = head3d.isel(time=0, layer=0) - head_df.isel(time=0, layer=0)

# %%
