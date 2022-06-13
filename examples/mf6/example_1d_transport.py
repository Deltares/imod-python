# this example is taken from https://modflow6-examples.readthedocs.io/en/master/_examples/ex-gwt-mt3dms-p01.html
# as explained there, the setup is a simple 1d homogeneous aquifer with a steady state flow field of constant
# velocity.
# The benchmark consists of 4 transport problems that are modeled using this flow field. Here we have modeled these 4
# transport problems as a single simulation with multiple species.
# In all cases the initial concentration in the domain is zero, but water entering the domain has a concentration of 1.
# species_a is transported with zero diffusion or dispersion and the concentration distribution should show a sharp front, but due to the numerical method
# we see some smearing, which is expected.
# species_b has a sizeable dispersivity and hence shows more smearing than species_a but the same centre of mass
# species_c has linear sorption and therefore the concentration doesn't enter the domain as far as species_a or species_b,
# but the front of the solute plume has the same overall shape as for species_a or species_b
# species_d has linear sorption and first order decay, and this changes the shape of the front of the solute plume


import tempfile
from datetime import date, timedelta

import matplotlib
import numpy as np
import xarray as xr

import imod.mf6
import imod.util


# helper function for creating a transport model that simulates advection, dispersion (but zero molecular diffusion),
# First order decay is modeled if the decay parameter is not zero.
def create_transport_model(flowmodel, speciesname, dispersivity, retardation, decay):

    rhobulk = 1150.0
    porosity = 0.25

    tpt_model = imod.mf6.GroundwaterTransportModel(flowmodel, speciesname)
    tpt_model["advection"] = imod.mf6.AdvectionUpstream()
    tpt_model["Dispersion"] = imod.mf6.Dispersion(
        diffusion_coefficient=0.0,
        longitudinal_horizontal=dispersivity,
        transversal_horizontal1=0.0,
        xt3d_off=False,
        xt3d_rhs=False,
    )

    # compute the sorption coefficient based on the desired retardation factor and the bulk density.
    # because of this, the exact value of bulk density does not matter for the solution.
    if retardation != 1.0:
        sorption = "linear"
        kd = (retardation - 1.0) * porosity / rhobulk
    else:
        sorption = None
        kd = 1.0

    tpt_model["storage"] = imod.mf6.MobileStorage(
        porosity=porosity,
        decay=decay,
        decay_sorbed=decay,
        bulk_density=rhobulk,
        distcoef=kd,
        decay_order="first",
        sorption=sorption,
    )

    tpt_model["ic"] = imod.mf6.InitialConditions(start=0.0)
    tpt_model["oc"] = imod.mf6.OutputControl(
        save_concentration="all", save_budget="last"
    )
    tpt_model.take_discretization_from_model(flowmodel)
    return tpt_model


# helper function for creating iterable given starting date and number of days
def daterange(date1, numberdays):
    for n in range(numberdays):
        yield date1 + timedelta(n)


nlay = 1
nrow = 2
ncol = 101

dx = 10.0
xmin = 0.0
xmax = dx * ncol

layer = [1]
y = [0.5, 1.5]
x = np.arange(xmin, xmax, dx) + 0.5 * dx

grid_dims = ("layer", "y", "x")
grid_coords = {"layer": layer, "y": y, "x": x}
grid_shape = (nlay, nrow, ncol)

grid = xr.DataArray(np.ones(grid_shape, dtype=int), coords=grid_coords, dims=grid_dims)

bottom = xr.full_like(grid, -1.0, dtype=float)

gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["ic"] = imod.mf6.InitialConditions(0.0)

# Constant head
constant_head = xr.full_like(grid, np.nan, dtype=float)
constant_head[..., 0] = 60.0
constant_head[..., 100] = 0.0

# Constant head associated concentration
constant_conc = xr.full_like(grid, np.nan, dtype=float)
constant_conc[..., 0] = 1.0
constant_conc[..., 100] = 0.0
constant_conc = constant_conc.expand_dims(
    species=["species_a", "species_b", "species_c", "species_d"]
)

gwf_model["edges_bc"] = imod.mf6.ConstantHead(constant_head, constant_conc)

# hydraulic conductivity
kxx = xr.full_like(grid, 1.0, dtype=float)
gwf_model["darcy_flow"] = imod.mf6.NodePropertyFlow(
    icelltype=1,
    k=kxx,
    k33=kxx,
    variable_vertical_conductance=True,
    dewatered=True,
    perched=True,
)

# discretization
gwf_model["discretization"] = imod.mf6.StructuredDiscretization(
    top=0.0,
    bottom=bottom,
    idomain=grid,
)

gwf_model["output"] = imod.mf6.OutputControl(save_head="all", save_budget="all")

gwf_model["storage"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)

simulation = imod.mf6.Modflow6Simulation("1d_tpt_benchmark")
simulation["flowmodel"] = gwf_model

simulation["tpt_a"] = create_transport_model(gwf_model, "species_a", 0.0, 1.0, 0.0)
simulation["tpt_b"] = create_transport_model(gwf_model, "species_b", 10.0, 1.0, 0.0)
simulation["tpt_c"] = create_transport_model(gwf_model, "species_c", 10.0, 5.0, 0.0)
simulation["tpt_d"] = create_transport_model(gwf_model, "species_d", 10.0, 5.0, 0.002)

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
    linear_acceleration="bicgstab",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
simtimes = daterange(date(2000, 1, 1), 2000)
simulation.create_time_discretization(additional_times=simtimes)

with tempfile.TemporaryDirectory() as tempdir:

    simulation.write(tempdir, False)
    simulation.run()

    # open the concentration results
    sim_concentration_a = imod.mf6.out.open_conc(
        tempdir / "tpt_a/tpt_a.ucn",
        tempdir / "flowmodel/discretization.dis.grb",
    )
    sim_concentration_b = imod.mf6.out.open_conc(
        tempdir / "tpt_b/tpt_b.ucn",
        tempdir / "flowmodel/discretization.dis.grb",
    )
    sim_concentration_c = imod.mf6.out.open_conc(
        tempdir / "tpt_c/tpt_c.ucn",
        tempdir / "flowmodel/discretization.dis.grb",
    )
    sim_concentration_d = imod.mf6.out.open_conc(
        tempdir / "tpt_d/tpt_d.ucn",
        tempdir / "flowmodel/discretization.dis.grb",
    )
    final_a = sim_concentration_a.sel(time=1999, y=15)
    final_b = sim_concentration_b.sel(time=1999, y=15)
    final_c = sim_concentration_c.sel(time=1999, y=15)
    final_d = sim_concentration_d.sel(time=1999, y=15)

    matplotlib.pyplot.scatter(list(range(0, 101, 1)), final_a.values, label="a")
    matplotlib.pyplot.scatter(list(range(0, 101, 1)), final_b.values, label="b")
    matplotlib.pyplot.scatter(list(range(0, 101, 1)), final_c.values, label="c")
    matplotlib.pyplot.scatter(list(range(0, 101, 1)), final_d.values, label="d")
    matplotlib.pyplot.xlabel("cell index")
    matplotlib.pyplot.ylabel("concentration")
    matplotlib.pyplot.legend(loc="upper right")

    input()
