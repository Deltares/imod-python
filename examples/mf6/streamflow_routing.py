# %%
from pathlib import Path

import numpy as np
import xarray as xr
import xugrid as xu

import imod

# %%
nlayer = 1
nrow = 5
ncol = 5

dx = 1000.0
dy = -1000.0

x = np.arange(0, dx * ncol, dx) + 0.5 * dx
y = np.arange(0, dy * nrow, dy) + 0.5 * dy
layer = np.array([1])

idomain = xr.DataArray(
    np.ones((nlayer, nrow, ncol), dtype=int),
    dims=("layer", "y", "x"),
    coords={"layer": layer, "y": y, "x": x},
)
# %%

elevation = xr.full_like(idomain.sel(layer=1), 5.0, dtype=float)
conductance = xr.full_like(idomain.sel(layer=1), 1000.0**2, dtype=float)
rch_rate = xr.full_like(idomain.sel(layer=1), 0.001, dtype=float)

# Node properties
icelltype = xr.DataArray([0], {"layer": layer}, ("layer",))
k = xr.DataArray([10.0], {"layer": layer}, ("layer",))
bottom = xr.DataArray([-10.0], {"layer": layer}, ("layer",))

# %%

network = xu.Ugrid1d(
    node_x=np.array([1000.0, 2000.0, 3000.0, 4000.0]),
    node_y=np.array([-2500.0, -2500.0, -2500.0, -2500.0]),
    fill_value=-1,
    edge_node_connectivity=np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
        ]
    ),
)
edgedim = network.edge_dimension
uds = xu.UgridDataset(grids=[network])
uds["reach_length"] = xr.DataArray(np.full(3, 1000.0), dims=[edgedim])
uds["reach_width"] = xr.DataArray(np.full(3, 10.0), dims=[edgedim])
uds["reach_gradient"] = xr.DataArray(np.full(3, 1.0e-3), dims=[edgedim])
uds["reach_top"] = xr.DataArray(np.array([1.0e-3, 0.0, -1.0e-3]), dims=[edgedim])
uds["streambed_thickness"] = xr.DataArray(np.full(3, 0.1), dims=[edgedim])
uds["bedk"] = xr.DataArray(np.full(3, 0.1), dims=[edgedim])
uds["manning_n"] = xr.DataArray(np.full(3, 0.03), dims=[edgedim])
uds["layer"] = xr.DataArray(np.full(3, 1), dims=[edgedim])
sfr = imod.mf6.StreamFlowRouting(**uds)


# %%

gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=5.0, bottom=bottom, idomain=idomain
)
gwf_model["drn"] = imod.mf6.Drainage(
    elevation=elevation,
    conductance=conductance,
    print_input=True,
    print_flows=True,
    save_flows=True,
)
gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=icelltype,
    k=k,
    variable_vertical_conductance=True,
    dewatered=True,
    perched=True,
    save_flows=True,
)
gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
gwf_model["rch"] = imod.mf6.Recharge(rch_rate)
gwf_model["sfr"] = sfr
# %%

# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("sfr-test")
simulation["GWF_1"] = gwf_model
# Define solver settings
simulation["solver"] = imod.mf6.Solution(
    modelnames=["GWF_1"],
    print_option="summary",
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
# Collect time discretization
simulation.create_time_discretization(
    additional_times=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
)

modeldir = Path("examples/mf6/stream-routing")
simulation.write(modeldir)
# %%
simulation.run()

# %%

head = imod.mf6.open_hds(
    modeldir / "GWF_1/GWF_1.hds",
    modeldir / "GWF_1/dis.dis.grb",
)
# %%

head.isel(time=0, layer=0).plot.contourf()

# %%
