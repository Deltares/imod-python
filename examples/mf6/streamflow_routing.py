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
uds["save_flows"] = True
uds["stage_fileout"] = "sfr-stage-fileout.bin"
uds["budget_fileout"] = "sfr-budget-fileout.bin"
uds["budgetcsv_fileout"] = "sfr-budget-fileout.csv"
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

modeldir = Path("stream-routing")
simulation.write(modeldir)
# %%
simulation.run()

# %%

head = imod.mf6.open_hds(
    modeldir / "GWF_1/GWF_1.hds",
    modeldir / "GWF_1/dis.dis.grb",
)
head.isel(time=0, layer=0).plot.contourf()

# %%

budgets = imod.mf6.open_cbc(
    modeldir / "GWF_1/GWF_1.cbc",
    modeldir / "GWF_1/dis.dis.grb",
    flowja=True,
)
sfr_budgets = budgets["sfr_sfr"].compute()
sfr_budgets.isel(time=0, layer=0).plot()

# %%

import os

import dask

from imod.mf6.out.common import read_times_dvs

FloatArray = np.ndarray


def read_dvs_timestep(path, n: int, pos: int) -> FloatArray:
    """
    Reads all values of one timestep.
    """
    with open(path, "rb") as f:
        f.seek(pos)
        f.seek(52, 1)  # skip kstp, kper, pertime
        a1d = np.fromfile(f, np.float64, n)
    return a1d


def open_sfr_stages(path, network):
    # TODO: get rid of indices, just propagate size?
    indices = np.arange(network.n_edge)
    filesize = os.path.getsize(path)
    ntime = filesize // (52 + (indices.size * 8))
    coords = {"time": read_times_dvs(path, ntime, indices)}
    dask_list = []
    for i in range(ntime):
        pos = i * (52 + indices.size * 8)
        a = dask.delayed(read_dvs_timestep)(path, network.n_edge, pos)
        x = dask.array.from_delayed(a, shape=(network.n_edge,), dtype=np.float64)
        dask_list.append(x)

    daskarr = dask.array.stack(dask_list, axis=0)
    da = xr.DataArray(daskarr, coords, ("time", network.edge_dimension), name="sfr")
    return xu.UgridDataArray(da, network)


sfr_stage = open_sfr_stages(modeldir / "sfr-stage-fileout.bin", network)
sfr_stage.isel(time=0).ugrid.plot()

# %%


from scipy.sparse import csr_matrix

from imod.mf6.out import cbc


def read_imeth6_budget_column(
    cbc_path,
    count,
    dtype,
    pos,
) -> np.ndarray:
    # Wrapped function returning the budget column only.
    a = cbc.read_imeth6_budgets(cbc_path, count, dtype, pos)
    return a["budget"]


def process_sfr_flowja(sfr_flowja, network):
    # If I understand correctly, SFR doesn't allow complex junctions.
    # This means that a junction is either a confluence (N to 1),
    # or a furcation (1 to N).
    # In either case, either the flow from upstream or the flow to downstream
    # is single valued and can be uniquely identified.
    i = sfr_flowja["id1"]
    j = sfr_flowja["id2"]
    flow = sfr_flowja["flow"]
    flow_area = sfr_flowja["flow-area"]

    shape = (network.n_edge, network.n_edge)
    connectivity_matrix = csr_matrix((flow, (i, j)), shape=shape)
    # Sum along the rows (axis=1, index=i): flow to downstream
    # Sum along the columns (axis=0, index=j): flow from upstream
    downstream_flow = connectivity_matrix.sum(axis=1)
    upstream_flow = connectivity_matrix.sum(axis=0)
    connectivity_matrix.data = flow_area
    downstream_flow_area = connectivity_matrix.sum(axis=1)
    upstream_flow_area = connectivity_matrix.sum(axis=0)

    return downstream_flow, upstream_flow, downstream_flow_area, upstream_flow_area


def open_sfr_cbc(cbc_path, network):
    headers = cbc.read_cbc_headers(cbc_path)
    data = {}
    for key, header_list in headers.items():
        if key in ("flow-ja-face_sfr", "gwf_gwf_1", "storage_sfr"):
            # TODO: these have multiple columns
            # flow-ja-face contains:
            # * reach_i
            # * reach_j
            # * flow
            # * flow_area
            #
            # This should be split into four DataArrays (see above).
            #
            # gwf_gwf_1 contains:
            # * reach_i
            # * modflow_gwf_i
            # * reach-aquifer flow
            # * reach-aquifer flow area
            #
            # Separate these into two DataArrays.
            #
            # storage contains:
            # * reach_i
            # * reach_i (the same)
            # * storage
            # * volume
            #
            # Separate these into two DataArrays.
            continue

        dtype = np.dtype(
            [("id1", np.int32), ("id2", np.int32), ("budget", np.float64)]
            + [(name, np.float64) for name in header_list[0].auxtxt]
        )

        dask_list = []
        time = []
        for header in header_list:
            a = dask.delayed(read_imeth6_budget_column)(
                cbc_path, network.n_edge, dtype, header.pos
            )
            x = dask.array.from_delayed(a, shape=(network.n_edge,), dtype=np.float64)
            dask_list.append(x)
            time.append(header.totim)

        daskarr = dask.array.stack(dask_list, axis=0)
        data[key] = xu.UgridDataArray(
            xr.DataArray(
                daskarr, coords={"time": time}, dims=("time", network.edge_dimension)
            ),
            network,
        )

    return data


sfr_budgets = open_sfr_cbc(modeldir / "sfr-budget-fileout.bin", network)

# %%

list(sfr_budgets.keys())
# %%
