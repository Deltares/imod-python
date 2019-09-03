import pathlib

import numpy as np
import pytest
import xarray as xr

from imod.wq import PreconditionedConjugateGradientSolver
from imod.wq import GeneralizedConjugateGradientSolver
from imod.wq import ParallelKrylovFlowSolver
from imod.wq import ParallelKrylovTransportSolver


@pytest.fixture(scope="module")
def load_weight_da(request):
    coords = {"y": np.arange(5.0), "x": np.arange(4.0)}
    dims = ("y", "x")
    da = xr.DataArray(np.full((5, 4), 1.0), coords, dims)
    return da


@pytest.fixture(scope="module")
def ibound_da(request):
    coords = {"layer": np.arange(1, 4), "y": np.arange(5.0), "x": np.arange(4.0)}
    dims = ("layer", "y", "x")
    da = xr.DataArray(np.full((3, 5, 4), 1.0), coords, dims)
    da[:, 0, 0] = -1.0
    return da


def test_pcg_render():
    pcg = PreconditionedConjugateGradientSolver(
        max_iter=150, inner_iter=30, hclose=0.0001, rclose=1000.0, relax=0.98, damp=1.0
    )

    compare = (
        "[pcg]\n"
        "    mxiter = 150\n"
        "    iter1 = 30\n"
        "    npcond = 1\n"
        "    hclose = 0.0001\n"
        "    rclose = 1000.0\n"
        "    relax = 0.98\n"
        "    iprpcg = 1\n"
        "    mutpcg = 0\n"
        "    damp = 1.0"
    )

    assert pcg._render() == compare


def test_gcg_render():
    gcg = GeneralizedConjugateGradientSolver(
        max_iter=150,
        inner_iter=30,
        cclose=1.0e-6,
        preconditioner="mic",
        lump_dispersion=True,
    )

    compare = (
        "[gcg]\n"
        "    mxiter = 150\n"
        "    iter1 = 30\n"
        "    isolve = 3\n"
        "    ncrs = 0\n"
        "    cclose = 1e-06\n"
        "    iprgcg = 0"
    )

    assert gcg._render() == compare


def test_compute_load_balance_weight(ibound_da):
    pksf = ParallelKrylovFlowSolver(
        max_iter=10,
        inner_iter=10,
        hclose=1.0e-4,
        rclose=100.0,
        relax=0.99,
        partition="rcb",
        load_balance_weight=None,
    )
    pksf._compute_load_balance_weight(ibound_da)
    da = pksf["load_balance_weight"]

    assert da.ndim == 2
    assert np.all(da == 3.0)


def test_pksf_render(load_weight_da):
    pksf = ParallelKrylovFlowSolver(
        max_iter=10, inner_iter=10, hclose=1.0e-4, rclose=100.0, relax=0.99
    )

    directory = pathlib.Path(".")

    compare = (
        "[pksf]\n"
        "    mxiter = 10\n"
        "    innerit = 10\n"
        "    hclosepks = 0.0001\n"
        "    rclosepks = 100.0\n"
        "    relax = 0.99\n"
        "    partopt = 0\n"
        "    isolver = 1\n"
        "    npc = 2\n"
        "    npcdef = 0\n"
        "    loadptr = None\n"
        "    pressakey = False\n"
    )

    assert pksf._render(directory=directory) == compare


def test_pksf_render_rcb(load_weight_da):
    pksf = ParallelKrylovFlowSolver(
        max_iter=10,
        inner_iter=10,
        hclose=1.0e-4,
        rclose=100.0,
        relax=0.99,
        partition="rcb",
        load_balance_weight=load_weight_da,
    )

    directory = pathlib.Path(".")

    compare = (
        "[pksf]\n"
        "    mxiter = 10\n"
        "    innerit = 10\n"
        "    hclosepks = 0.0001\n"
        "    rclosepks = 100.0\n"
        "    relax = 0.99\n"
        "    partopt = 5\n"
        "    isolver = 1\n"
        "    npc = 2\n"
        "    npcdef = 0\n"
        "    loadptr = load_balance_weight.asc\n"
        "    pressakey = False\n"
    )

    assert pksf._render(directory=directory) == compare


def test_pkst_render(load_weight_da):
    pkst = ParallelKrylovTransportSolver(
        max_iter=1000,
        inner_iter=30,
        cclose=1e-6,
        relax=0.98,
        partition="uniform",
        solver="bicgstab",
        preconditioner="ilu",
        debug=False,
    )

    directory = pathlib.Path(".")

    compare = (
        "[pkst]\n"
        "    mxiter = 1000\n"
        "    innerit = 30\n"
        "    cclosepks = 1e-06\n"
        "    relax = 0.98\n"
        "    partopt = 0\n"
        "    isolver = 2\n"
        "    npc = 2\n"
        "    loadptr = None\n"
        "    pressakey = False\n"
    )

    assert pkst._render(directory=directory) == compare


def test_pkst_render_rcb(load_weight_da):
    pkst = ParallelKrylovTransportSolver(
        max_iter=1000,
        inner_iter=30,
        cclose=1e-6,
        relax=0.98,
        partition="rcb",
        solver="bicgstab",
        preconditioner="ilu",
        debug=False,
        load_balance_weight=load_weight_da,
    )

    directory = pathlib.Path(".")

    compare = (
        "[pkst]\n"
        "    mxiter = 1000\n"
        "    innerit = 30\n"
        "    cclosepks = 1e-06\n"
        "    relax = 0.98\n"
        "    partopt = 5\n"
        "    isolver = 2\n"
        "    npc = 2\n"
        "    loadptr = load_balance_weight.asc\n"
        "    pressakey = False\n"
    )

    assert pkst._render(directory=directory) == compare
