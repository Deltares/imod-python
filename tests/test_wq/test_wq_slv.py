import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.wq import (GeneralizedConjugateGradientSolver,
                     ParallelKrylovFlowSolver, ParallelKrylovTransportSolver,
                     PreconditionedConjugateGradientSolver)


@pytest.fixture(scope="module")
def load_weight_da():
    coords = {"y": np.arange(5.0), "x": np.arange(4.0)}
    dims = ("y", "x")
    da = xr.DataArray(np.full((5, 4), 1.0), coords, dims)
    return da


@pytest.fixture(scope="module")
def ibound_da():
    coords = {"layer": np.arange(1, 4), "y": np.arange(5.0), "x": np.arange(4.0)}
    dims = ("layer", "y", "x")
    da = xr.DataArray(np.full((3, 5, 4), 1.0), coords, dims)
    da[:, 0, 0] = -1.0
    return da


def test_pcg_render():
    pcg = PreconditionedConjugateGradientSolver(
        max_iter=150, inner_iter=30, hclose=0.0001, rclose=1000.0, relax=0.98, damp=1.0
    )

    compare = textwrap.dedent(
        """\
        [pcg]
            mxiter = 150
            iter1 = 30
            npcond = 1
            hclose = 0.0001
            rclose = 1000.0
            relax = 0.98
            iprpcg = 1
            mutpcg = 0
            damp = 1.0"""
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

    compare = textwrap.dedent(
        """\
        [gcg]
            mxiter = 150
            iter1 = 30
            isolve = 3
            ncrs = 0
            cclose = 1e-06
            iprgcg = 0"""
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
        max_iter=10,
        inner_iter=10,
        hclose=1.0e-4,
        rclose=100.0,
        relax=0.99,
        h_fstrict=0.1,
        r_fstrict=0.01,
    )

    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [pksf]
            mxiter = 10
            innerit = 10
            hclosepks = 0.0001
            rclosepks = 100.0
            relax = 0.99
            h_fstrictpks = 0.1
            r_fstrictpks = 0.01
            partopt = 0
            isolver = 1
            npc = 2
            npcdef = 0
            loadptr = None
            pressakey = False
        """
    )

    assert pksf._render(directory=directory) == compare


def test_pksf_render_rcb(load_weight_da):
    pksf = ParallelKrylovFlowSolver(
        max_iter=10,
        inner_iter=10,
        hclose=1.0e-4,
        rclose=100.0,
        relax=0.99,
        h_fstrict=0.1,
        r_fstrict=0.01,
        partition="rcb",
        load_balance_weight=load_weight_da,
    )

    directory = pathlib.Path(".")

    compare = textwrap.dedent(
        """\
        [pksf]
            mxiter = 10
            innerit = 10
            hclosepks = 0.0001
            rclosepks = 100.0
            relax = 0.99
            h_fstrictpks = 0.1
            r_fstrictpks = 0.01
            partopt = 5
            isolver = 1
            npc = 2
            npcdef = 0
            loadptr = load_balance_weight.asc
            pressakey = False
        """
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

    compare = textwrap.dedent(
        """\
        [pkst]
            mxiter = 1000
            innerit = 30
            cclosepks = 1e-06
            relax = 0.98
            partopt = 0
            isolver = 2
            npc = 2
            loadptr = None
            pressakey = False
        """
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

    compare = textwrap.dedent(
        """\
        [pkst]
            mxiter = 1000
            innerit = 30
            cclosepks = 1e-06
            relax = 0.98
            partopt = 5
            isolver = 2
            npc = 2
            loadptr = load_balance_weight.asc
            pressakey = False
        """
    )

    assert pkst._render(directory=directory) == compare
