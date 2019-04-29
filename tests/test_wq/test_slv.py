from imod.wq import PreconditionedConjugateGradientSolver
from imod.wq import GeneralizedConjugateGradientSolver
from imod.wq import ParallelKrylovFlowSolver
from imod.wq import ParallelKrylovTransportSolver
import xarray as xr
import pathlib

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

def test_pksf_render():
    pksf = ParallelKrylovFlowSolver(
            max_iter=10,
            inner_iter=10,
            hclose=1.0e-4,
            rclose=100.0,
            relax=0.99
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
                "    loadpatr = None\n"
                "    pressakey = False\n"
                )
    
    assert pksf._render(directory) == compare

def test_pksf_render_rcb():
    pksf = ParallelKrylovFlowSolver(
            max_iter=10,
            inner_iter=10,
            hclose=1.0e-4,
            rclose=100.0,
            relax=0.99,
            partition="rcb",
            load_balance_weight=xr.DataArray([[1, 1], [1, 1]], 
                                             coords={"y": [0, 1], "x": [0, 1]}, 
                                             dims=("y", "x"))
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
                "    loadpatr = load_balance_weight.asc\n"
                "    pressakey = False\n"
                )
    
    assert pksf._render(directory=directory) == compare    

def test_pkst_render():
    pkst = ParallelKrylovTransportSolver(
            max_iter = 1000, 
            inner_iter = 30, 
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
            "    loadpatr = None\n"
            "    pressakey = False\n"
                )
    
    assert pkst._render(directory) == compare

def test_pkst_render_rcb():
    pkst = ParallelKrylovTransportSolver(
            max_iter = 1000, 
            inner_iter = 30, 
            cclose=1e-6,
            relax=0.98,
            partition="rcb",
            solver="bicgstab",
            preconditioner="ilu",
            debug=False,
            load_balance_weight=xr.DataArray([[1, 1], [1, 1]], 
                                 coords={"y": [0, 1], "x": [0, 1]}, 
                                 dims=("y", "x"))
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
            "    loadpatr = load_balance_weight.asc\n"
            "    pressakey = False\n"
                )
    
    assert pkst._render(directory=directory) == compare