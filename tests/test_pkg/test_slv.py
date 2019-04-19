from imod.wq import PreconditionedConjugateGradientSolver
from imod.wq import GeneralizedConjugateGradientSolver


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
        "    damp = 1.0\n"
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
        "    iprgcg = 0\n"
    )

    assert gcg._render() == compare
