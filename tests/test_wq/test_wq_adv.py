from imod.wq import AdvectionFiniteDifference, AdvectionTVD, AdvectionModifiedMOC


def test_render__DF():
    adv = AdvectionFiniteDifference(courant=1.0, weighting="upstream")
    out = adv._render()
    compare = "[adv]\n" "    mixelm = 0\n" "    percel = 1.0\n" "    nadvfd = 0\n"
    print(out)
    print(compare)
    assert out == compare

    adv = AdvectionFiniteDifference(courant=0.5, weighting="central")
    out = adv._render()
    compare = "[adv]\n" "    mixelm = 0\n" "    percel = 0.5\n" "    nadvfd = 1\n"
    print(out)
    print(compare)
    assert out == compare


def test_render__TVD():
    adv = AdvectionTVD(courant=1.0)
    out = adv._render()
    compare = "[adv]\n" "    mixelm = -1\n" "    percel = 1.0\n"
    assert out == compare


def test_render__ModifiedMOC():
    adv = AdvectionModifiedMOC()
    out = adv._render()
    compare = (
        "[adv]\n"
        "    mixelm = 2\n"
        "    percel = 1.0\n"
        "    itrack = 3\n"
        "    wd = 0.5\n"
        "    interp = 1\n"
        "    nlsink = 2\n"
        "    npsink = 40\n"
    )
    assert out == compare
