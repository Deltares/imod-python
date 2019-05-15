from imod.wq import AdvectionTVD, AdvectionModifiedMOC


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
        "    nlsink = 0\n"
        "    npsink = 15\n"
    )
    assert out == compare
