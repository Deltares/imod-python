from imod.pkg import AdvectionTVD


def test_render():
    adv = AdvectionTVD(courant=1.0)
    out = adv._render()
    compare = "[adv]\n" "    mixelm = -1\n" "    percel = 1.0\n"
    assert out == compare
