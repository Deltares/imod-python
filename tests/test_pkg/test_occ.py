from imod.pkg import OutputControl


def test_render():
    occ = OutputControl()

    compare = (
    "[oc]\n"
    "    savehead_p?_l? = False\n"
    "    saveconclayer_p?_l? = False\n"
    "    savebudget_p?_l? = False\n"
    "    saveheadtec_p?_l? = False\n"
    "    saveconctec_p?_l? = False\n"
    "    savevxtec_p?_l? = False\n"
    "    savevytec_p?_l? = False\n"
    "    savevztec_p?_l? = False\n"
    "    saveheadvtk_p?_l? = False\n"
    "    saveconcvtk_p?_l? = False\n"
    "    savevelovtk_p?_l? = False\n"
    )

    assert occ._render() == compare