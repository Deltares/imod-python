import textwrap

from imod.wq import OutputControl


def test_render():
    oc = OutputControl()

    compare = textwrap.dedent(
        """\
        [oc]
            savehead_p?_l? = False
            saveconclayer_p?_l? = False
            savebudget_p?_l? = False
            saveheadtec_p?_l? = False
            saveconctec_p?_l? = False
            savevxtec_p?_l? = False
            savevytec_p?_l? = False
            savevztec_p?_l? = False
            saveheadvtk_p?_l? = False
            saveconcvtk_p?_l? = False
            savevelovtk_p?_l? = False"""
    )

    assert oc._render() == compare
