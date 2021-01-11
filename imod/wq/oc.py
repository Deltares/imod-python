from imod.wq.pkgbase import Package


class OutputControl(Package):
    """
    The Output Control Option is used to specify if head, drawdown, or budget
    data should be saved and in which format.

    Parameters
    ----------
    save_head_idf: bool, optional
        Save calculated head values in IDF format.
        Default value is False.
    save_concentration_idf: bool, optional
        Save calculated concentration values in IDF format.
        Default value is False.
    save_budget_idf: bool, optional
        Save calculated budget in IDF format.
        Default value is False.
    save_head_tec: bool, optional
        Save calculated head values in a format compatible with Tecplot.
        Default value is False.
    save_concentration_tec: bool, optional
        Save calculated concentration values in a format compatible with
        Tecplot.
        Default value is False.
    save_budget_tec: bool, optional
        Save calculated budget in a format compatible with Tecplot.
        Default value is False.
    save_head_vtk: bool, optional
        Save calculated head values in a format compatible with ParaView (VTK).
        Default value is False.
    save_concentration_vtk: bool, optional
        Save calculated concentration values in a format compatible with
        ParaView (VTK).
        Default value is False.
    save_budget_vtk: bool, optional
        Save calculated budget in a format compatible with ParaView (VTK).
        Default value is False.
    """

    __slots__ = (
        "save_head_idf",
        "save_concentration_idf",
        "save_budget_idf",
        "save_head_tec",
        "save_concentration_tec",
        "save_budget_tec",
        "save_head_vtk",
        "save_concentration_vtk",
        "save_budget_vtk",
    )
    _pkg_id = "oc"
    _template = (
        "[oc]\n"
        "    savehead_p?_l? = {save_head_idf}\n"
        "    saveconclayer_p?_l? = {save_concentration_idf}\n"
        "    savebudget_p?_l? = {save_budget_idf}\n"
        "    saveheadtec_p?_l? = {save_head_tec}\n"
        "    saveconctec_p?_l? = {save_concentration_tec}\n"
        "    savevxtec_p?_l? = {save_budget_tec}\n"
        "    savevytec_p?_l? = {save_budget_tec}\n"
        "    savevztec_p?_l? = {save_budget_tec}\n"
        "    saveheadvtk_p?_l? = {save_head_vtk}\n"
        "    saveconcvtk_p?_l? = {save_concentration_vtk}\n"
        "    savevelovtk_p?_l? = {save_budget_vtk}"
    )

    def __init__(
        self,
        save_head_idf=False,
        save_concentration_idf=False,
        save_budget_idf=False,
        save_head_tec=False,
        save_concentration_tec=False,
        save_budget_tec=False,
        save_head_vtk=False,
        save_concentration_vtk=False,
        save_budget_vtk=False,
    ):
        super(__class__, self).__init__()
        self["save_head_idf"] = save_head_idf
        self["save_concentration_idf"] = save_concentration_idf
        self["save_budget_idf"] = save_budget_idf
        self["save_head_tec"] = save_budget_tec
        self["save_concentration_tec"] = save_concentration_tec
        self["save_budget_tec"] = save_budget_tec
        self["save_head_vtk"] = save_head_vtk
        self["save_concentration_vtk"] = save_concentration_vtk
        self["save_budget_vtk"] = save_budget_vtk

    def _pkgcheck(self, ibound=None):
        pass
