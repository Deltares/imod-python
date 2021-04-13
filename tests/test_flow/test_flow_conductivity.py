from imod.flow import (
    HorizontalHydraulicConductivity,
    VerticalHydraulicConductivity,
    VerticalAnisotropy,
)
import pathlib


def test_horizontal_conductivity(basic_dis, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    ibound, _, _ = basic_dis
    conductivity = HorizontalHydraulicConductivity(k_horizontal=10.0)
    nlayer = len(ibound["layer"])

    to_render = get_render_dict(conductivity, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (khv), 1, HorizontalHydraulicConductivity, ['k_horizontal']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, 10.0, ""\n'
        '1, 1, 002, 1.000, 0.000, 10.0, ""\n'
        '1, 1, 003, 1.000, 0.000, 10.0, ""'
    )

    rendered = conductivity._render_projectfile(**to_render)

    assert rendered == compare


def test_vertical_conductivity(basic_dis, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    ibound, _, _ = basic_dis
    conductivity = VerticalHydraulicConductivity(k_vertical=10.0)
    nlayer = len(ibound["layer"])

    to_render = get_render_dict(conductivity, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (kvv), 1, VerticalHydraulicConductivity, ['k_vertical']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, 10.0, ""\n'
        '1, 1, 002, 1.000, 0.000, 10.0, ""\n'
        '1, 1, 003, 1.000, 0.000, 10.0, ""'
    )

    rendered = conductivity._render_projectfile(**to_render)

    assert rendered == compare


def test_vertical_anisotropy(basic_dis, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    ibound, _, _ = basic_dis
    anisotropy = VerticalAnisotropy(vertical_anisotropy=1.0)
    nlayer = len(ibound["layer"])

    to_render = get_render_dict(anisotropy, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (kva), 1, VerticalAnisotropy, ['vertical_anisotropy']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, 1.0, ""\n'
        '1, 1, 002, 1.000, 0.000, 1.0, ""\n'
        '1, 1, 003, 1.000, 0.000, 1.0, ""'
    )

    rendered = anisotropy._render_projectfile(**to_render)

    assert rendered == compare
