from imod.flow import HorizontalAnisotropy
import pathlib


def test_anisotropy(basic_dis, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    ibound, _, _ = basic_dis
    anisotropy = HorizontalAnisotropy(anisotropy_factor=1.0, anisotropy_angle=45.0)
    nlayer = len(ibound["layer"])

    to_render = get_render_dict(anisotropy, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (ani), 1, HorizontalAnisotropy, ['anisotropy_factor', 'anisotropy_angle']\n"
        "002, 003\n"
        '1, 1, 001, 1.000, 0.000, 1.0, ""\n'
        '1, 1, 002, 1.000, 0.000, 1.0, ""\n'
        '1, 1, 003, 1.000, 0.000, 1.0, ""\n'
        '1, 1, 001, 1.000, 0.000, 45.0, ""\n'
        '1, 1, 002, 1.000, 0.000, 45.0, ""\n'
        '1, 1, 003, 1.000, 0.000, 45.0, ""'
    )

    rendered = anisotropy._render_projectfile(**to_render)

    assert rendered == compare