from imod.flow import Top, Bottom, Boundary, StartingHead
import pathlib
import os

def test_boundary(basic_dis, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())
    
    ibound, _, _ = basic_dis
    boundary = Boundary(ibound=ibound)
    nlayer = len(boundary["layer"])
    
    to_render = get_render_dict(boundary, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (bnd), 1, Boundary, ['ibound']\n"
        "001, 003\n"
        f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}ibound_l1.idf\n"
        f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}ibound_l2.idf\n"
        f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}ibound_l3.idf"
    )

    rendered = boundary._render_projectfile(**to_render)

    assert rendered == compare


def test_top(basic_dis, get_render_dict):
    _, top, _ = basic_dis
    top = Top(top=top)
    nlayer = len(top["layer"])
    
    to_render = get_render_dict(top, ".", None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (top), 1, Top, ['top']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, 0.0, ""\n'
        '1, 1, 002, 1.000, 0.000, -5.0, ""\n'
        '1, 1, 003, 1.000, 0.000, -35.0, ""'
    )

    rendered = top._render_projectfile(**to_render)

    assert rendered == compare


def test_bot(basic_dis, get_render_dict):
    _, _, bottom = basic_dis
    bottom = Bottom(bottom=bottom)
    nlayer = len(bottom["layer"])
    
    to_render = get_render_dict(bottom, ".", None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (bot), 1, Bottom, ['bottom']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, -5.0, ""\n'
        '1, 1, 002, 1.000, 0.000, -35.0, ""\n'
        '1, 1, 003, 1.000, 0.000, -135.0, ""'
    )

    rendered = bottom._render_projectfile(**to_render)

    assert rendered == compare

def test_starting_head(basic_dis, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())
    
    ibound, _, _ = basic_dis
    starting_head = StartingHead(starting_head=ibound)
    nlayer = len(starting_head["layer"])
    
    to_render = get_render_dict(starting_head, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (shd), 1, StartingHead, ['starting_head']\n"
        "001, 003\n"
        f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}starting_head_l1.idf\n"
        f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}starting_head_l2.idf\n"
        f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}starting_head_l3.idf"
    )

    rendered = starting_head._render_projectfile(**to_render)

    assert rendered == compare
