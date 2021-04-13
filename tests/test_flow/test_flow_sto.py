from imod.flow import StorageCoefficient, SpecificStorage
import pathlib


def test_storage_coefficient(basic_dis, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    ibound, _, _ = basic_dis
    storage = StorageCoefficient(storage_coefficient=0.2)
    nlayer = len(ibound["layer"])

    to_render = get_render_dict(storage, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (sto), 1, StorageCoefficient, ['storage_coefficient']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, 0.2, ""\n'
        '1, 1, 002, 1.000, 0.000, 0.2, ""\n'
        '1, 1, 003, 1.000, 0.000, 0.2, ""'
    )

    rendered = storage._render_projectfile(**to_render)

    assert rendered == compare


def test_specific_storage(basic_dis, get_render_dict):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    ibound, _, _ = basic_dis
    storage = SpecificStorage(specific_storage=1e-5)
    nlayer = len(ibound["layer"])

    to_render = get_render_dict(storage, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = (
        "0001, (ssc), 1, SpecificStorage, ['specific_storage']\n"
        "001, 003\n"
        '1, 1, 001, 1.000, 0.000, 1e-05, ""\n'
        '1, 1, 002, 1.000, 0.000, 1e-05, ""\n'
        '1, 1, 003, 1.000, 0.000, 1e-05, ""'
    )

    rendered = storage._render_projectfile(**to_render)

    assert rendered == compare
