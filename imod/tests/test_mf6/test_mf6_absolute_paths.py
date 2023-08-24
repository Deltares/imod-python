import pytest


@pytest.mark.usefixtures("circle_model")
def test_simulation_writes_full_paths_if_requested(circle_model, tmp_path):
    simulation = circle_model
    sim_dir = tmp_path / "circle"

    simulation.write(sim_dir, binary=False, use_absolute_paths=True)

    simdir_separator_count = str(sim_dir.as_posix()).count("/")
    modeldir_separator_count = simdir_separator_count + 2
    pkgdir_separator_count = simdir_separator_count + 3
    toplevel_files = {"mfsim.nam": modeldir_separator_count}
    gwf1_files = {
        "chd.chd": pkgdir_separator_count + 1,
        "disv.disv": 2 * pkgdir_separator_count + 2,
        "GWF_1.nam": 7 * modeldir_separator_count,
        "ic.ic": 0,
        "npf.npf": 3 * pkgdir_separator_count + 3,
        "oc.oc": 2 * modeldir_separator_count,
        "rch.rch": pkgdir_separator_count + 1,
        "sto.sto": 0,
    }

    for fname, expected_separator_count in toplevel_files.items():
        with open(sim_dir / fname, "r") as f:
            content = f.read()
            assert content.count("/") == expected_separator_count

    for fname, expected_separator_count in gwf1_files.items():
        with open(sim_dir / "GWF_1" / fname, "r") as f:
            content = f.read()
            assert content.count("/") == expected_separator_count
