import pytest

import imod


def roundtrip(simulation, tmp_path):
    simulation.dump(tmp_path)
    back = imod.mf6.Modflow6Simulation.from_file(tmp_path / f"{simulation.name}.toml")
    assert isinstance(back, imod.mf6.Modflow6Simulation)


@pytest.mark.usefixtures("circle_model")
def test_circle_roundtrip(circle_model, tmp_path):
    roundtrip(circle_model, tmp_path)
