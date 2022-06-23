import pytest

from imod.mf6 import read_input as ri


def test_parse_model():
    ri.parse_model("gwf6 GWF_1.nam GWF_1", "sim.nam") == ["gwf6", "GWF_1_nam", "GWF_1"]
    with pytest.raises(ValueError, match="ftype, fname and pname expected"):
        ri.parse_model("gwf6 GWF_1.nam", "sim.nam")


def test_parse_exchange():
    ri.parse_exchange(
        "GWF6-GWF6 simulation.exg GWF_Model_1 GWF_Model_2", "sim.nam"
    ) == ["GWF6-GWF6, simulation.exg, GWF_Model_1, GWF_Model_2"]
    with pytest.raises(ValueError, match="exgtype, exgfile, exgmnamea, exgmnameb"):
        ri.parse_exchange("GWF6-GWF6 simulation.exg GWF_Model_1", "sim.nam")


def test_parse_solutiongroup():
    ri.parse_solutiongroup("ims6 solver.ims GWF_1", "sim.nam") == [
        "ims6",
        "solver.ims",
        "GWF_1",
    ]
    with pytest.raises(ValueError, match="Expected at least three entries"):
        ri.parse_solutiongroup("ims6 solver.ims", "sim.name")


def test_parse_package():
    ri.parse_package("dis6 dis.dis", "gwf.nam") == ["dis6", "dis.dis", "dis"]
    ri.parse_package("dis6 dis.dis discretization", "gwf.nam") == [
        "dis6",
        "dis.dis",
        "discretization",
    ]
    with pytest.raises(ValueError, match="Expected ftype, fname"):
        ri.parse_package("dis6", "gwf.nam")
    with pytest.raises(ValueError, match="Expected ftype, fname"):
        ri.parse_package("dis6 dis.dis abc def", "gwf.nam")


def test_parse_tdis_perioddata():
    perlen, nstp, tsmult = ri.parse_tdis_perioddata("1.0 1 1.0", "timedis.tdis")
    assert isinstance(perlen, float)
    assert perlen == 1.0
    assert isinstance(nstp, int)
    assert nstp == 1
    assert isinstance(tsmult, float)
    assert tsmult == 1.0
    with pytest.raises(ValueError, match="perlen, nstp, tsmult expected"):
        ri.parse_tdis_perioddata("1.0 1.0", "timedis.tdis")
