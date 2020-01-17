import imod

# these examples are not meant for testing purposes
# however to keep them up to date, we check if they don't crash


def test_elder(tmp_path):
    with imod.util.cd(tmp_path):
        from examples import Elder


def test_freshwaterlens(tmp_path):
    with imod.util.cd(tmp_path):
        from examples import FreshwaterLens


def test_henrycase(tmp_path):
    with imod.util.cd(tmp_path):
        from examples import HenryCase


def test_hydrocoin(tmp_path):
    with imod.util.cd(tmp_path):
        from examples import Hydrocoin


def test_mf6_ex01_twri(tmp_path):
    with imod.util.cd(tmp_path):
        from examples import mf6_ex01_twri


def test_saltwaterpocket(tmp_path):
    with imod.util.cd(tmp_path):
        from examples import SaltwaterPocket


def test_verticalinterface(tmp_path):
    with imod.util.cd(tmp_path):
        from examples import VerticalInterface
