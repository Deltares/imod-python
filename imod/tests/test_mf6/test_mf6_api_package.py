import textwrap
from pathlib import Path

import imod


def test_render(tmp_path: Path):
    api = imod.mf6.ApiPackage(
        maxbound=33, print_input=True, print_flows=True, save_flows=True
    )
    actual = api._render(tmp_path, "api", [], True)

    expected = textwrap.dedent(
        """\
        begin options
          print_input
          print_flows
          save_flows
        end options

        begin dimensions
          maxbound 33
        end dimensions"""
    )
    assert actual == expected
