import textwrap
from pathlib import Path

from imod.wq import HorizontalFlowBarrier


def test_render():
    hfb = HorizontalFlowBarrier(hfbfile=None)

    compare = textwrap.dedent(
        """\
        [hfb6]
            hfbfile = hfb/test.hfb
        """
    )

    assert hfb._render(modelname="test", directory=Path("./hfb")) == compare
