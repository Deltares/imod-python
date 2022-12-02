import textwrap
from pathlib import Path

from imod.wq import HorizontalAnisotropy, HorizontalAnisotropyFile


def test_render_ani():
    ani = HorizontalAnisotropy(factor=1.0, angle=0.0)

    compare = textwrap.dedent(
        """\
        [ani]
            anifile = ani/test.ani
        """
    )

    assert ani._render(modelname="test", directory=Path("./ani"), nlayer=1) == compare


def test_render_anifile():
    ani = HorizontalAnisotropyFile(anifile="test.test")

    compare = textwrap.dedent(
        """\
        [ani]
            anifile = ani/test.ani
        """
    )

    assert ani._render(modelname="test", directory=Path("./ani"), nlayer=1) == compare
