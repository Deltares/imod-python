from textwrap import dedent
from typing import Any


class MetaMod:
    """
    This class has been moved to ``primod``,
    `See example here <https://deltares.github.io/iMOD-Documentation/coupler_metamod_example.html>`_.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        message = dedent(
            """\
            This class has been moved to the ``primod`` package.
            For an example, see: 
            https://deltares.github.io/iMOD-Documentation/coupler_metamod_example.html
            """
        )
        raise NotImplementedError(message)
