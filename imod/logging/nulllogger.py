from typing import Optional

from imod.logging.ilogger import ILogger


class NullLogger(ILogger):
    """
    The :class:`NullLogger` is used as a dummy logger that doesn't log anything.
    """

    def __init__(self) -> None:
        pass

    def debug(self, message: str, additional_depth: Optional[int] = None) -> None:
        pass

    def info(self, message: str, additional_depth: Optional[int] = None) -> None:
        pass

    def warning(self, message: str, additional_depth: Optional[int] = None) -> None:
        pass

    def error(self, message: str, additional_depth: Optional[int] = None) -> None:
        pass

    def critical(self, message: str, additional_depth: Optional[int] = None) -> None:
        pass
