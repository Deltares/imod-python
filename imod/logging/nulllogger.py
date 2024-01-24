from imod.logging.ilogger import ILogger


class NullLogger(ILogger):
    """
    The :class:`NullLogger` is used as a dummy logger that doesn't log anything.
    """

    def __init__(self) -> None:
        pass

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def critical(self, message: str) -> None:
        pass
