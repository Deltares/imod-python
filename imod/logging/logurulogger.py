import sys

from loguru import logger

from imod.logging.ilogger import ILogger
from imod.logging.loglevel import LogLevel


class LoguruLogger(ILogger):
    """
    The :class:`LoguruLogger` is used to log messages using the loguru logging framework.
    """

    def __init__(
        self,
        log_level: LogLevel,
        add_default_stream_handler: bool,
        add_default_file_handler: bool,
    ) -> None:
        # Remove default handler set by loguru
        logger.remove()

        if add_default_stream_handler:
            logger.add(sys.stdout, level=log_level.value)
        if add_default_file_handler:
            logger.add("imod-python.log", level=log_level.value)

    def debug(self, message: str) -> None:
        logger.opt(depth=2).debug(message)

    def info(self, message: str) -> None:
        logger.opt(depth=2).info(message)

    def warning(self, message: str) -> None:
        logger.opt(depth=2).warning(message)

    def error(self, message: str) -> None:
        logger.opt(depth=2).error(message)

    def critical(self, message: str) -> None:
        logger.opt(depth=2).critical(message)
