import sys
from typing import Optional

from loguru import logger

from imod.logging.ilogger import ILogger
from imod.logging.loglevel import LogLevel


def _depth_level(additional_depth: Optional[int]) -> int:
    """
    The depth level is used to print the file and line number of the line being logged.
    Because there are a few layers between the place where the imod logger is used and
    the place where the pyton logger is used we need to set a custom stack level.

    An additional_depth can be provided to add to this default depth level.
    This is useful when a decorator is added which introduces an additional level between
    the imod logger and the loguru logger.
    """

    default_stack_level = 2
    if additional_depth is not None:
        return default_stack_level + additional_depth
    else:
        return default_stack_level


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

    def debug(self, message: str, additional_depth: Optional[int] = None) -> None:
        logger.opt(depth=_depth_level(additional_depth)).debug(message)

    def info(self, message: str, additional_depth: Optional[int] = None) -> None:
        logger.opt(depth=_depth_level(additional_depth)).info(message)

    def warning(self, message: str, additional_depth: Optional[int] = None) -> None:
        logger.opt(depth=_depth_level(additional_depth)).warning(message)

    def error(self, message: str, additional_depth: Optional[int] = None) -> None:
        logger.opt(depth=_depth_level(additional_depth)).error(message)

    def critical(self, message: str, additional_depth: Optional[int] = None) -> None:
        logger.opt(depth=_depth_level(additional_depth)).critical(message)
