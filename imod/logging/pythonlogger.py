import logging
import sys

from imod.logging.ilogger import ILogger
from imod.logging.loglevel import LogLevel


def _formatter():
    return logging.Formatter(
        "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s"
    )


def _stack_level(additional_depth: int) -> int:
    """
    The stack level is used to print the file and line number of the line being logged.
    Because there are a few layers between the place where the imod logger is used and
    the place where the pyton logger is used we need to set a custom stack level.

    An additional_depth can be provided to add to this default stack level.
    This is useful when a decorator is added which introduces an additional level between
    the imod logger and the python logger.
    """

    default_stack_level = 3
    return default_stack_level + additional_depth


class PythonLogger(ILogger):
    """
    The :class:`PythonLogger` is used to log messages using the default python logging framework.
    """

    def __init__(
        self,
        log_level: LogLevel,
        add_default_stream_handler: bool,
        add_default_file_handler: bool,
    ) -> None:
        self.logger = logging.getLogger("imod")
        self._set_level(log_level)

        if add_default_stream_handler:
            self._add_stream_handler()
        if add_default_file_handler:
            self._add_file_handler()

    def debug(self, message: str, additional_depth: int = 0) -> None:
        self.logger.debug(message, stacklevel=_stack_level(additional_depth))

    def info(self, message: str, additional_depth: int = 0) -> None:
        self.logger.info(message, stacklevel=_stack_level(additional_depth))

    def warning(self, message: str, additional_depth: int = 0) -> None:
        self.logger.warning(message, stacklevel=_stack_level(additional_depth))

    def error(self, message: str, additional_depth: int = 0) -> None:
        self.logger.error(message, stacklevel=_stack_level(additional_depth))

    def critical(self, message: str, additional_depth: int = 0) -> None:
        self.logger.critical(message, stacklevel=_stack_level(additional_depth))

    def _set_level(self, log_level: LogLevel) -> None:
        self.logger.setLevel(log_level.value)

    def _add_stream_handler(self) -> None:
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setFormatter(_formatter())

        self.logger.addHandler(stdout)

    def _add_file_handler(self) -> None:
        file = logging.FileHandler("imod-python.log")
        file.setFormatter(_formatter())

        self.logger.addHandler(file)
