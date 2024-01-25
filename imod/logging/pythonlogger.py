import logging
import sys

from imod.logging.ilogger import ILogger
from imod.logging.loglevel import LogLevel


def _formatter():
    return logging.Formatter(
        "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s"
    )


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

    def debug(self, message: str) -> None:
        self.logger.debug(message, stacklevel=3)

    def info(self, message: str) -> None:
        self.logger.info(message, stacklevel=3)

    def warning(self, message: str) -> None:
        self.logger.warning(message, stacklevel=3)

    def error(self, message: str) -> None:
        self.logger.error(message, stacklevel=3)

    def critical(self, message: str) -> None:
        self.logger.critical(message, stacklevel=3)

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
