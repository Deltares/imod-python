from enum import Enum

import imod

from .loglevel import LogLevel
from .logurulogger import LoguruLogger
from .nulllogger import NullLogger
from .pythonlogger import PythonLogger


class LoggerType(Enum):
    """
    The available logging frameworks.
    """

    PYTHON = PythonLogger.__name__
    """
    The default python logging framework.
    """
    LOGURU = LoguruLogger.__name__
    """
    The loguru logging framework.
    """
    NULL = NullLogger.__name__
    """
    A dummy logger that doesn't log anything.
    """


def configure(
    logger_type: LoggerType,
    log_level: LogLevel = LogLevel.WARNING,
    add_default_stream_handler: bool = True,
    add_default_file_handler: bool = False,
) -> None:
    """
    Setup the logging framework and assign it a log level.
    To add a default stream- and/or file-handler you can use the
    ``add_default_stream_handler`` or ``add_default_file_handler`` flags. If a
    default file-handler is added then the log output will be written to
    the `imod-python.log` file

    Parameters
    ----------
    logger_type : LoggerType
        The logging framework to be used.
    log_level : LogLevel
        The log level to be set.
    add_default_stream_handler : bool
        A flag that specifies if a default stream-handler should be added.
        True by default.
    add_default_file_handler : bool
        A flag that specifies if a default filehandler should be added.
        The log will be written to `imod-python.log`. False by default.
    """
    match logger_type:
        case LoggerType.PYTHON:
            imod.logging.logger.instance = PythonLogger(
                log_level, add_default_stream_handler, add_default_file_handler
            )
        case LoggerType.LOGURU:
            imod.logging.logger.instance = LoguruLogger(
                log_level, add_default_stream_handler, add_default_file_handler
            )
        case _:
            imod.logging.logger.instance = NullLogger()
