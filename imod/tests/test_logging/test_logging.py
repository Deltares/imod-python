from unittest.mock import patch

import pytest

import imod.logging
from imod.logging import LoggerType, LogLevel
from imod.logging.logurulogger import LoguruLogger
from imod.logging.nulllogger import NullLogger
from imod.logging.pythonlogger import PythonLogger


def test_logging_no_configuration():
    # Arrange.
    logger = imod.logging.logger

    # Assert.
    assert isinstance(logger.instance, NullLogger)


@pytest.mark.parametrize(
    ("logger_type", "logger_class"),
    [
        (LoggerType.NULL, imod.logging.config.NullLogger),
        (LoggerType.PYTHON, imod.logging.config.PythonLogger),
        (LoggerType.LOGURU, imod.logging.config.LoguruLogger),
    ],
)
def test_logging_configure_logger(logger_type, logger_class):
    # Arrange.
    imod.logging.configure(logger_type)
    logger = imod.logging.logger

    # Assert.
    assert isinstance(logger.instance, logger_class)


def test_logging_change_logger_during_runtime():
    def test_method(logger=imod.logging.logger):
        assert isinstance(logger.instance, LoguruLogger)

    # Arrange.
    imod.logging.configure(LoggerType.PYTHON)
    assert isinstance(imod.logging.logger.instance, PythonLogger)

    # Act.
    imod.logging.configure(LoggerType.LOGURU)

    # Assert
    test_method()
    assert isinstance(imod.logging.logger.instance, LoguruLogger)


@pytest.mark.parametrize(
    ("logger_type", "patched_logger"),
    [
        (LoggerType.NULL, "imod.logging.config.NullLogger"),
        (LoggerType.PYTHON, "imod.logging.config.PythonLogger"),
        (LoggerType.LOGURU, "imod.logging.config.LoguruLogger"),
    ],
)
def test_logging_calls_forwarded_to_loggers(logger_type, patched_logger):
    # Arrange.
    with patch(patched_logger) as MockClass:
        # Arrange.
        imod.logging.configure(logger_type)
        logger = imod.logging.logger

        # Act.
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

        # Assert.
        logger_instance = MockClass.return_value
        logger_instance.debug.assert_called_with("debug message", 0)
        logger_instance.info.assert_called_with("info message", 0)
        logger_instance.warning.assert_called_with("warning message", 0)
        logger_instance.error.assert_called_with("error message", 0)
        logger_instance.critical.assert_called_with("critical message", 0)


@pytest.mark.parametrize(
    "logger_level",
    [
        LogLevel.DEBUG,
        LogLevel.INFO,
        LogLevel.WARNING,
        LogLevel.ERROR,
        LogLevel.CRITICAL,
    ],
)
@pytest.mark.parametrize("add_default_stream_handler", [True, False])
@pytest.mark.parametrize("add_default_file_handler", [True, False])
@pytest.mark.parametrize(
    ("logger_type", "patched_logger"),
    [
        (LoggerType.PYTHON, "imod.logging.config.PythonLogger"),
        (LoggerType.LOGURU, "imod.logging.config.LoguruLogger"),
    ],
)
def test_logging_configure_param_forwarded_to_loggers(
    logger_type,
    patched_logger,
    logger_level,
    add_default_stream_handler,
    add_default_file_handler,
):
    # Arrange.
    with patch(patched_logger) as MockClass:
        # Arrange/ Act.
        imod.logging.configure(
            logger_type,
            logger_level,
            add_default_stream_handler,
            add_default_file_handler,
        )

        # Assert.
        MockClass.assert_called_with(
            logger_level, add_default_stream_handler, add_default_file_handler
        )
