"""
Package used for providing logging support to imod-pyton.

Examples
--------

If you want to directly use the logger in your project and make use of the default handlers

>>> #Setup imod python logging using the python logging framework
>>> import imod
>>> from imod.logging import LoggerType
>>>
>>> imod.logging.configure(LoggerType.LOGURU)

>>> # Setup imod python logging using the python logging framework and write the log output to a file
>>> import imod
>>> from imod.logging import LoggerType
>>>
>>> imod.logging.configure(LoggerType.PYTHON, add_default_file_handler=True)

If you want to integrate imod-python logging into your own logging framework

>>> # Setup imod python logging integration into an existing python logger
>>> import imod
>>> from imod.logging import LoggerType, LogLevel
>>> import logging
>>>
>>> imod.logging.configure(LoggerType.PYTHON, LogLevel.INFO, add_default_stream_handler=False, add_default_file_handler=False)
>>>
>>> logging.basicConfig(
>>>       level=logging.INFO,
>>>       format="%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s",
>>>       handlers=[
>>>           logging.StreamHandler(),
>>>           logging.FileHandler("imod-python.log")
>>>      ]
>>> )
>>>
>>> logger = logging.getLogger()
>>> logger.info('info message')

>>> # Setup imod python logging integration into an existing loguru logger
>>> import sys
>>> import imod
>>> from imod.logging import LoggerType, LogLevel
>>> from loguru import logger
>>>
>>> imod.logging.configure(LoggerType.LOGURU, LogLevel.INFO, add_default_stream_handler=False, add_default_file_handler=False)
>>>
>>> config = {
>>>     "handlers": [
>>>        {"sink": sys.stdout},
>>>        {"sink": "imod-python.log"},
>>>     ],
>>> }
>>> logger.configure(**config)
>>> logger.info('info message')

"""

from imod.logging._loggerholder import _LoggerHolder
from imod.logging.config import LoggerType, configure
from imod.logging.ilogger import ILogger  # noqa: I001
from imod.logging.loglevel import LogLevel

logger = _LoggerHolder()
