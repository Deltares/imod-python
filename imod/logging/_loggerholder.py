from imod.logging.ilogger import ILogger
from imod.logging.nulllogger import NullLogger


class _LoggerHolder(ILogger):
    """
    The :class:`_LoggerHolder` is wrapper that allows us to change the logger during runtime.

    This makes it less critical at which stage the logger is being configured. For instance, without
    this wrapper, when the logger is being imported during initialization time it will get
    the default NullLogger object. If after the initialization time the user
    calls the :func:`imod.logging.configure` method and configures a
    :class:`~imod.logging.config.LoggerType`, the logger retrieved during initialization
     time won't be updated.

     >>> import imod
     >>> from imod.logging import LoggerType
     >>>
     >>> def foo(logger: ILogger = imod.logging.logger)
     >>>    pass
     >>>
     >>> imod.logging.configure(LoggerType.LOGURU)
     >>>
     >>> # Event hough we've configured the logger to use Loguru, foo is initialized with the default
     >>> # NullLogger. By using the LogerHolder this issue is solved and we can configure the logger
     >>> # whenever we want.
     >>> foo()

    This wrapper solves this issue. During initialization time the Holder is returned,
    which holds an instance of the actual logger being used. When the user calls the configure
    method the instance within this holder is updated. All the calls are forwarded to that instance.
    For the user it seems that they are directly using the logger the configured, and they are not
    aware they are making use of an intermediary object.
    """

    def __init__(self) -> None:
        self._instance = NullLogger()

    @property
    def instance(self) -> ILogger:
        """
        Contains the actual ILogger object
        """
        return self._instance

    @instance.setter
    def instance(self, value: ILogger) -> None:
        self._instance = value

    def debug(self, message: str) -> None:
        self.instance.debug(message)

    def info(self, message: str) -> None:
        self.instance.info(message)

    def warning(self, message: str) -> None:
        self.instance.warning(message)

    def error(self, message: str) -> None:
        self.instance.error(message)

    def critical(self, message: str) -> None:
        self.instance.critical(message)
