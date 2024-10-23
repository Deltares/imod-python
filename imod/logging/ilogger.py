from abc import abstractmethod

from imod.logging.loglevel import LogLevel


class ILogger:
    """
    Interface to be implemented by all logger wrappers.
    """

    @abstractmethod
    def debug(self, message: str, additional_depth: int = 0) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.DEBUG`'.

        Parameters
        ----------
        message : str
            message to be logged
        additional_depth: Optional[int]
            additional depth level. Use this to correct the filename and line number
            when you add logging to a decorator
        """
        raise NotImplementedError

    @abstractmethod
    def info(self, message: str, additional_depth: int = 0) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.INFO`'.

        Parameters
        ----------
        message : str
            message to be logged
        additional_depth: Optional[int]
            additional depth level. Use this to correct the filename and line number
            when you add logging to a decorator
        """
        raise NotImplementedError

    @abstractmethod
    def warning(self, message: str, additional_depth: int = 0) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.WARNING`'.

        Parameters
        ----------
        message : str
            message to be logged
        additional_depth: Optional[int]
            additional depth level. Use this to correct the filename and line number
            when you add logging to a decorator
        """
        raise NotImplementedError

    @abstractmethod
    def error(self, message: str, additional_depth: int = 0) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.ERROR`'.

        Parameters
        ----------
        message : str
            message to be logged
        additional_depth: Optional[int]
            additional depth level. Use this to correct the filename and line number
            when you add logging to a decorator
        """
        raise NotImplementedError

    @abstractmethod
    def critical(self, message: str, additional_depth: int = 0) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.CRITICAL`'.

        Parameters
        ----------
        message : str
            message to be logged
        additional_depth: Optional[int]
            additional depth level. Use this to correct the filename and line number
            when you add logging to a decorator
        """
        raise NotImplementedError

    def log(self, loglevel: LogLevel, message: str, additional_depth: int = 0) -> None:
        """
        logs a message with the specified urgency level.
        """
        match loglevel:
            case LogLevel.DEBUG:
                self.debug(message, additional_depth)
            case LogLevel.INFO:
                self.info(message, additional_depth)
            case LogLevel.WARNING:
                self.warning(message, additional_depth)
            case LogLevel.ERROR:
                self.error(message, additional_depth)
            case LogLevel.CRITICAL:
                self.critical(message, additional_depth)
            case _:
                raise ValueError(f"Unknown logging urgency at level {loglevel}")
