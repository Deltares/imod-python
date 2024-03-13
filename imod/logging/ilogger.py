from abc import abstractmethod

from imod.logging.loglevel import LogLevel

class ILogger:
    """
    Interface to be implemented by all logger wrappers.
    """

    @abstractmethod
    def debug(self, message: str) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.DEBUG`'.

        Parameters
        ----------
        message : str
            message to be logged
        """
        raise NotImplementedError

    @abstractmethod
    def info(self, message: str) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.INFO`'.

        Parameters
        ----------
        message : str
            message to be logged
        """
        raise NotImplementedError

    @abstractmethod
    def warning(self, message: str) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.WARNING`'.

        Parameters
        ----------
        message : str
            message to be logged
        """
        raise NotImplementedError

    @abstractmethod
    def error(self, message: str) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.ERROR`'.

        Parameters
        ----------
        message : str
            message to be logged
        """
        raise NotImplementedError

    @abstractmethod
    def critical(self, message: str) -> None:
        """
        Log message with severity ':attr:`~imod.logging.loglevel.LogLevel.CRITICAL`'.

        Parameters
        ----------
        message : str
            message to be logged
        """
        raise NotImplementedError

def log_with_level(logger: ILogger,loglevel: LogLevel, message: str) -> None:
    if loglevel == LogLevel.DEBUG:
        logger.debug(message)
    elif  loglevel == LogLevel.INFO:
        logger.info(message)
    elif  loglevel == LogLevel.WARNING:
        logger.warning(message)   
    elif loglevel == LogLevel.ERROR:
        logger.error(message)   
    elif loglevel == LogLevel.CRITICAL:      
        logger.critical(message)   
    else:
        raise ValueError(f"Unknown logging urgency at level {loglevel}")
