from abc import abstractmethod


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
