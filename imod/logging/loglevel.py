from enum import Enum


class LogLevel(Enum):
    """
    The available log levels for the logger.
    """

    DEBUG = 10
    """
    A log level used for events considered to be useful during software 
    debugging when more granular information is needed.
    """
    INFO = 20
    """
    An event happened, the event is purely informative and can be ignored 
    during normal operations.
    """
    WARNING = 30
    """
    Unexpected behavior happened inside the application, but it is continuing
    its work and the key business features are operating as expected.
    """
    ERROR = 40
    """
    One or more functionalities are not working, preventing some functionalities 
    from working correctly.
    """
    CRITICAL = 50
    """
    One or more key business functionalities are not working and the whole 
    system doesn’t fulfill the business functionalities.
    """
