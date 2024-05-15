from functools import wraps
from time import time

from imod.logging.loglevel import LogLevel


def standard_log_decorator(
    start_level: LogLevel = LogLevel.INFO, end_level: LogLevel = LogLevel.DEBUG
):
    """
    Decorator to print log messages announcing the beginning and end of the decorated method
    """

    def decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            from imod.logging import logger

            # anounce start of function
            object_name = str(type(args[0]).__name__)
            start_message = f"Beginning execution of {fun.__module__}.{fun.__name__} for object {object_name}..."

            # anounce start of function
            start_time = time()
            logger.log(loglevel=start_level, message=start_message, additional_depth=2)

            # run function
            return_value = fun(*args, **kwargs)
            end_time = time()

            # anounce end of function
            end_message = f"Finished execution of {fun.__module__}.{fun.__name__}  for object {object_name} in {end_time - start_time} seconds..."
            logger.log(loglevel=end_level, message=end_message, additional_depth=2)
            return return_value

        return wrapper

    return decorator


def init_log_decorator(
    start_level: LogLevel = LogLevel.INFO, end_level: LogLevel = LogLevel.DEBUG
):
    """
    Decorator to print log messages announcing the beginning and end of initialization methods
    """

    def decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            from imod.logging import logger

            # anounce start of function
            object_name = str(type(args[0]).__name__)
            start_message = f"Initializing the {object_name} package..."
            start_time = time()
            logger.log(loglevel=start_level, message=start_message, additional_depth=2)

            # run function
            return_value = fun(*args, **kwargs)
            end_time = time()

            # anounce end of function
            end_message = f"Successfully initialized the {object_name} in {end_time - start_time} seconds..."
            logger.log(loglevel=end_level, message=end_message, additional_depth=2)
            return return_value

        return wrapper

    return decorator
