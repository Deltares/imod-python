from imod.logging.loglevel import LogLevel


def standard_log_decorator(
    start_level: LogLevel = LogLevel.INFO, end_level: LogLevel = LogLevel.DEBUG
):
    """
    Decorator to print log messages announcing the beginning and end of the decorated method
    """

    def decorator(fun):
        def wrapper(*args, **kwargs):
            from imod.logging import logger

            object_name = str(type(args[0]).__name__)
            start_message = f"Beginning execution of {fun.__module__}.{fun.__name__} for object {object_name}..."
            end_message = f"Finished execution of {fun.__module__}.{fun.__name__}  for object {object_name}..."

            logger.log(loglevel=start_level, message=start_message, additional_depth=2)
            return_value = fun(*args, **kwargs)
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
        def wrapper(*args, **kwargs):
            from imod.logging import logger

            object_name = str(type(args[0]).__name__)
            start_message = f"Initializing the {object_name} package..."
            end_message = f"Successfully initialized the {object_name}..."

            logger.log(loglevel=start_level, message=start_message, additional_depth=2)
            return_value = fun(*args, **kwargs)
            logger.log(loglevel=end_level, message=end_message, additional_depth=2)
            return return_value

        return wrapper

    return decorator
