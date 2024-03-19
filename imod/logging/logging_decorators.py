from imod.logging import logger
from imod.logging.loglevel import LogLevel


# decorator to print log messages announcing the begin and end of the decorated method
def standard_log_decorator(start_level: LogLevel = LogLevel.INFO, end_level: LogLevel = LogLevel.DEBUG):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            object_name = str(type(args[0]).__name__)
            start_message = f"beginning execution of {fun.__module__}.{fun.__name__} for object {object_type}..."
            end_message = f"finished execution of {fun.__module__}.{fun.__name__}  for object {object_type}..."

            logger.log(loglevel=start_level, message=start_message)
            return_value = fun(*args, **kwargs)
            logger.log(loglevel=end_level, message=end_message)
            return return_value
        return wrapper
    return  decorator
       
# decorator to print log messages announcing the begin and end of initialization methods
def init_log_decorator(start_level: LogLevel = LogLevel.INFO, end_level: LogLevel = LogLevel.DEBUG):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            object_type = str(type(args[0]))
            start_message = f"Initializing the {object_type} package..."
            end_message = f"Succesfully initialized the {object_type}..."

            logger.log( loglevel=start_level,message=start_message)
            return_value = fun(*args, **kwargs)
            logger.log(loglevel=end_level,message=end_message)
            return return_value
        return wrapper
    return  decorator
         