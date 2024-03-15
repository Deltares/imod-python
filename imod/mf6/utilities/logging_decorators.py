from imod.logging import ILogger, logger
from imod.logging.loglevel import LogLevel


def log_with_level(logger: ILogger,loglevel: LogLevel, message: str) -> None:
    '''
    logs a message with the specified urgency level.
    '''
    match loglevel:
        case LogLevel.DEBUG:
            logger.debug(message)
        case LogLevel.INFO:
            logger.info(message)
        case LogLevel.WARNING:
            logger.warning(message)   
        case LogLevel.ERROR:
            logger.error(message)   
        case LogLevel.CRITICAL:      
            logger.critical(message)   
        case _:
            raise ValueError(f"Unknown logging urgency at level {loglevel}")

# decorator to print log messages announcing the begin and end of the decorated method
def standard_log_decorator(start_level: LogLevel = LogLevel.INFO, end_level: LogLevel = LogLevel.DEBUG):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            object_type = str(type(args[0]))
            start_message = f"beginning execution of {fun.__module__}.{fun.__name__} for object {object_type}..."
            end_message = f"finished execution of {fun.__module__}.{fun.__name__}  for object {object_type}..."

            log_with_level(logger, loglevel=start_level,message=start_message)
            return_value = fun(*args, **kwargs)
            log_with_level(logger, loglevel=end_level,message=end_message)
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

            log_with_level(logger, loglevel=start_level,message=start_message)
            return_value = fun(*args, **kwargs)
            log_with_level(logger, loglevel=end_level,message=end_message)
            return return_value
        return wrapper
    return  decorator
         