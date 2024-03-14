from imod.logging import logger, ILogger
from imod.logging.loglevel import LogLevel


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

# decorator to calculate duration
# taken by any function.
def log_decorator(start_level: LogLevel = LogLevel.DEBUG, end_level: LogLevel = LogLevel.DEBUG):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            object_type = str(type(args[0]))
            start_message = f"beginning execution of {fun.__module__}.{fun.__name__} for object {object_type}"
            end_message = f"finished execution of {fun.__module__}.{fun.__name__}  for object {object_type}"

            log_with_level(logger, loglevel=start_level,message=start_message)
            fun(*args, **kwargs)
            log_with_level(logger, loglevel=end_level,message=end_message)

        return wrapper
    return  decorator
       
    
   