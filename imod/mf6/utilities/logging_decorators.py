from imod.logging import logger, log_with_level
from imod.logging.loglevel import LogLevel


# decorator to calculate duration
# taken by any function.
def log_decorator(function_name,start_level,end_level ):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            start_message = f"beginning execution of {function_name}"
            end_message = f"finished execution of {function_name}"

            log_with_level(logger, loglevel=start_level,message=start_message)
            fun(*args, **kwargs)
            log_with_level(logger, loglevel=end_level,message=end_message)
                  
        return wrapper
        #logger.log_with_level(end_level,end_message)
    return  decorator
       
    
   