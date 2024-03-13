from imod.logging import logger, log_with_level
from imod.logging.loglevel import LogLevel


# decorator to calculate duration
# taken by any function.
def log_decorator(start_level, start_message ):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            #end_level =  kwargs["end_level"]      
            #end_message =  kwargs["end_message"]

            log_with_level(logger, loglevel=start_level,message=start_message)
            fun(*args, **kwargs)
        return wrapper
        #logger.log_with_level(end_level,end_message)
    return  decorator
       
    
   