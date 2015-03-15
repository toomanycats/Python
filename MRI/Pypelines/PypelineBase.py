import LoggingTools
import logging

class PypelineBase(object):
    def __init__(self, log_path, log_level='info'):
        if logging.getLogger(__name__) is not None:
            self.log = logging.getLogger(__name__)  
            return
        
        self.log = LoggingTools.SetupLogger(log_path, log_level).get_logger()
        
        

#TODO: consider adding imagetools instance, so that all pypelines have self.imagetools