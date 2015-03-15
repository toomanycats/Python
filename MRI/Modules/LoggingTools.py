'''
Created on Dec 12, 2013

@author: dpc
'''

import logging

class SetupLogger(object):
    '''
    Collection of method to setup the logger. 
    '''


    def __init__(self, log_path, level='debug'):
        self.log_path = log_path
        log_level_dict = {'debug':10, 'info':20, 'error':40} 
        self.level = log_level_dict[level]
        self.template = "\n%(levelname)s:%(message)s" 

    def get_basic_logger(self):
        logging.basicConfig(filename=self.log_path, level=self.level, name='simple')        
        log = logging
        
        return log
    
    def get_module_logger(self):
        log = logging.getLogger('module_logger')
        log.setLevel(self.level)
        file_handler = logging.FileHandler(self.log_path)

        formatter = logging.Formatter(self.template)        
        file_handler.setFormatter(formatter)
        
        log.addHandler(file_handler)
        
        return log