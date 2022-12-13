# -*- coding: utf-8 -*-
"""
@author: dorge
"""


# =============================================================================
# Logger file 
# =============================================================================
#     filname:        logger.py
#     version:        1.0
#     author:         dorge
#     creation_date:  11-12-2022
#     
#     change history:
#     
#     who       when      version     changes
#     ----      -----     -------     --------
#     dorge     11-12-22   1.0         INIT
#
#   description: this class used to generate logs in logs files.
# =============================================================================

import logging

class Logger:
    def __init__(self, run_id, module, mode):
        """
        Generate logs.
        Parameters
        ----------
        run_id : string
            unique process ID.
        module : string
            name of module which sends a log.
        mode : string
            'detecting' or 'analyzing'.
        Returns
        -------
        None.
        """
        
        # get relevent logger instant and set the level to debug.
        self.logger=logging.getLogger(str(module)+'_' + str(run_id))
        self.logger.setLevel(logging.DEBUG)
        
        # check if it a log of training or predicting procedure.
        if mode=='detecting':
            file_handler = logging.FileHandler('C:/Users/dorge/neolithics/logs/detecting/' + str(run_id) + '.log')
        
        else:  # mode=='analyzing':
            file_handler = logging.FileHandler('C:/Users/dorge/neolithics/logs/analyzing/' + str(run_id) + '.log')
        
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self,message):
        # write information log
        self.logger.info(message)
    
    def exception(self,message):
        # write exception log
        self.logger.exception(message)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    