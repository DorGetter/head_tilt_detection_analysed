# -*- coding: utf-8 -*-
"""
@author: dorge
"""
# =============================================================================
# This script will hold the configuration class
# =============================================================================
#     filname:        config.py
#     version:        1.0
#     author:         dorge
#     creation_date:  11-12-2022
#     
#     change history:
#     
#     who       when      version     changes
#     ----      -----     -------     --------
#     dorge      11-12-22   1.0         INIT
# =============================================================================
    
import os
import cv2
import random
from datetime import datetime
import core.file_operation as file_op
from pathlib import Path

Root = Path(__file__).resolve().parents[1]

class Config:
    
    def __init__(self, mode):
        self.mode=mode
        self.debug_mode=False
        self.saved_txt_path=''
        # detecting:
        if mode=='detecting':
            self.camera_source=0
            self.eyes_brightness = 1
            self.casc_clf=os.path.join(Root,r'sources/haarcascade_eye.xml')
            self.display=True
            self.num_threads_save=7
            self.save_txt_path=os.path.join(Root,r'saved_results')
        
        
        elif mode=='analyzing': # analyzing for this moment can be scalable. 
            self.iou_thresh=0.5
            self.display_graph=True
            self.display_hist=True
        
        self.validate_configurations()
    
    
    
    
    
    
    
    
    
    
    
    
    
    def get_run_id(self):
        """
        gives a unique key value "run_id" for logging purposes.
        Returns
        -------
        string: unique run id for request.
        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H%M%S")
        self.run_id=str(self.date)+"_"+str(self.current_time)+"_"+str(random.randint(1000000, 9999999))
        
        if self.mode=='detecting':
            self.run_id="det_"+str(self.date)+"_"+str(self.current_time)+"_"+str(random.randint(1000000, 9999999))
            self.save_txt_path=os.path.join(self.save_txt_path, self.run_id+'.txt')
            file_op.create_file(self.save_txt_path)
        else:
            self.run_id="anl_"+str(self.date)+"_"+str(self.current_time)+"_"+str(random.randint(1000000, 9999999))

        return self.run_id
    
    
    
    def returnCameraIndexes(self):
        # checks the first 5 indexes.
        index = 0
        arr = []
        i = 5
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        return arr 
    
    def validate_configurations(self):
        # check source availability.
        if self.mode=='detecting':
            available_sources=self.returnCameraIndexes()
            if self.camera_source not in available_sources:
                raise ValueError(f"Source {self.camera_source} not found. Available sorces: {available_sources}")
            
            if not os.path.exists(self.casc_clf):
                raise ValueError(f"Casscade Classifier xml {self.casc_clf} not found.") 
            
            if not os.path.exists(self.save_txt_path):
                raise ValueError(f"Saving directory not found: {self.save_txt_path}.")
      
        pass
    
    
