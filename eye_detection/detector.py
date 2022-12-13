# -*- coding: utf-8 -*-
"""@author: dorge"""

# =============================================================================
# This script perform the eyes detection class 
# =============================================================================
#     filname:        detector.py
#     version:        1.0
#     author:         dorge
#     creation_date:  11-12-2022
#     
#     change history:
#     
#     who       when      version     changes
#     ----      -----     -------     --------
#     dorge     11-12-22   1.0         INIT
# =============================================================================

import sys
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


sys.path.append('../')
import core.logger as logg
import core.config as conf
import core.file_operation as file_op


class Detector:
    def __init__(self):
        config=conf.Config(mode='detecting')
        self.logger=logg.Logger(run_id=config.get_run_id(), module='Detector', mode='detecting')
        self.source=config.camera_source
        self.save_file_path=config.save_txt_path
        self.display=config.display
        self.debug_mode=config.debug_mode
        if not self.debug_mode:
            self.pool=ThreadPoolExecutor(config.num_threads_save)        
        self.eye_cascade=cv2.CascadeClassifier(config.casc_clf)
        
        
    def drawer(self, img, eyes):
        if len(eyes)>=2: # if there are multiple eyes we only presents 2.
            eyes=np.stack((eyes[0], eyes[1]), axis=0)
        for (x,y,w,h) in eyes:
            sub_img = img[y:y+h, x:x+w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            higlight = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            img[y:y+h, x:x+w] = higlight
        return img



    def detect_face(self, img):
        return self.eye_cascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 4)
            
    
    def run(self):
        frame_counter=0
        cap = cv2.VideoCapture(self.source)
        while(True):
            _,img=cap.read()
            frame_counter+=1
            eyes_cords=self.detect_face(img)
            
            if self.display:
                present_img=self.drawer(img, eyes_cords)
                cv2.imshow("Eyes Detection", present_img)
            
            # submit to the thread pool for saving coordinates.
            if not self.debug_mode:
                self.pool.submit(file_op.append_detections_to_text_file, 
                                      self.save_file_path, eyes_cords)
                
            if cv2.waitKey(1) &  0xFF == ord('q'): # Escape when q is pressed
                break
                
        if self.display:
            cv2.destroyAllWindows()
        self.pool.shutdown()
        self.logger.info(f"number of frames in session: {frame_counter}")


if __name__=="__main__":
    Detector().run()