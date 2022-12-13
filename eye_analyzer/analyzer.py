# -*- coding: utf-8 -*-
"""
@author: dorge
"""
# =============================================================================
# This script will analyzed the results from the detector
# =============================================================================
#     filname:        analyzer.py
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
import os
import sys
import ast
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from datetime import datetime, timedelta


sys.path.append('../')
import core.logger as logg
import core.config as conf
import core.file_operation as file_op


class Analyzer:
    def __init__(self):
        config=conf.Config(mode='analyzing')
        self.logger=logg.Logger(run_id=config.get_run_id(), module='Analyzer', mode='analyzing')
        self.iou_thresh=config.iou_thresh
        self.display_grph=config.display_graph
        self.display_hist=config.display_hist
        self.save_txt_path=''
        self.df=pd.DataFrame(columns=['time', 'cords'])
        
    def display_graph(self):
        """
        Displating a graph of median slope value at each 2-sec period of time.
        -------
        None.
        """
        t0=self.df.time.tolist()[0]
        tf=self.df.time.tolist()[-1]
        df_g=pd.DataFrame(columns=['start_time','end_time','median_tilt'])
        while t0<tf:
            cur=t0+timedelta(seconds=2)
            temp_df=self.df[(self.df.time>t0) & (self.df.time<cur)]
            if len(temp_df)>0:
                median=temp_df.tilt.median()
                df_g =df_g.append({'start_time':t0,
                                   'end_time':tf,
                                   'median_tilt':median},ignore_index=True)
            t0=cur
        plt.plot(df_g['start_time'],df_g['median_tilt'])
        plt.show()
        pass
    
    def display_histogram(self):
        """
        Display a histogram of the slope.
        Returns
        -------
        None.
        """
        self.logger.info("displaying histogram")
        plt.hist(self.df['tilt'], color='blue', edgecolor='black')
        plt.xlabel('Tilt in degreese')
        plt.ylabel('No. of samples')
        plt.title('histogram of angles')
        plt.show()
        pass
    
    def calculte_tilt(self):
        """
        calculates the tilt angle of the face by finding the slope between the 
        two eyes.
        Returns
        -------
        None.

        """
        tilt=[]
        for r in self.df.iterrows():
            l_eye, r_eye = r[1]['cords'][0], r[1]['cords'][1]
            l_center_x,l_center_y= l_eye[0]+(l_eye[2]//2), l_eye[1]+(l_eye[3]//2)
            r_center_x,r_center_y= r_eye[0]+(r_eye[2]//2), r_eye[1]+(r_eye[3]//2)
            try:
                myradians = math.atan2(l_center_y-r_center_y, l_center_x-r_center_x)
                mydegrees = math.degrees(myradians)
                tilt.append(mydegrees)
            except ArithmeticError as e:
                self.logger.exception(f"arithmetic error: {str(e)}")
            
        self.df['tilt']=tilt
        self.logger.info(f"mean tilt angle: {self.df.tilt.mean()}")

        pass
    
    def arrange_boxes(self, box_1,box_2):
        """
        Helper function to arrange boxes to desired shape.

        Parameters
        ----------
        box_1 : TYPE
            x,y,w,h of the right eye box.
        box_2 : TYPE
            x,y,w,h of the left eye box.

        Returns
        -------
        box_1 : TYPE
            list contains 4 points of box1.
        box_2 : TYPE
            list contains 4 points of box2.
        """
        box_1 = [[box_1[0], box_1[1]], [box_1[0]+box_1[2], box_1[1]],
                 [box_1[0]+box_1[2], box_1[1]+box_1[3]], [box_1[0], box_1[1]+box_1[3]]]
        
        box_2 = [[box_2[0], box_2[1]], [box_2[0]+box_2[2], box_2[1]], 
                 [box_2[0]+box_2[2], box_2[1]+box_2[3]],[box_2[0], box_2[1]+box_2[3]],]
        return box_1,box_2
        
    
    def calc_iou_helper(self, box_1,box_2):
        """
        Calculate the IoU Between two boxes of eyes from successive frames

        Parameters
        ----------
        box_1 : list 
            t0 eye box. [[x,y], [x,y], [x,y], [x,y]]
        box_2 : list
            t1 eye box. [[x,y], [x,y], [x,y], [x,y]]

        Returns
        -------
        iou : float
            intersect over unions value.

        """
        box_1,box_2=self.arrange_boxes(box_1,box_2)
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    
    def calc_iou(self, t0,t1):
        """
        Parameters
        ----------
        t0 : np.array
            cooardinates of eyes at t0.
        t1 :  np.array
            cooardinates of eyes at t1.

        Returns
        -------
        bool
            Intersect over union of t0 and t1 is greater than 0.5.

        """    
        if ((self.calc_iou_helper(t0[0], t1[0])>self.iou_thresh) and 
            (self.calc_iou_helper(t0[1], t1[1])>self.iou_thresh)):
            return True
        return False
        
    
    def filter_results(self, gen_rows):
        """
        leaving only valid records.
        Valid Record: 
            1. exacly two eyes.
            2. prev frame contains 2 eyes.
            3. IoU is less than Threshold.
        
        Parameters
        ----------
        gen_rows : generator
            txt file iterrator.
        iterrator which will allow reading large txt files with little memo.

        Returns
        -------
        None.
        """
        t0=next(gen_rows)# I decided to not push the first frame.
        if t0=='0' or t0=='':
            self.logger.exception('File is empty')
            raise EOFError("data file is empty!")
            
        for t1 in gen_rows:
            if t1=='': continue
            t0_time, t0_cords=t0.split('\t')
            t1_time, t1_cords=t1.split('\t')
          
            t0_cords=np.asarray(ast.literal_eval(t0_cords))
            t1_cords=np.asarray(ast.literal_eval(t1_cords))
            
            if len(t0_cords)!=2: 
                t0=t1
                continue
            
            if len(t1_cords)!=2:
                t0=next(gen_rows)
                continue
            
            if self.calc_iou(t0_cords,t1_cords):
                t1_time=datetime.fromtimestamp(float(t1_time))
                self.df = self.df.append({'time': t1_time,'cords': t1_cords}, ignore_index=True)
        pass
    
    
    def read_results(self):
        """
        reading the file and create generator.
        Returns
        -------
        None.
        """
        try:
            with open(self.save_txt_path, 'r') as f:
                gen_rows=file_op.rows(f)
                self.filter_results(gen_rows)
                f.close()
        except Exception as e:
            self.logger.exception("path provided not found: {}".format(str(e)))
            raise ValueError("path provided not found: {}".format(str(e)))
  
    
    def run(self, saved_txt_path):
          self.save_txt_path=saved_txt_path
          self.logger.info(f"start analyzing file")
          self.read_results()
          self.calculte_tilt()
          if self.display_hist:
              self.display_histogram()
          if self.display_grph:
              self.display_graph()
        
        

    
Root=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

if __name__=="__main__":
    sample_path=os.path.join(Root,'saved_results\det_2022-12-12_000131_5523654.txt')
    Analyzer().run(saved_txt_path=sample_path)
    print("finished.")