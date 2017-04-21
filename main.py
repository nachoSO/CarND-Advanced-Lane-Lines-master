import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import warp as w
import camera_cal as cc
import line_finding as lf
from line_finding import Line

import preprocess as pre
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque

def process_image_pipeline(image):
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    image = cc.calibrate_image(image)
    window = image[0:, :, :]
    # Apply each of the thresholding functions
    gradx = pre.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = pre.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = pre.mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = pre.dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

    ylw = pre.extract_yellow(window)
    highlights = pre.extract_highlights(window[:, :, 0])

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | (ylw == 255) | 
    (highlights == 255) |((mag_binary == 1) & (dir_binary == 1))] = 1

    global left_lane, right_lane,first_time,frame_counter,curve_radius
    warped,Minv=w.perspective_transform(combined)

    if(first_time==True):
        left_lane, right_lane = lf.sliding_line_finding(warped)
    else:
        left_lane, right_lane = lf.line_finding_after_sliding(warped, left_lane, right_lane)
        
    first_time=False
    img=lf.draw(warped, left_lane, right_lane, image, Minv)
    
    if(frame_counter>=10):
        curve_radius = lf.measure_curvature(warped,left_lane,right_lane)
        frame_counter=0
      
    img = cv2.putText(img, 'Radius of curvature: '+str(curve_radius)+' m', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    #offset = lf.compute_offset(undist, left_lane, right_lane)
    frame_counter+=1
    img = cv2.putText(img, 'Radius of curvature: '+str(curve_radius)+' m', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    #img = cv2.putText(img, 'Offset: '+str(offset)+' m', (30, 80), font, 1, (0,255,0), 2)
    return img
  
    
def main():
    global first_time,frame_counter
    frame_counter=15
    first_time=True
    clip = VideoFileClip('project_video.mp4').fl_image(process_image_pipeline)
    clip.write_videofile('out_project_video.mp4', audio=False, verbose=False)
    
if __name__ == "__main__":
    main()   

