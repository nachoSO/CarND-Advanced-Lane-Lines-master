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
    
    combined=pre.threshold_pipeline(image)

    # f, ax1 = plt.subplots(1, 1, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(combined, cmap='gray')
    # ax1.set_title('Original Image', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)    
    # plt.show()
    
    global left_lane, right_lane,first_time,frame_counter,curve_radius_left,curve_radius_right,offset
    warped,Minv=w.perspective_transform(combined)

    if(first_time==True):
        left_lane, right_lane = lf.sliding_line_finding(warped)
    else:
        left_lane, right_lane = lf.line_finding_after_sliding(warped, left_lane, right_lane)
    
    first_time=False
    img=lf.draw(warped, left_lane, right_lane, image, Minv)
    
    if(frame_counter>=10):
        curve_radius_left,curve_radius_right = lf.measure_curvature(warped,left_lane,right_lane)
        offset = lf.compute_offset(warped, left_lane, right_lane)
        frame_counter=0

    frame_counter+=1
      
    #compute offset
    str_offset='right'
    if(offset<=0):
        str_offset='left'
    avg_curve_radius = np.mean([curve_radius_left + curve_radius_right])
    img = cv2.putText(img, 'Radius of curvature : '+str(round(avg_curve_radius,3))+' m', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    img = cv2.putText(img, 'Offset: '+str(abs(round(offset,3)))+'m '+str_offset+' of center', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return img
    
def main():
    global first_time,frame_counter
    frame_counter=15
    first_time=True

    clip = VideoFileClip('project_video.mp4').fl_image(process_image_pipeline)
    clip.write_videofile('out_project_video.mp4', audio=False, verbose=False)
    # img = cv2.imread('C:/Users/LPC/Documents/GitHub/CarND-Advanced-Lane-Lines-master/test_images/test2.jpg')
    # img = process_image_pipeline(img)
    # f, ax1 = plt.subplots(1, 1, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(img,cmap='brg')
    # ax1.set_title('Original Image', fontsize=50)  
    # plt.show()

if __name__ == "__main__":
    main()   

