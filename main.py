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
    #gradx = pre.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    #grady = pre.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = pre.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(40, 100))
    dir_binary = pre.dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    #threshold on the saturation channel
    s_threshold = pre.hls_select(image, thresh=(150, 255))        
    
    combined = np.zeros_like(dir_binary)

    combined[(((mag_binary == 1) & (dir_binary == 1))  ) | (s_threshold==1)] = 1

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
    
    first_time=True
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
    #Draw offset & radius
    img = cv2.putText(img, 'Radius of curvature left: '+str(round(curve_radius_left,3))+' m'+' | right: '+str(round(curve_radius_right,3))
    +' m', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    img = cv2.putText(img, 'Offset: '+str(abs(round(offset,3)))+'m '+str_offset+' of center', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return img
  
    
def main():
    
    global first_time,frame_counter
    frame_counter=15
    first_time=True
    clip = VideoFileClip('project_video.mp4')
    # out=clip.fl_image(lambda frame: process_image_pipeline(frame))
    # out.write_videofile('out_project_video.mp4', audio=False, verbose=False)
    
    #single image test (1,4)
    img = cv2.imread('C:/Users/LPC/Documents/GitHub/CarND-Advanced-Lane-Lines-master/test_images/test4.jpg')
    img = process_image_pipeline(img)
    f, ax1 = plt.subplots(1, 1, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)  
    plt.show()
    cv2.imwrite('asdadadsa.jpg', img)
    
if __name__ == "__main__":
    main()   

