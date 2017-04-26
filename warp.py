import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def perspective_transform(img):

    img_size = (img.shape[1], img.shape[0])

    src = np.float32([[220, 700],
                      [555, 455],
                      [730, 455],
                      [1090, 700]])

    # Choose x positions that allow for 3.7m for the lane position closest to car.
    dst = np.float32([[ 212,  700],
                      [ 212,    0],
                      [ 1086,    0],
                      [ 1086,  700]])
                      
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size,flags=cv2.INTER_LINEAR)
        
    # Return the resulting image and matrix
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(img, cmap='gray')
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.imshow(warped, cmap='gray')
    # ax2.set_title('Perspective', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)    
    # plt.show()
    
    #cv2.imwrite('a_non_warped.jpg', img)
    #cv2.imwrite('a_warped.jpg', warped)
    
    return warped,Minv
    
def main():
    img_warp = cv2.imread('C:/Users/LPC/Documents/GitHub/CarND-Advanced-Lane-Lines-master/test_images/2.jpg')
    perspective_transform(img_warp)
    
if __name__ == "__main__":
    main()   