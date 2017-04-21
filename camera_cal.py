import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import pickle

def calibrate_image(img):
    dist_pickle = pickle.load( open( "C:/Users/LPC/Documents/GitHub/CarND-Advanced-Lane-Lines-master/wide_dist_pickle.p", "rb" ) )
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    dst=cal_undistort(img, objpoints, imgpoints)
    return dst

def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# This function test the camera calibration
def test_cal(img,objpoints,imgpoints):
    dst=cal_undistort(img, objpoints, imgpoints)
    cv2.imwrite('road_before_calibration.jpg', img)
    cv2.imwrite('road_after_calibration.jpg', dst)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(dst)
    ax2.set_title('Calibrated image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)    
    plt.show()
    
def calibrate_camera():
    # prepare object points
    nx = 9
    ny = 6

    # Make a list of calibration images
    images = glob.glob('C:/Users/LPC/Documents/GitHub/CarND-Advanced-Lane-Lines-master/camera_cal/calibration*.jpg')

    objpoints = []  # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            #img2=cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    cv2.imwrite('before_calibration.jpg', img)
    cv2.imwrite('after_calibration.jpg', dst)

    cal = {'objpoints': objpoints,
                   'imgpoints': imgpoints,
                   'mtx'      : mtx,
                   'dist'     : dist,
                  }
                   
    with open('C:/Users/LPC/Documents/GitHub/CarND-Advanced-Lane-Lines-master/wide_dist_pickle.p', 'wb') as f:
        pickle.dump(cal, file=f)
    
def main():
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    dist_pickle = pickle.load( open( "C:/Users/LPC/Documents/GitHub/CarND-Advanced-Lane-Lines-master/wide_dist_pickle.p", "rb" ) )
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    img = cv2.imread('C:/Users/LPC/Documents/GitHub/CarND-Advanced-Lane-Lines-master/camera_cal/calibration1.jpg')

    #calibrate_camera()
    test_cal(img,objpoints,imgpoints)

if __name__ == "__main__":
    main()


