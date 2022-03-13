# In[1.1]: Compute the camera calibration using chessboard images   
import numpy as np
import cv2
import glob

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('C:/Users/chris/Google Drive/Documents/Continuous Learning/Self Driving Cars Nanodegree/Finding Lane Lines Advanced Project/camera_cal/calibration*.jpg')
set_images = []

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(100)
    set_images.append(img)

cv2.destroyAllWindows()

# Obtain distortion coefficients
_, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)

# In[1.2]: Apply distortion correction to raw images DONE
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

test_image = mpimg.imread('C:/Users/chris/Google Drive/Documents/Continuous Learning/Self Driving Cars Nanodegree/Finding Lane Lines Advanced Project/test_images/straight_lines1.jpg')
#test_image = mpimg.imread('C:/Users/chris/Google Drive/Documents/Continuous Learning/Self Driving Cars Nanodegree/Finding Lane Lines Advanced Project/camera_cal/calibration1.jpg')

def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

undistorted = undistort(test_image, mtx, dist)

'''
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted, cmap='gray')
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''
# In[1.3]: Apply a perspective transform to rectify binary image ("birds-eye view")

def warp(img):
    width, height = (img.shape[1], img.shape[0])
    # Determine source points as a trapezoid covering the desired portion of a straight lane
    # Use square coordinates as destiny perspective
    # Start from the top left clock-wisely
    src = np.float32([[575, 460], [704, 460], [1003, 646], [300, 646]])
    dst = np.float32([[250, 0], [width - 250, 0], [width - 250, height], [250, height]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)

    warped = np.copy(img)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped, M, invM

warped, M, invM = warp(undistorted)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(undistorted)
ax1.set_title('Corrected Image', fontsize=50)
ax2.imshow(warped, cmap='gray')
ax2.set_title('Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# In[1.4]: Use color transforms, gradients, etc., to create a thresholded binary image
def thresholdWarped(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx / np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # A combination of the previous factors is enough for an acceptable performance
    # Binary combined
    combined = np.zeros_like(s_binary)
    combined[(sxbinary == 1) | (s_binary == 1)] = 1
    return combined
    
binary_warped = thresholdWarped(warped)
plt.imshow(binary_warped, cmap= 'gray')

'''
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(warped)
ax1.set_title('Warped Image', fontsize=50)
ax2.imshow(binary_warped, cmap='gray')
ax2.set_title('Thresholded Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''
# In[1.5]: Detect lane pixels and fit to find the lane boundary
def findCurve(img):
    # Input must be a warped binary image
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis = 0)
    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((img, img, img)) * 255)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Set height of windows
    window_height = np.int(img.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, out_img

def visualizeCurve(img, out_img, left_fit, right_fit, left_lane_inds, right_lane_inds):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = findCurve(binary_warped)
visualizeCurve(binary_warped, out_img, left_fit, right_fit, left_lane_inds, right_lane_inds)

# In[1.7]: Determine the curvature of the lane and vehicle position with respect to center

def measureCurvature(binary_warped, left_lane_inds, right_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    y_eval = np.max(ploty)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    return left_curverad, right_curverad

def measurePosition(binary_warped, left_fit, right_fit):
    xm_per_pix = 3.7/700
    vehicle_position = binary_warped.shape[1]/2
    height = binary_warped.shape[0]
    
    left_fit_x = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    right_fit_x = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    
    lane_center_position = (right_fit_x + left_fit_x) /2
    
    center_distance = (vehicle_position - lane_center_position) * xm_per_pix
    
    return center_distance

left_curverad, right_curverad = measureCurvature(binary_warped, left_lane_inds, right_lane_inds)
print(left_curverad, 'm', right_curverad, 'm')
vehicle_pos = measurePosition(binary_warped, left_fit, right_fit)
print(vehicle_pos, 'm')

# In[1.8]: Draw lines and contained area over the undistorted image
    
def drawContent(original_img, binary_img, left_fit, right_fit, invM):
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 50, 100))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=50)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 255, 0), thickness=50)
    # Warp the blank back to original image space using inverse perspective matrix (invM)
    newwarp = cv2.warpPerspective(color_warp, invM, (original_img.shape[1], original_img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(original_img, 1, newwarp, 1, 0)
    
    return result

img_content = drawContent(undistorted, binary_warped, left_fit, right_fit, M)
plt.imshow(img_content)

# In[1.9]: Print data parameters on the image
    
def drawData(img, curv_rad, vehicle_pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text = 'Deviation <Vehicle,Center>: ' + '{:.3f}'.format(vehicle_pos) + 'm'
    cv2.putText(img, text, (30, 50), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    text = 'R: ' + '{:.1f}'.format(curv_rad) + 'm'
    cv2.putText(img, text, (30, 100), font, 1, (255,255,255), 2, cv2.LINE_AA)
    
    return img

final_img = drawData(img_content, (left_curverad + right_curverad) / 2, vehicle_pos)
plt.imshow(final_img)

# In[2.1]: Pipeline
def pipeline(img):  
    original_img = np.copy(img)
    
    undist = cv2.undistort(original_img, mtx, dist, None, mtx)
    
    warped_image, M, invM = warp(undist)
    
    binary_img = thresholdWarped(warped_image)
    
    left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = findCurve(binary_img)   

    img_content = drawContent(original_img, binary_img, left_fit, right_fit, invM)
    left_curverad, right_curverad = measureCurvature(binary_img, left_lane_inds, right_lane_inds)
    vehicle_pos = measurePosition(binary_img, left_fit, right_fit)
    final_img = drawData(img_content, (left_curverad + right_curverad) / 2, vehicle_pos)
    
    return img_content
    
# In[2.2]: I/O Video through the pipeline
from moviepy.editor import VideoFileClip

video_output1 = 'C:/Users/chris/Google Drive/Documents/Continuous Learning/Self Driving Cars Nanodegree/Finding Lane Lines Advanced Project/project_video_output.mp4'
video_input1 = VideoFileClip('C:/Users/chris/Google Drive/Documents/Continuous Learning/Self Driving Cars Nanodegree/Finding Lane Lines Advanced Project/project_video.mp4')#.subclip(22, 30)
processed_video = video_input1.fl_image(pipeline)
%time processed_video.write_videofile(video_output1, audio=False)
