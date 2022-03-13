## README

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output/Figure_0.png "Undistorted"
[image2]: ./output/Figure_1.png "Road Transformed"
[image3]: ./output/Figure_2.png "Warp Example"
[image4]: ./output/Figure_3.png "Binary Example"
[image5]: ./output/Figure_4.png "Fit Visual"
[image6]: ./output/Figure_6.png "Output"
[video1]: ./output/project_video_output.mp4 "Video"

---

### Camera Calibration

### The code is modulized in blocks so it can be executed step by step and understood clearly. There are 2 main blocks, one for the image pipeline designed by 1. and the other one, 2. for the video pipeline. All the code is in P1.py file and each module can be consulted directly as the documentation follows:

#### In[1.1]: Compute the camera calibration using chessboard images

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### In[1.2]: Apply distortion correction to raw images

Once `cv2.calibrateCamera()` is executed in the previous module, `mtx` and `dist` parameters are obtained, which will be fed to `cv2.undistort()` function. The output image will be free of the distortions produced by the camera lens.
![alt text][image2]

#### In[1.3]: Apply a perspective transform to rectify binary image ("birds-eye view")

The `warp()` function takes as inputs an image (`img`). The source (`src`) points are selected manually composing a trapezoidal region of interest. And the destination (`dst`) points are defined as a square shape.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 460      | 250, 0        | 
| 704, 460      | 1030, 0      |
| 1003, 646     | 1030, 720      |
| 300, 646      | 250, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### In[1.4]: Use color transforms, gradients, etc., to create a thresholded binary image

A combination of color and gradient (1-D) thresholds to generate a binary image was implemented.  Here's an example of my output for this step.

![alt text][image4]

#### In[1.5]: Detect lane pixels and fit to find the lane boundary

The implementation of this module is based in the Sliding Window Search algorithm which consist in the following steps:

1. The left and right base points are computed from the peaks of the histogram of the vertical non-zero pixels
2. Then all non-zero pixels are located
3. Iterations begin over the windows from a.
4. Non-zero pixels contained in the defined window are gathered
5. The indices of such pixels are retrieved in a list and the center of the next window is based using such points
6. The left points are separated from the right ones
7. A 2nd degree polynomial is fit to such points

![alt text][image5]

#### In[1.7]: Determine the curvature of the lane and vehicle position with respect to center

The radius is computed following these steps:
1. Use scale values to convert from pixel to meter domain
2. Plot scaled polynomial from bottom to top of the image as horizontal axis
3. Apply curvature formula from the scaled polynomial for left and right curves
4. Compute average of both side values

The distance from the center is calculated by knowing that the center of the image corresponds to the center of the vehicle. It is possible to infere the deviation distance by taking the average of the left bottom and right bottom most points of the lines and substracting from the image center.

#### In[1.8]: Draw lines and contained area over the undistorted image
Here is an example of my result on a test image:
![alt text][image6]

---

### Pipeline (video)

Here's an output sample:
![alt text][video1]

---

### Discussion

#### 1. Briefly problems / issues are discussed in the implementation of this project.  Where will the pipeline likely fail?  What could I do to make it more robust?

The designed pipeline works reasonably well as long as the curvature of the lanes is not aggressive and there is consistency of brightness. The pipeline was tested on the alternative video with terrible results because in such samples, there were sporadic shadows on the lanes and strong curvatures in the lanes of the mountain road video.

The future work that can be apply to enhance the robustness is to use a low-pass filter to smooth out the line projections "shaking" effect. In addition, to improve the performance of the algorithm, an extra module right after the block in charge of finding the curves can be added. One that takes a prior knowledge of the polynomial fit of the sliding window to continue the process, since the latter one is comparatively more expensive in terms of computational cost.
