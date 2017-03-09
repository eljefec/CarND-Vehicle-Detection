##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/car.png
[noncar]: ./output_images/noncar.png
[hog]: ./output_images/hog.png
[sliding_window]: ./output_images/sliding_window.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is my write-up. Thank you for reviewing my project!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This code is in `classify.py` in functions `get_hog_features()` (lines 79-97), and `full_hog_single_img()` (lines 197-274). `single_img_features()` (lines 131-176) gets spatial, color, and HOG features for one patch at a time, whereas `full_hog_single_img()` calculates HOG once for an image then searches across the image for classifier matches.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

#####Car
![alt text][car]

#####Not Car
![alt text][noncar]

The code depends on `skimage.feature.hog()` to extract HOG features from an image.

Here is an example of HOG output using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters including HOG and evaluated the test accuracy of a trained linear SVM classifier on a set-aside test set. I settled on the HOG parameters that led to reasonably high test accuracy of the classifier.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for classifier training is in `train.py`, mainly in function `train_classifier()` (lines 16-65).

I trained a linear SVM using `sklearn.preprocessing.StandardScaler` for normalizing features, `sklearn.model_selection.train_test_split` for splitting training and test data, and `sklearn.svm.LinearSVC` for fitting a linear SVM classifier.

I followed these steps to train a classifier:

1. Read the images from the provided datasets `vehicles.zip` and `non-vehicles.zip`.
2. Extract features (HOG, spatial binning, and color histogram).
3. Normalize features using `StandardScaler`.
4. Split data into training and test sets.
5. Fit a linear SVM classifier.
6. Measure test accuracy of classifier.
7. Save the classifier for future use.

I tried various experiments like disabling features and tweaking parameters with the goal of maximizing test accuracy. 

My final model pickled in `YCrCb-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.72.p` has a test accuracy of 99.72%. Here are its parameters:

* Test accuracy: 0.9972
* Spatial features: True
* Spatial size: (16, 16)
* Histogram features: True
* Color histogram bins: 16
* Color space: YCrCb
* HOG features: True
* HOG channel: ALL
* HOG orientations: 9
* HOG pixels per cell: 8
* HOG cell per block: 2

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

##TODO

![alt text][sliding_window]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

##TODO
Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
##TODO
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

##TODO
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##TODO
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

