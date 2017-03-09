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
[color_spatial]: ./output_images/color_spatial.png
[heatmap_label1]: ./output_images/heatmap_label1.png
[heatmap_label2]: ./output_images/heatmap_label2.png
[heatmap_label3]: ./output_images/heatmap_label3.png
[heatmap_label4]: ./output_images/heatmap_label4.png
[heatmap_label5]: ./output_images/heatmap_label5.png
[heatmap_label6]: ./output_images/heatmap_label6.png
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

I also included spatial information and color histogram in the features. Here is an example with parameters `spatial_size=(16, 16)` and `hist_bins=16`:

![alt text][color_spatial]

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

This code is in `search.py` in function `slide_window()` for the patch-by-patch implementation and in `classify.py` in function `full_hog_single_img()` for the implementation where HOG is calculated once per scale level.

My implementation supports multi-scale search.

I chose multiple scales that were appropriate for finding cars at different distances. Near the horizon, I searched smaller scales in a narrow band, and in the bottom half of the image, I searched larger scales in a wider band.

I ran various searches with different scale parameters on test images, then on the project video to see what values worked well.

I settled on these values. The parameters are `(scale, y_start_stop, cells_per_step)`.

    def get_defaults():
        return [SearchParams(0.75, (400, 500), 4),
                SearchParams(1, (400, 500), 4),
                SearchParams(1.5, (400, 550), 2),
                SearchParams(2, (400, 656), 2)]

cells_per_step controls overlap between patches in `full_hog_single_img()`. 4 is an overlap of 0.5, while 2 is an overlap of 0.75.

To keep the number of search windows small, I had less overlap at smaller scales and more overlap at larger scales.

![alt text][sliding_window]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

##TODO
I experimented with the feature parameters (like spatial size, color histogram bin count, and HOG parameters) to optimize the test accuracy of my classifier as I described above. I manually tried different search parameters (like scale, y_start_stop, overlap, and scale count) and visually evaluated the results on the 6 test images below. I also tweaked the search parameters when I was evaluating my pipeline's output for the project video.

At first, I applied a heatmap threshold to each frame in my search pipeline, but I found I got better video results when I instead applied the heatmap threshold to the multi-frame heatmap. This is explained more below.

Here are the heatmap and labeled bounding boxes for the 6 test images:

![alt text][heatmap_label1]
![alt text][heatmap_label2]
![alt text][heatmap_label3]
![alt text][heatmap_label4]
![alt text][heatmap_label5]
![alt text][heatmap_label6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_boost_heat_th1.7_hw12_vw7.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

My vehicle tracking pipeline is in `track.py` in class `Tracker`. A client calls `Tracker.track(img)` for each successive frame in a video.

For each frame, `Tracker` calls `Searcher.search()`. `Searcher.search()` returns a single-frame heatmap of classifier detections. `Tracker` ignores the bounding boxes returned by `Searcher.search()` and instead makes bounding boxes on a multi-frame heatmap. The `Searcher` class is in `search.py`.

`Tracker` filters for false positives by building a multi-frame heatmap and applying a threshold (see `Tracker.smooth_heatmaps()` in lines 188-197). The threshold value is per-frame and I settled on a value of 1.7. This means that as the `Tracker` builds up a larger sliding window of frames, the threshold for the heatmap increases.

    heatmap = sr.apply_threshold(heatmap, int(self.heatmap_threshold_per_frame * len(self.frames)))

On the thresholded multi-frame heatmap, `Tracker` combines overlapping detections using `scipy.ndimage.measurements.label()` and returns a bounding box for each label blob. `Tracker` assumes each bounding box is a vehicle.

To smooth the bounding box around vehicles, I tracked individual vehicles in a `Vehicle` class. The `Vehicle` class recorded a sliding window of bounding boxes, and averaged them over time. This presented some challenges when cars overlapped because their respective bounding boxes merged in the heatmap. I dealt with this by resetting the vehicle tracker when the number of bounding boxes changed and stabilized at a new number. This code is in `Tracker.check_box_change()` in lines 125-137.

Tracking vehicles requires an algorithm for deciding when a car has disappeared from view. My `Vehicle` class considered itself as disappeared when it has not found itself in several frames. The code for this is in `Vehicle.check_ownership()` in lines 66-69.

I also added a feature for boosting the heatmap around confident vehicle detections. After a vehicle has been detected in enough frames, the `Tracker` boosts the heatmap within the vehicle's bounding box by a factor of 3. This code is in `Tracker.boost_heatmap()` in lines 199-204.

I experimented with the `Tracker` parameters and evaluated the performance on the project video. I settled on these `Tracker` parameters:

* Heatmap window size: 12
* Vehicle window size: 7
* Heatmap threshold per frame: 1.7
* Threshold for boosting heatmap around a vehicle: 24 detected frames
* Multiplier for boosting heatmap: 3
* Threshold of frames indicating genuine box count change (for resetting vehicle-tracking): 3

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##TODO
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

