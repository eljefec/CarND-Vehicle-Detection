import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label

import classify as cl
import train as tr
from Stopwatch import Stopwatch

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, feature_params):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = cl.single_img_features(test_img, feature_params)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap
    
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
    
class SearchParams:
    def __init__(self, scale, y_start_stop):
        self.scale = scale
        self.y_start_stop = y_start_stop
        
    def str(self):
        return 'Scale: ' + str(self.scale) + ', y_start_stop: ' + str(self.y_start_stop)
        
    @staticmethod
    def get_defaults():
        return [SearchParams(1, (400, 500)),
                SearchParams(1.5, (400, 550)),
                SearchParams(2, (400, 656))]
    
class Searcher:
    def __init__(self, feature_params, clf, X_scaler):
        self.feature_params = feature_params
        self.clf = clf
        self.X_scaler = X_scaler
    
    def search(self, img, hog_flavor, search_params):
        search_window_total = 0
        all_hot_windows = []
        for sp in search_params:
            hot_windows, search_window_count = self.single_search(img, hog_flavor, sp.scale, sp.y_start_stop)
            all_hot_windows.extend(hot_windows)
            search_window_total += search_window_count

        window_img = draw_boxes(img, all_hot_windows, color = (0, 0, 255), thick = 5)
            
        heatmap = np.zeros_like(img[:,:,0])
        for bbox in all_hot_windows:
            top_left = bbox[0]
            bottom_right = bbox[1]
            heatmap[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] += 1

        heatmap = apply_threshold(heatmap, 1)
        labels = label(heatmap)
        labeled_img = draw_labeled_bboxes(img, labels)
        
        return heatmap, labeled_img, search_window_total
    
    # hog_flavor values: 'local', 'full'
    def single_search(self, img, hog_flavor, scale, y_start_stop):
        if hog_flavor == 'local':
            return self.search_local_hog(img, scale, y_start_stop)
        elif hog_flavor == 'full':
            return self.search_full_hog(img, scale, y_start_stop)
        else:
            raise ValueError('Invalid hog_flavor. [{}]'.format(hog_flavor))
        
    def search_local_hog(self, img, scale, y_start_stop):
        overlap = 0.5
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255
        
        window = int(scale * 64)
        xy_window = (window, window)
        
        windows = slide_window(img, x_start_stop = [None, None], y_start_stop = y_start_stop, 
                        xy_window = xy_window, xy_overlap  = (overlap, overlap))
                        
        hot_windows = search_windows(img, windows, self.clf, self.X_scaler, self.feature_params)
        
        return hot_windows, len(windows)
        
    def search_full_hog(self, img, scale, y_start_stop):
        hot_windows, window_count = cl.full_hog_single_img(img, 
                                                            self.feature_params, 
                                                            self.clf, 
                                                            self.X_scaler, 
                                                            scale, 
                                                            y_start_stop)
        
        return hot_windows, window_count
    
if __name__ == '__main__':
    # model = 'trained_models/HSV-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.49.p'
    # model = 'trained_models/HLS-ss(16, 16)-hb16-o9-p10-c2-hcALL-sf1-hist1-hog1-acc99.32.p'
    # model = 'trained_models/YCrCb-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.21.p'
    model = 'trained_models/YCrCb-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.72.p'
    (fp, clf, X_scaler) = tr.load_classifier(model)

    searcher = Searcher(fp, clf, X_scaler)
    searchpath = 'test_images/*'
    example_imgs = glob.glob(searchpath)
    #example_imgs = ['test_images/test1.jpg']
    imgs = []
    titles = []
    search_params = SearchParams.get_defaults()
    
    for sp in search_params:
        print(sp.str())

    for img_src in example_imgs:
        img = mpimg.imread(img_src)

        for flavor in ['full']:
            sw = Stopwatch()
            heatmap, labeled_img, window_count = searcher.search(img, flavor, search_params)
            sw.stop()
            
            imgs.append(heatmap)
            imgs.append(labeled_img)
            for i in range(2):
                titles.extend(['{} {}'.format(flavor, img_src)])
            print('Flavor: {}, Window count: {}, Time to search one image: {}'.format(flavor, window_count, sw.format_duration(coarse=False)))

    fig = plt.figure(figsize = (8, 11))
    cl.visualize(fig, len(example_imgs), len(imgs) / len(example_imgs), imgs, titles)
