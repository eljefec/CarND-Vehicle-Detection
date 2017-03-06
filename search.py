import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

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
    
class Searcher:
    def __init__(self, feature_params, clf, X_scaler):
        self.feature_params = feature_params
        self.clf = clf
        self.X_scaler = X_scaler
        
    # hog_flavor values: 'local', 'full'
    def search(self, img, hog_flavor):
        if hog_flavor == 'local':
            return self.search_local_hog(img)
        elif hog_flavor == 'full':
            return self.search_full_hog(img)
        else:
            raise ValueError('Invalid hog_flavor. [{}]'.format(hog_flavor))
        
    def search_local_hog(self, img):
        y_start_stop = [400, 656]
        overlap = 0.5
        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255
        
        windows = slide_window(img, x_start_stop = [None, None], y_start_stop = y_start_stop, 
                        xy_window = (96, 96), xy_overlap  = (overlap, overlap))
                        
        hot_windows = search_windows(img, windows, self.clf, self.X_scaler, self.feature_params)
        
        window_img = draw_boxes(draw_img, hot_windows, color = (0, 0, 255), thick = 6)
        
        print('Search window count: ', len(windows))
        
        return [window_img]
        
    def search_full_hog(self, img):
        draw_img, heatmap = cl.full_hog_single_img(img, self.feature_params, self.clf, self.X_scaler)
        return [draw_img, heatmap]
    
if __name__ == '__main__':
    # model = 'trained_models/HSV-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.49.p'
    # model = 'trained_models/HLS-ss(16, 16)-hb16-o9-p10-c2-hcALL-sf1-hist1-hog1-acc99.32.p'
    model = 'trained_models/YCrCb-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.21.p'
    (fp, clf, X_scaler) = tr.load_classifier(model)

    searcher = Searcher(fp, clf, X_scaler)
    searchpath = 'test_images/*'
    example_imgs = glob.glob(searchpath)
    imgs = []
    titles = []

    for img_src in example_imgs:
        img = mpimg.imread(img_src)

        for flavor in ['local', 'full']:
            sw = Stopwatch()
            found_imgs = searcher.search(img, flavor)
            sw.stop()
            imgs.extend(found_imgs)
            for i in range(len(found_imgs)):
                titles.extend(['{} {}'.format(flavor, img_src)])
            print('Flavor: {}, Time to search one image: {}'.format(flavor, sw.format_duration(coarse=False)))

    fig = plt.figure(figsize = (8, 11))
    cl.visualize(fig, len(example_imgs), 3, imgs, titles)
