import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
#from sklearn.preprocessing import StandardScalar
from sklearn.model_selection import train_test_split

class FeatureParams:
    def __init__(self, color_space, spatial_size, hist_bins, 
                       orient, pix_per_cell, cell_per_block, hog_channel,
                       spatial_feat, hist_feat, hog_feat):
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        
    def str(self):
        return '{}-ss{}-hb{}-o{}-p{}-c{}-hc{}-sf{:d}-hist{:d}-hog{:d}'.format(
                    self.color_space,
                    self.spatial_size,
                    self.hist_bins,
                    self.orient,
                    self.pix_per_cell,
                    self.cell_per_block,
                    self.hog_channel,
                    self.spatial_feat,
                    self.hist_feat,
                    self.hog_feat)
    
    def descriptive_str(self):
        d = dict()
        d['Color space'] = self.color_space
        d['Spatial size'] = self.spatial_size
        d['Color histogram bins'] = self.hist_bins
        d['HOG orientations'] = self.orient
        d['HOG pixels per cell'] = self.pix_per_cell
        d['HOG cell per block'] = self.cell_per_block
        d['HOG channel'] = self.hog_channel
        d['Spatial features'] = self.spatial_feat
        d['Histogram features'] = self.hist_feat
        d['HOG features'] = self.hog_feat
        
        result = ''
        for key, value in d.items():
            result += key + ': ' + str(value) + '\n'
        return result

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    c0 = cv2.resize(img[:,:,0], size).ravel()
    c1 = cv2.resize(img[:,:,1], size).ravel()
    c2 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((c0, c1, c2))

# Define a function to compute color histogram features  
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    hist0 = np.histogram(img[:,:,0], bins=nbins)
    hist1 = np.histogram(img[:,:,1], bins=nbins)
    hist2 = np.histogram(img[:,:,2], bins=nbins)
    # Generating bin centers
    # bin_edges = hist0[1]
    # bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((hist0[0], hist1[0], hist2[0]))
    return hist_features

from skimage.feature import hog
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, 
                                  orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=True, 
                                  feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, 
                       orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=False, 
                       feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of image_paths
# Have this function call bin_spatial() and color_hist()
def extract_features(image_paths, feature_params):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of image_paths
    for file in image_paths:
        # Read in each one by one
        image = mpimg.imread(file)
        file_features = single_img_features(image, feature_params)
        features.append(file_features)
    # Return list of feature vectors
    return features
    
def convert_color(img, f_params):
    color_space = f_params.color_space
    if color_space != 'RGB':
        if color_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            raise ValueError('Unsupported color_space. [{}]'.format(color_space))
    else:
        return np.copy(image)

def single_img_features(image, f_params, vis=False):
    # Apply color conversion if other than 'RGB'
    feature_image = convert_color(image, f_params)

    features = []
    
    if f_params.spatial_feat:
        spatial_features = bin_spatial(feature_image, size = f_params.spatial_size)
        features.append(spatial_features)
    if f_params.hist_feat:
        hist_features = color_hist(feature_image, nbins = f_params.hist_bins)
        features.append(hist_features)
    if f_params.hog_feat:
        if f_params.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                                    f_params.orient, 
                                                    f_params.pix_per_cell, 
                                                    f_params.cell_per_block, 
                                                    vis=False, 
                                                    feature_vec=True))
            hog_features = np.ravel(hog_features)
            #hog_features = np.concatenate(hog_features)
        else:
            if vis:
                hog_features, hog_image = get_hog_features(feature_image[:,:,f_params.hog_channel], 
                                                           f_params.orient, 
                                                           f_params.pix_per_cell, 
                                                           f_params.cell_per_block, 
                                                           vis, 
                                                           feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,f_params.hog_channel], 
                                                f_params.orient, 
                                                f_params.pix_per_cell, 
                                                f_params.cell_per_block, 
                                                vis, 
                                                feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    
    if vis:
        return np.concatenate(features), hog_image
    else:
        return np.concatenate(features)
    
def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])
    plt.show()
    
def full_hog(filename, fp, clf, X_scaler, scale, cells_per_step):
    img = mpimg.imread(filename)
    return full_hog_single_image(img, fp, clf, X_scaler, scale, cells_per_step)
    
# Based on Ryan Keenan's code in Vehicle Detection Walkthrough of Project Q&A video.
# https://www.youtube.com/watch?v=P2zwrTM8ueA
def full_hog_single_img(img, fp, clf, X_scaler, scale, y_start_stop, cells_per_step):
    ystart = y_start_stop[0]
    ystop = y_start_stop[1]
    
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32) / 255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, fp)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    channel_count = 3
    channels = []
    for i in range(channel_count):
        channels.append(ctrans_tosearch[:,:,i])
        
    full_hog = []
    for ch in channels:
        hog = get_hog_features(ch, 
                                fp.orient, 
                                fp.pix_per_cell, 
                                fp.cell_per_block, 
                                vis=False, 
                                feature_vec=False)
        full_hog.append(hog)
    
    nxblocks = (channels[0].shape[1] // fp.pix_per_cell) - 1
    nyblocks = (channels[0].shape[0] // fp.pix_per_cell) - 1
    nfeat_per_block = fp.orient * fp.cell_per_block ** 2
    window = 64
    nblocks_per_window = (window // fp.pix_per_cell) - 1
    # Controls overlap. 
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    hot_windows = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            
            hog_feat = []
            for h in full_hog:
                hog_feat.append(h[ypos : ypos + nblocks_per_window,
                                  xpos : xpos + nblocks_per_window].ravel())
                                  
            hog_features = np.hstack((hog_feat[0], hog_feat[1], hog_feat[2]))
            
            xleft = xpos * fp.pix_per_cell
            ytop = ypos * fp.pix_per_cell
            
            subimg = cv2.resize(ctrans_tosearch[ytop : ytop + window,
                                                xleft : xleft + window], (64, 64))
                                                
            spatial_features = bin_spatial(subimg, size=fp.spatial_size)
            hist_features = color_hist(subimg, nbins=fp.hist_bins)
            
            stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(stacked)
            test_prediction = clf.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                top_left = (xbox_left, ytop_draw + ystart)
                bottom_right = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                bbox = (top_left, bottom_right)
                hot_windows.append(bbox)
    
    window_count = nxsteps * nysteps
    
    return (hot_windows, window_count)
