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

def single_img_features(image, f_params, vis=False):
    # apply color conversion if other than 'RGB'
    feature_image = np.copy(image)
    
    color_space = f_params.color_space
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    features = []
    
    if f_params.spatial_feat:
        spatial_features = bin_spatial(feature_image, size = f_params.spatial_size)
        features.append(spatial_features)
    if f_params.hist_feat:
        hist_features = color_hist(feature_image, nbins = f_params.hist_bins)
        features.append(hist_features)
    if f_params.hog_feat:
        # Call get_hog_features() with vis=False, feature_vec=True
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
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], 
                                                           f_params.orient, 
                                                           f_params.pix_per_cell, 
                                                           f_params.cell_per_block, 
                                                           vis, 
                                                           feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], 
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