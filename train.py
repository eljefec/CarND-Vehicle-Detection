import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import classify as cl
import load as ld

from Stopwatch import Stopwatch 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Set n_samples = 0 when you want to use all training data.
def train_classifier(n_samples, feature_params):
    lists = ld.get_file_lists(['../data/vehicles/', '../data/non-vehicles/'])

    feature_sw = Stopwatch()

    class_features = []
    for img_list in lists:
        if n_samples > 0:
            random_idxs = np.random.randint(0, len(img_list), n_samples)
            selected_paths = np.array(img_list)[random_idxs]
        else:
            selected_paths = img_list

        features = cl.extract_features(selected_paths, feature_params)
        class_features.append(features)

    feature_sw.stop()

    X = np.vstack((class_features[0], class_features[1])).astype(np.float64)

    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(class_features[0])), 
                   np.zeros(len(class_features[1]))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

    svc = LinearSVC()
    fit_sw = Stopwatch()
    svc.fit(X_train, y_train)
    fit_sw.stop()
    
    data = dict()
    data['time_for_feature_computation'] = feature_sw.format_duration()
    data['feature_vector_length'] = class_features[0][0].shape
    data['car_example_counts'] = len(class_features[0])
    data['notcar_example_counts'] = len(class_features[1])
    data['time_for_fitting_classifier'] = fit_sw.format_duration()
    data['test_accuracy'] = svc.score(X_test, y_test)
    data['rand_state'] = rand_state
    data['feature_params'] = feature_params
    
    print(data)
    
    data['clf'] = svc
    data['x_scaler'] = X_scaler
    
    return (svc, X_scaler, data)
    
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def make_pickle_filename(folder, feature_params, test_accuracy):
    return folder + '/' + feature_params.str() + '-acc' + str(test_accuracy) + '.p' 

def train_and_save_classifier(feature_params, n_samples = 0):
    print('Training classifier. Params=[{}]'.format(feature_params.str()))

    folder = 'trained_models'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    (svc, X_scaler, data) = train_classifier(n_samples, feature_params)
    test_accuracy = round(data['test_accuracy'], 4) * 100

    filename = make_pickle_filename(folder, feature_params, test_accuracy)
    save_data(data, filename)
    
def load_classifier(filename):
    data = load_data(filename)
    
    fp = data['feature_params']
    clf = data['clf']
    X_scaler = data['x_scaler']
    test_accuracy = data['test_accuracy']
    
    print('Loaded classifier. ', filename)
    print('Test accuracy: ', round(test_accuracy, 4))
    print(fp.descriptive_str())
    
    return (fp, clf, X_scaler)
    
def get_defaults():
    return cl.FeatureParams(color_space = 'YCrCb', 
                            spatial_size = (16, 16),
                            hist_bins = 16,
                            orient = 9,
                            pix_per_cell = 8,
                            cell_per_block = 2,
                            hog_channel = 'ALL',
                            spatial_feat = True,
                            hist_feat = True,
                            hog_feat = True)
    
if __name__ == '__main__':
    # Experiment with different feature extraction parameters.
    for attempt in range(4):
        print()
        print('Attempt ', attempt)
        for color_space in ['YCrCb']:
            for pix_per_cell in [8, 10]:
                for hog_channel in [0, 1, 2, 'ALL']:
                    print()
                    fp = get_defaults()
                    fp.color_space = color_space
                    fp.pix_per_cell = pix_per_cell
                    fp.hog_channel = hog_channel
                    train_and_save_classifier(fp)
