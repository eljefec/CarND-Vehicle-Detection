import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import classify as cl
import load as ld
from Stopwatch import Stopwatch 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

#files = ld.get_file_list('vehicles/')
lists = ld.get_file_lists(['vehicles/', 'non-vehicles/'])

color_space='YCrCb', 
spatial_size=(16, 16)
hist_bins = 16
orient=9
pix_per_cell=8
cell_per_block=2
hog_channel='ALL'
spatial_feat=True
hist_feat=True
hog_feat=True
vis = True

sw = Stopwatch()
sw.start()

class_features = []
for img_list in lists:
    print(len(img_list))
    n_samples = 1000
    random_idxs = np.random.randint(0, len(img_list), n_samples)
    selected_paths = np.array(img_list)[random_idxs]
    print('e.g., ', selected_paths[0])
    # imgs.append(mpimg.imread(img_list[i]))
    features = cl.extract_features(selected_paths,
                                    color_space, 
                                    spatial_size, 
                                    hist_bins,
                                    orient, 
                                    pix_per_cell, 
                                    cell_per_block,
                                    hog_channel,
                                    spatial_feat,
                                    hist_feat,
                                    hog_feat)
    print('len(features)', len(features))
    class_features.append(features)

sw.stop()
print('Time for feature computation: ', sw.format_duration())
print('Feature vector length: ', class_features[0][0].shape)

X = np.vstack((class_features[0], class_features[1])).astype(np.float64)

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(class_features[0])), 
               np.zeros(len(class_features[1]))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

svc = LinearSVC()
sw = Stopwatch()
sw.start()
svc.fit(X_train, y_train)
sw.stop()
print('Time for fitting classifier: ', sw.format_duration())
print('Test accuracy of SVC: ', round(svc.score(X_test, y_test), 4))