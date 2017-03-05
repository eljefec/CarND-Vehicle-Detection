import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import classify as cl
import load as ld
from Stopwatch import Stopwatch 

#files = ld.get_file_list('vehicles/')
lists = ld.get_file_lists(['non-vehicles/', 'vehicles/'])

color_space='RGB', 
spatial_size=(16, 16)
hist_bins = 16
orient=6
pix_per_cell=8
cell_per_block=2
hog_channel=0
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