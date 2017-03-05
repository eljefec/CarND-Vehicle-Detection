import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import classify as cl
import load as ld

#files = ld.get_file_list('vehicles/')
lists = ld.get_file_lists(['vehicles/', 'non-vehicles/'])

while True:
    imgs = []
    for img_list in lists:
        print(len(img_list))
        i = np.random.randint(0, len(img_list))
        imgs.append(mpimg.imread(img_list[i]))
        
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

    plot_images = []
    plot_titles = []

    for img in imgs:
        features, hog_image = cl.single_img_features(img, 
                                                  color_space, 
                                                  spatial_size, 
                                                  hist_bins,
                                                  orient, 
                                                  pix_per_cell, 
                                                  cell_per_block,
                                                  hog_channel,
                                                  spatial_feat,
                                                  hist_feat,
                                                  hog_feat,
                                                  vis)
        plot_images.append(img)
        plot_titles.append('Original')
        plot_images.append(hog_image)
        plot_titles.append('HOG')
        
    fig = plt.figure(figsize=(12, 3))
    cl.visualize(fig, 1, 4, plot_images, plot_titles)