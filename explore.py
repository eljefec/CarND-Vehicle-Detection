import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import classify as cl
import load as ld
import train as tr

lists = ld.get_file_lists(['../data/vehicles/', '../data/non-vehicles/'])

while True:
    imgs = []
    for img_list in lists:
        print(len(img_list))
        i = np.random.randint(0, len(img_list))
        imgs.append(mpimg.imread(img_list[i]))
        
    fp = tr.get_defaults()
    vis = True

    plot_images = []
    plot_titles = []

    for img in imgs:
        fp.hog_channel = 2
        features, hog_image = cl.single_img_features(img, fp, vis)
        plot_images.append(img)
        plot_titles.append('Original')
        plot_images.append(hog_image)
        plot_titles.append('HOG')
        
        fp.hog_channel = 'ALL'
        draw_img, heatmap = cl.full_hog_single_img(img, fp)
        plot_images.append(draw_img)
        plot_titles.append('Full HOG')
        plot_images.append(heatmap)
        plot_titles.append('Full HOG Heatmap')
        
    fig = plt.figure(figsize=(12, 3))
    cl.visualize(fig, 2, 4, plot_images, plot_titles)