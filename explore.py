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
    fp.hog_channel = 2
    vis = True

    plot_images = []
    plot_titles = []

    for img in imgs:
        features, hog_image = cl.single_img_features(img, fp, vis)
        plot_images.append(img)
        plot_titles.append('Original')
        plot_images.append(hog_image)
        plot_titles.append('HOG')
        
    fig = plt.figure(figsize=(12, 3))
    cl.visualize(fig, 1, 4, plot_images, plot_titles)