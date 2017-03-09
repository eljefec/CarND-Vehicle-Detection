import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import classify as cl
import load as ld
import train as tr

lists = ld.get_file_lists(['../data/vehicles/', '../data/non-vehicles/'])

model = 'trained_models/YCrCb-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.72.p'
(fp, clf, X_scaler) = tr.load_classifier(model)
    

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

    for img, label in zip(imgs, ['Car', 'Not Car']):
        plot_images.append(img)
        plot_titles.append(label + ' Original')
        for hog_channel in [0, 1, 2]:
            fp.hog_channel = hog_channel
            features, hog_image = cl.single_img_features(img, fp, vis)
            plot_images.append(hog_image)
            plot_titles.append(label + ' HOG Ch-' + str(hog_channel))
        
        #fp.hog_channel = 'ALL'
        #(hot_windows, window_count) = cl.full_hog_single_img(img, fp, clf, X_scaler, 1, (0, img.shape[0]), 2)
        #plot_images.append(draw_img)
        #plot_titles.append('Full HOG')
        #plot_images.append(heatmap)
        #plot_titles.append('Full HOG Heatmap')
        
    fig = plt.figure(figsize=(12, 3))
    cl.visualize(fig, 2, 4, plot_images, plot_titles)