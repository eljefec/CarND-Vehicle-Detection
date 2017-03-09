import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import classify as cl
import train as tr

if __name__ == '__main__':
    fp = tr.get_defaults()
    orig = mpimg.imread('test_images/test1.jpg')
    img = cl.convert_color(orig, fp)
    hist = cl.color_hist(img, fp.hist_bins)
    
    spatial = cl.bin_spatial(img, fp.spatial_size)

    plt.subplot(2, 2, 1)
    plt.imshow(orig)
    plt.title('Original')
    
    plt.subplot(2, 2, 2)
    plt.imshow(img)
    plt.title('YCrCb')
    
    plt.subplot(2, 2, 3)
    plt.plot(hist)
    plt.title('Color Histogram')
    
    plt.subplot(2, 2, 4)
    plt.plot(spatial)
    plt.title('Spatial Histogram')
    
    plt.show()