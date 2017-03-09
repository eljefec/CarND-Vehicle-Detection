import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import search as sr

if __name__ == '__main__':
    img = mpimg.imread('test_images/test1.jpg')
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    search_params = sr.SearchParams.get_defaults()
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    
    for sp, color in zip(search_params, colors):
        overlap = sp.cells_per_step / 8
        
        window = int(sp.scale * 64)
        xy_window = (window, window)
        
        windows = sr.slide_window(img, x_start_stop = [None, None], y_start_stop = sp.y_start_stop, 
                        xy_window = xy_window, xy_overlap  = (overlap, overlap))
                        
        draw_img = sr.draw_boxes(draw_img, windows, color, thick=2)
        
    plt.imshow(draw_img)
    plt.title('Sliding windows')
    plt.show()