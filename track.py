from collections import deque
import moviepy
from moviepy.editor import VideoFileClip
import numpy as np
from scipy.ndimage.measurements import label

import search as sr
import train as tr

class Frame:
    def __init__(self, heatmap, label_boxes):
        self.heatmap = heatmap
        self.label_boxes = label_boxes
        
class Tracker:
    def __init__(self, searcher, search_params, window_size, heatmap_threshold_per_frame):
        self.searcher = searcher
        self.search_params = search_params
        self.window_size = window_size
        self.heatmap_threshold_per_frame = heatmap_threshold_per_frame
        self.frames = deque()
        
    def track(self, img):
        self.add_frame(searcher.search(img, 'full', self.search_params))
        
        heatmap, label_boxes = self.smooth()
        labeled_img = sr.draw_boxes(img, label_boxes)
        return labeled_img
        
    def add_frame(self, search_result):
        heatmap = search_result[0]
        label_boxes = search_result[1]
        self.frames.appendleft(Frame(heatmap, label_boxes))
        
        if len(self.frames) > self.window_size:
            discard_frame = self.frames.pop()

    def smooth(self):
        heatmaps = []
        for f in self.frames:
            heatmaps.append(f.heatmap)
        heatmap = np.sum(heatmaps, axis = 0)
        heatmap = sr.apply_threshold(heatmap, self.heatmap_threshold_per_frame * len(self.frames))
        labels = label(heatmap)
        label_boxes = sr.convert_to_bboxes(labels)
        return heatmap, label_boxes
            
def process_img(img):
    global tracker
    return tracker.track(img)
    
def process_video(input_fname, output_fname):
    input_clip = VideoFileClip(input_fname)
    output_clip = input_clip.fl_image(process_img)
    output_clip.write_videofile(output_fname, audio=False)
        
if __name__ == '__main__':
    # model = 'trained_models/HSV-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.49.p'
    # model = 'trained_models/HLS-ss(16, 16)-hb16-o9-p10-c2-hcALL-sf1-hist1-hog1-acc99.32.p'
    # model = 'trained_models/YCrCb-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.21.p'
    model = 'trained_models/YCrCb-ss(16, 16)-hb16-o9-p8-c2-hcALL-sf1-hist1-hog1-acc99.72.p'
    (fp, clf, X_scaler) = tr.load_classifier(model)
    
    searcher = sr.Searcher(fp, clf, X_scaler)    
    
    sp = sr.SearchParams.get_defaults()
    
    for threshold in [1, 2, 3]:
        print('Threshold: ', threshold)
        tracker = Tracker(searcher, sp, 5, heatmap_threshold_per_frame = threshold)
        process_video('test_video.mp4', 'output_video/test_video_th{}.mp4'.format(threshold))
    
    for threshold in [1, 2, 3]:
        print('Threshold: ', threshold)    
        tracker = Tracker(searcher, sp, 5, heatmap_threshold_per_frame = threshold)
        process_video('project_video.mp4', 'output_video/project_video_th{}.mp4'.format(threshold))