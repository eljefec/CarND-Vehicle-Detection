import moviepy
from moviepy.editor import VideoFileClip
import search as sr
import train as tr

class Tracker:
    def __init__(self, searcher, window_size, search_params):
        self.searcher = searcher
        self.window_size = window_size
        self.search_params = search_params
        
    def track(self, img):
        ### TODO: Implement tracking.
        heatmap, label_boxes, window_count = searcher.search(img, 'full', self.search_params)
        labeled_img = sr.draw_boxes(img, label_boxes)
        return labeled_img

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
    tracker = Tracker(searcher, 5, sp)
    
    process_video('test_video.mp4', 'output_video/test_video.mp4')
    #process_video('project_video.mp4', 'output_video/project_video.mp4')