import os
import glob

def get_file_list(basedir):
    if not os.path.isdir(basedir):
        raise FileNotFoundError(basedir)
        
    image_types = os.listdir(basedir)
    list = []
    for imtype in image_types:
        subpath = basedir + imtype
        if os.path.isdir(subpath):
            list.extend(glob.glob(subpath + '/*'))
        
    print('Dir: {}, Number Found: {}'.format(basedir, len(list)))
    if not list:
        raise FileNotFoundError('Files were not found. basedir=[{}]'.format(basedir))
    return list
    
def get_file_lists(basedirs):
    lists = []
    for basedir in basedirs:
        lists.append(get_file_list(basedir))
    print('len(lists):', len(lists))
    return lists