import glob
from os.path import join, realpath, dirname, exists, isdir
from os import makedirs
import cv2
import numpy as np
import time

debug = False
debug_period = 1./(60.*10)

# Move the contents of TLP into the train directory

# parses bbox text representation into image masks
base_path = join(realpath(dirname(__file__)), 'train')
video_files = glob.glob(join(base_path, '*'))

for video_file in video_files:
    print("Parsing: "+video_file)
    bbox_file = join(video_file, "groundtruth_rect.txt")
    img_files = sorted(glob.glob(join(video_file, 'img/*.jpg')))

    if not exists(bbox_file):
        print("No groundtruth_rect.txt found for: " + video_file)
        continue
    
    with open(bbox_file) as f:
        bboxes = [[int(x) for x in s.split(',')] for s in f.readlines()]
        
    # Sanity check the data
    try:
        #assert(list(range(len(bboxes))) == [b[0] for b in bboxes])
        assert(len(bboxes) == len(img_files))
    except:
        print("WARNING: not the same number of files as bounding boxes: ")
        print("%d annotations vs. %d imgs"%(len(bboxes), len(img_files)))
    print("Number of images: " + str(len(img_files)))
        
    anno_folder = join(video_file, "annotation")
    if not isdir(anno_folder): makedirs(anno_folder)
    
    for bbox, img_file in zip(bboxes, img_files):
        im = cv2.imread(img_file)
        out_im = np.zeros(im.shape)
        frameID, xmin, ymin, width, height, isLost = bbox

        out_name = "%05d.jpg"%(frameID)

        if not isLost:
            out_im = cv2.rectangle(out_im, (xmin, ymin), (xmin+width, ymin+height), (0,0,128), -1)

        if debug:
            if not isLost:
                im = cv2.rectangle(im, (xmin, ymin), (xmin+width, ymin+height), (0,0,128), 3)
            else:
                im = cv2.rectangle(im, (xmin, ymin), (xmin+width, ymin+height), (128,128,128), 3)
            cv2.imshow("im",im)
            cv2.waitKey(1)
            #time.sleep(debug_period)
            
        cv2.imwrite(join(anno_folder, out_name), out_im)
            
