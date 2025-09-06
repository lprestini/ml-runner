from enum import Enum
import numpy as np
from PIL import Image
import cv2


def save_mask(save_path, mask):
    cv2.imwrite(save_path, mask)

def mask2box(mask):
    if mask is None:
        return None
    mask_bin = mask > 0
    if mask_bin.sum() == 0:
        return None
    x_idxs = np.where(mask_bin.sum(0)>0)[0]
    y_idxs = np.where(mask_bin.sum(1)>0)[0]
    x0 = x_idxs.min()
    x1 = x_idxs.max()
    y0 = y_idxs.min()
    y1 = y_idxs.max()
    bbox = [x0, y0, x1-x0+1, y1-y0+1]
    return bbox

def save_boxes(file_path, bboxes):
    with open(file_path, 'w') as fp:
        for bbox in bboxes:
            if len(bbox) == 4:
                fp.write('%.2f,%.2f,%.2f,%.2f\n' % (bbox[0], bbox[1], bbox[2], bbox[3]))
            else:
                print('Error: Bounding box format must be a list with 4 elements. This one has', len(bbox))
                exit(-1)
        
def load_results_rect(results_path):
    results = []
    with open(results_path, 'r') as f:
        for line in f.readlines():
            results.append([float(el) for el in line.strip().split(',')])
            # results.append([float(el) for el in line.strip().split('\t')])
    return results

class AnnotationType(Enum):
    UNANNOTATED = 0
    ANNOTATED = 1
    IGNORE = 2  # ignore annotation due to weak visible information (annotation is probably poor)
    REFINE = 3  # correct and refine SAM-annotated mask
