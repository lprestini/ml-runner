import os
import cv2
import numpy as np
from vot.region import RegionType
from vot.region.raster import calculate_overlaps


def keep_largest_component(mask):
    """
    Keeps only the largest connected component from a binary mask.
    
    Args:
    - mask (numpy array): 2D binary mask where object pixels are non-zero and background is 0.
    
    Returns:
    - filtered_mask (numpy array): Binary mask with only the largest connected component.
    """
    # Perform connected components analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Find the index of the largest component (excluding background)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background (index 0)
    # Create a mask that contains only the largest component
    filtered_mask = np.zeros_like(mask)
    filtered_mask[labels == largest_component] = 1
    return filtered_mask

def determine_tracker(tracker_name):
    path_ = os.path.abspath(os.path.dirname(__file__))
    if tracker_name == "sam21pp-L":
        checkpoint = os.path.join(path_, "../checkpoints/sam2.1_hiera_large.pt")
        model_cfg = "sam21pp_hiera_l.yaml"
    elif tracker_name == "sam21pp-B":
        checkpoint = os.path.join(path_, "../checkpoints/sam2.1_hiera_base_plus.pt")
        model_cfg = "sam21pp_hiera_b+.yaml"
    elif tracker_name == "sam21pp-S":
        checkpoint = os.path.join(path_, "../sam2.1_hiera_small.pt")
        model_cfg = "sam21pp_hiera_s.yaml"
    elif tracker_name == "sam21pp-T":
        checkpoint = os.path.join(path_, "../sam2.1_hiera_tiny.pt")
        model_cfg = "sam21pp_hiera_t.yaml"
    elif tracker_name == "sam2pp-L":
        checkpoint = os.path.join(path_, "../sam2_hiera_large.pt")
        model_cfg = "sam2pp_hiera_l.yaml"
    elif tracker_name == "sam2pp-B":
        checkpoint = os.path.join(path_, "../sam2_hiera_base_plus.pt")
        model_cfg = "sam2pp_hiera_b+.yaml"
    elif tracker_name == "sam2pp-S":
        checkpoint = os.path.join(path_, "../sam2_hiera_small.pt")
        model_cfg = "sam2pp_hiera_s.yaml"
    elif tracker_name == "sam2pp-T":
        checkpoint = os.path.join(path_, "../sam2_hiera_tiny.pt")
        model_cfg = "sam2pp_hiera_t.yaml"
    return checkpoint, model_cfg

def get_seq_names(dataset_path):
    list_path = os.path.join(dataset_path, 'list.txt')
    with open(list_path, 'r') as f:
        lines = f.readlines()
    seq_names = [line.strip() for line in lines]
    return seq_names

def compute_seq_perf(pred_masks_, gt, bounds, sequence_name):
    pred_masks = pred_masks_.copy()
    # bounds: tuple (width, height)
    # convert gt to bboxes:
    gt = gt[1:]
    pred_masks = pred_masks[1:]

    for i in range(len(gt)):
        gt_ = gt[i]
        if gt_.type is not RegionType.SPECIAL and not gt_.is_empty():
            if gt_ != RegionType.RECTANGLE:
                gt[i] = gt_.convert(RegionType.RECTANGLE)
                pred_masks[i] = pred_masks[i].convert(RegionType.RECTANGLE)    

    overlaps = calculate_overlaps(pred_masks, gt, bounds)
    overlaps_arr = np.array(overlaps)

    avg_overlap = overlaps_arr.mean()
    robustness = (overlaps_arr>0).sum() / len(gt)
    
    print('--------------------------------')
    print('Performance on %s:' % sequence_name)
    print('Average overlap: %.3f' % (avg_overlap))
    print('Robustness: %.2f' % (robustness))
    print('--------------------------------')

    return (sequence_name, avg_overlap, robustness)
