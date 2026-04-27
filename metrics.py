import torch
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, binary_erosion
from medpy import metric



def calculate_metrics(pred, target):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Arguments:
        pred: Predicted mask (tensor)
        target: Ground truth mask (tensor)
    
    Returns:
        iou: IoU score (tensor scalar)
        hd95
    """

    pred_binary = (pred > 0.5).float()
    label_binary = (target > 0.5).float()
    pred_binary = pred_binary.cpu().numpy().astype(bool)
    label_binary = label_binary.cpu().numpy().astype(bool)

    intersection = np.logical_and(pred_binary, label_binary)
    union = np.logical_or(pred_binary, label_binary)
    dice = (2.0 * np.sum(intersection)) / (np.sum(pred_binary) + np.sum(label_binary) + 1e-8)

    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    try:
        if np.sum(pred_binary) > 0 and np.sum(label_binary) > 0:
            hd95 = metric.binary.hd95(pred_binary, label_binary)
        else:
            hd95 = np.nan
    except:
        hd95 = np.nan

    return dice, iou, hd95

def SegMetrics(pred, label, metrics):
    metric_list = []
    if isinstance(metrics, str):
        metrics = [metrics, ]
    dice_val, iou_val, hd95_val = calculate_metrics(pred, label)
    lookup = {'dice': dice_val, 'iou': iou_val, 'hd95': hd95_val}
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric in lookup:
            metric_list.append(lookup[metric])
        else:
            raise ValueError('metric %s not recognized' % metric)
    if not metric_list:
        raise ValueError('metric mistakes in calculations')
    return np.array(metric_list)