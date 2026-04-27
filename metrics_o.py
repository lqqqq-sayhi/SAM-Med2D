import torch
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, binary_erosion

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()


def hd95(pr, gt, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    
    pr_np = pr_.cpu().numpy()
    gt_np = gt_.cpu().numpy()
    
    batch_size = pr_np.shape[0]
    hd95_values = np.zeros(batch_size)
    
    for b in range(batch_size):
        pred_mask = pr_np[b, 0].astype(bool)
        gt_mask = gt_np[b, 0].astype(bool)
        
        if not pred_mask.any() or not gt_mask.any():
            hd95_values[b] = np.nan
            continue
        
        pred_edges = pred_mask ^ binary_erosion(pred_mask)
        gt_edges = gt_mask ^ binary_erosion(gt_mask)
        
        if not pred_edges.any() or not gt_edges.any():
            hd95_values[b] = np.nan
            continue
        
        dist_pred_to_gt = distance_transform_edt(~gt_edges)
        dist_gt_to_pred = distance_transform_edt(~pred_edges)
        
        surface_pred = dist_pred_to_gt[pred_edges]
        surface_gt = dist_gt_to_pred[gt_edges]
        
        all_distances = np.concatenate([surface_pred, surface_gt])
        hd95_values[b] = np.percentile(all_distances, 95)
    
    return hd95_values


def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'hd95':
            hd95_val = hd95(pred, label)
            metric_list.append(np.nanmean(hd95_val))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric