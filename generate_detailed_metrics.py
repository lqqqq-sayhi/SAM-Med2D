import os
import glob
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from medpy import metric
import warnings

warnings.filterwarnings("ignore")

def calculate_metrics(pred_mask, gt_mask):
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)
    
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    
    iou = float(inter / union) if union > 0 else 0.0
    dice = (2.0 * inter) / (pred_b.sum() + gt_b.sum() + 1e-8)
    precision = float(inter / pred_b.sum()) if pred_b.sum() > 0 else 0.0
    recall = float(inter / gt_b.sum()) if gt_b.sum() > 0 else 0.0
    
    try:
        hd95 = metric.binary.hd95(pred_b, gt_b) if pred_b.any() and gt_b.any() else float("nan")
    except:
        hd95 = float("nan")
        
    return float(iou), float(dice), float(hd95), float(precision), float(recall)

def generate_per_image_metrics(fold):
    pred_dir = f"/mnt/hdd2/task2/sam-med2d/predict/fold{fold}_predict/boxes_prompt"
    gt_dir = "/mnt/hdd2/task2/sam_lora/test/masks"
    output_csv = f"/mnt/hdd2/task2/sam-med2d/predict/fold{fold}_predict_per_image_detailed_metrics.csv"
    
    if not os.path.exists(pred_dir):
        print(f"Skipping Fold {fold}: Directory {pred_dir} not found.")
        return
        
    pred_files = glob.glob(os.path.join(pred_dir, "*.png"))
    if not pred_files:
        print(f"Skipping Fold {fold}: No .png files found in {pred_dir}")
        return
        
    metrics_list = []
    print(f"Processing Fold {fold} ({len(pred_files)} files)...")
    
    for pred_path in tqdm(sorted(pred_files)):
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, filename)
        
        try:
            base_name, cls_str = filename.rsplit("_class", 1)
            cls_id = int(cls_str.replace(".png", ""))
        except Exception:
            continue
            
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            continue
            
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                gt_mask = np.zeros_like(pred_mask)
        else:
            gt_mask = np.zeros_like(pred_mask)
            
        iou, dice, hd95, precision, recall = calculate_metrics(pred_mask > 0, gt_mask > 0)
        
        metrics_list.append({
            "filename": base_name + ".png",
            "class": cls_id,
            "IOU": iou,
            "Dice": dice,
            "HD95": hd95,
            "Precision": precision,
            "Recall": recall
        })

    if not metrics_list:
        print("No metrics collected.")
        return

    df = pd.DataFrame(metrics_list)
    
    # Compute image-level average (class=-1)
    avg_df = df.groupby("filename").mean(numeric_only=True).reset_index()
    avg_df["class"] = -1
    
    final_df = pd.concat([df, avg_df], ignore_index=True)
    final_df.sort_values(by=["filename", "class"], inplace=True)
    
    final_df.to_csv(output_csv, index=False)
    print(f"Saved -> {output_csv}")

if __name__ == "__main__":
    for fold in range(5):
        generate_per_image_metrics(fold)
