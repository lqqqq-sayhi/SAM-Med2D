"""
SAM-Med2D: 5-fold class-level IOU/Dice in parallel. HD95 from previous run merged in.
"""
import os, sys, glob, time, csv
import numpy as np
from PIL import Image
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd

PRED_BASE = "/mnt/hdd2/task2/sam-med2d/predict"
GT_MASK_DIR = "/mnt/hdd2/task2/sam_lora/test/masks"
OUT_DIR = "/mnt/hdd2/task2/sam-med2d/class_eval_results"
OLD_CSV = os.path.join(OUT_DIR, "sam_med2d_class_level_5fold.csv")
ALL_CLASSES = list(range(1, 29))
ORGAN = {26, 27, 28}
INSTR = set(range(1, 26))
FOLDS = list(range(5))
os.makedirs(OUT_DIR, exist_ok=True)


def get_image_size(pred_dir):
    for f in os.listdir(pred_dir):
        if f.endswith('.png'):
            return Image.open(os.path.join(pred_dir, f)).size[::-1]
    return (1024, 1024)


def evaluate_fold_class_level(fold):
    pred_dir = os.path.join(PRED_BASE, f"fold{fold}_predict", "boxes_prompt")
    if not os.path.exists(pred_dir):
        print(f"Fold {fold}: prediction dir not found: {pred_dir}")
        return None

    H, W = get_image_size(pred_dir)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    img_preds = defaultdict(list)
    for f in pred_files:
        basename = os.path.basename(f).replace(".png", "")
        parts = basename.rsplit("_class", 1)
        if len(parts) != 2:
            continue
        img_preds[parts[0]].append((int(parts[1]), f))

    all_cls_results = []
    for img_key, cls_path_list in img_preds.items():
        gt_files = sorted(glob.glob(os.path.join(GT_MASK_DIR, f"{img_key}_class*.png")))
        if not gt_files:
            continue

        gt_map = np.zeros((H, W), dtype=np.uint8)
        for gf in gt_files:
            cls = int(os.path.basename(gf).replace('.png', '').rsplit('_class', 1)[1])
            gt_map[np.array(Image.open(gf)) > 0] = cls

        pred_map = np.zeros((H, W), dtype=np.uint8)
        for cls_id, pf in cls_path_list:
            try:
                img = np.array(Image.open(pf))
                if img.ndim == 3:
                    img = img[:, :, 0]
                pred_map[img > 0] = cls_id
            except Exception:
                continue

        for cls in ALL_CLASSES:
            gt_bin = (gt_map == cls)
            if not gt_bin.any():
                continue
            pred_bin = (pred_map == cls)
            inter = (pred_bin & gt_bin).sum()
            union = (pred_bin | gt_bin).sum()
            iou = inter / union if union > 0 else 0.0
            dice = 2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-8)
            all_cls_results.append({"class": cls, "iou": iou, "dice": dice})

    cls_list = all_cls_results
    all_iou = [r["iou"] for r in cls_list]
    all_dice = [r["dice"] for r in cls_list]

    return {
        "fold": fold,
        "n_images": len(img_preds),
        "n_rows": len(cls_list),
        "mIoU": float(np.mean(all_iou)),
        "mDice": float(np.mean(all_dice)),
        "Organ mIoU": float(np.mean([r["iou"] for r in cls_list if r["class"] in ORGAN])),
        "Organ mDice": float(np.mean([r["dice"] for r in cls_list if r["class"] in ORGAN])),
        "Instr mIoU": float(np.mean([r["iou"] for r in cls_list if r["class"] in INSTR])),
        "Instr mDice": float(np.mean([r["dice"] for r in cls_list if r["class"] in INSTR])),
    }


# ============================================================
# Load old HD95 values
# ============================================================
old = pd.read_csv(OLD_CSV)
hd95_by_fold = {}
for _, row in old.iterrows():
    f = row["fold"]
    if isinstance(f, str) and f == "mean":
        continue
    hd95_by_fold[int(f)] = {
        "mHD95": row["mHD95"],
        "Organ HD95": row["Organ HD95"],
        "Instr HD95": row["Instr HD95"],
    }
mean_row = old[old["fold"] == "mean"].iloc[0]
old_mean_hd95 = {k: mean_row[k] for k in ["mHD95", "Organ HD95", "Instr HD95"]}

# ============================================================
# Main: parallel 5-fold
# ============================================================
t0 = time.time()
print(f"Starting parallel 5-fold IOU/Dice eval at {time.strftime('%H:%M:%S')}")

with Pool(5) as pool:
    fold_results = [r for r in pool.map(evaluate_fold_class_level, FOLDS) if r is not None]

fold_results.sort(key=lambda r: r["fold"])

for r in fold_results:
    old_hd = hd95_by_fold.get(r["fold"], {})
    r["mHD95"] = old_hd.get("mHD95", float("nan"))
    r["Organ HD95"] = old_hd.get("Organ HD95", float("nan"))
    r["Instr HD95"] = old_hd.get("Instr HD95", float("nan"))

t_elapsed = (time.time() - t0) / 60

# Print summary
print(f"\n{'='*110}")
print("SAM-Med2D Class-Level 5-Fold Summary (parallel, {:.1f} min total)".format(t_elapsed))
print(f"{'='*110}")
h = f"{'Fold':<8} {'mIoU':>10} {'mDice':>10} {'mHD95':>10} {'Org mIoU':>10} {'Org mDice':>10} {'Org HD95':>10} {'Ins mIoU':>10} {'Ins mDice':>10} {'Ins HD95':>10}"
print(h)
for r in fold_results:
    print(f"{r['fold']:<8} {r['mIoU']:>10.4f} {r['mDice']:>10.4f} {r['mHD95']:>10.2f} {r['Organ mIoU']:>10.4f} {r['Organ mDice']:>10.4f} {r['Organ HD95']:>10.2f} {r['Instr mIoU']:>10.4f} {r['Instr mDice']:>10.4f} {r['Instr HD95']:>10.2f}")

avg = lambda key: np.mean([r[key] for r in fold_results if not np.isnan(r[key])])
avg_iou = avg("mIoU")
avg_dice = avg("mDice")
avg_org_iou = avg("Organ mIoU")
avg_org_dice = avg("Organ mDice")
avg_ins_iou = avg("Instr mIoU")
avg_ins_dice = avg("Instr mDice")

print(f"{'Mean':<8} {avg_iou:>10.4f} {avg_dice:>10.4f} {old_mean_hd95['mHD95']:>10.2f} {avg_org_iou:>10.4f} {avg_org_dice:>10.4f} {old_mean_hd95['Organ HD95']:>10.2f} {avg_ins_iou:>10.4f} {avg_ins_dice:>10.4f} {old_mean_hd95['Instr HD95']:>10.2f}")

# Save CSV
csv_path = os.path.join(OUT_DIR, "sam_med2d_class_level_5fold.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["fold", "n_images", "n_rows",
                "Mean IOU", "Mean Dice", "Mean HD95",
                "(Organ) Mean IOU", "(Organ) Mean Dice", "(Organ) Mean HD95",
                "(Instr) Mean IOU", "(Instr) Mean Dice", "(Instr) Mean HD95"])
    for r in fold_results:
        w.writerow([r["fold"], r["n_images"], r["n_rows"],
                    r["mIoU"], r["mDice"], r["mHD95"],
                    r["Organ mIoU"], r["Organ mDice"], r["Organ HD95"],
                    r["Instr mIoU"], r["Instr mDice"], r["Instr HD95"]])
    w.writerow(["mean", "", "",
                avg_iou, avg_dice, old_mean_hd95["mHD95"],
                avg_org_iou, avg_org_dice, old_mean_hd95["Organ HD95"],
                avg_ins_iou, avg_ins_dice, old_mean_hd95["Instr HD95"]])

print(f"\nSaved to {csv_path}")
print("Done!")
