import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from collections import defaultdict

from dataset import SurgicalSegmentationDataset
from train_unet_baseline import UNet

# -------------------------
# CONFIG
# -------------------------

DATASET_ROOT = "DATASET"
CHECKPOINT_PATH = r"checkpoints/best_unet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 6
BATCH_SIZE = 1

SAVE_DIR = "test_results"
VIS_DIR = os.path.join(SAVE_DIR, "qualitative")
os.makedirs(VIS_DIR, exist_ok=True)

CLASS_NAMES = {
    0: "Background",
    1: "Organ-1",
    2: "Organ-2",
    3: "Organ-3",
    4: "Organ-4",
    5: "Small-organs"
}

# -------------------------
# METRICS
# -------------------------

def dice_score(pred, target):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return None
    return (2 * inter / (union + 1e-6)).item()

def iou_score(pred, target):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    if union == 0:
        return None
    return (inter / (union + 1e-6)).item()

# -------------------------
# LOAD MODEL
# -------------------------

model = UNet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# LOAD TEST DATA
# -------------------------

test_ds = SurgicalSegmentationDataset(DATASET_ROOT, "TEST")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("\n====================================================")
print("TEST SET EVALUATION STARTED")
print(f"Total test samples: {len(test_ds)}")
print("====================================================\n")

# -------------------------
# METRIC ACCUMULATORS
# -------------------------

dice_per_class = defaultdict(list)
iou_per_class  = defaultdict(list)

# -------------------------
# EVALUATION LOOP
# -------------------------

with torch.no_grad():
    for idx, (img, mask) in enumerate(test_loader):
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        logits = model(img)
        pred = torch.argmax(logits, dim=1)

        for cls in range(1, NUM_CLASSES):  # ignore background
            pred_cls = (pred == cls).float()
            tgt_cls  = (mask == cls).float()

            d = dice_score(pred_cls, tgt_cls)
            i = iou_score(pred_cls, tgt_cls)

            if d is not None:
                dice_per_class[cls].append(d)
            if i is not None:
                iou_per_class[cls].append(i)

        # ---------- SAVE QUALITATIVE ----------
        if idx < 10:
            img_np = img[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            mask_np = mask[0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()

            overlay = img_np.copy()
            overlay[pred_np > 0] = [1, 0, 0]

            mask_vis = (mask_np * 40).astype(np.uint8)
            pred_vis = (pred_np * 40).astype(np.uint8)

            out = np.hstack([
                (img_np * 255).astype(np.uint8),
                cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
            ])


            cv2.imwrite(
                os.path.join(VIS_DIR, f"sample_{idx:03d}.png"),
                out
            )

# -------------------------
# PRINT METRICS TABLE
# -------------------------

print("\nPer-Class Test Metrics")
print("----------------------------------------------------")
print("Class | Dice Score | IoU Score")
print("----------------------------------------------------")

mean_dice = []
mean_iou  = []

for cls in range(1, NUM_CLASSES):
    d = np.mean(dice_per_class[cls]) if dice_per_class[cls] else 0.0
    i = np.mean(iou_per_class[cls])  if iou_per_class[cls]  else 0.0

    mean_dice.append(d)
    mean_iou.append(i)

    print(f"{CLASS_NAMES[cls]:<12} | {d:.4f}     | {i:.4f}")

# -------------------------
# MEAN METRICS
# -------------------------

print("\n----------------------------------------------------")
print("Mean Test Metrics")
print("----------------------------------------------------")
print(f"Mean Dice : {np.mean(mean_dice):.4f}")
print(f"Mean IoU  : {np.mean(mean_iou):.4f}")

# -------------------------
# STRONG / WEAK ANALYSIS
# -------------------------

print("\n----------------------------------------------------")
print("Class-wise Performance Interpretation")
print("----------------------------------------------------")

for cls in range(1, NUM_CLASSES):
    d = np.mean(dice_per_class[cls]) if dice_per_class[cls] else 0.0

    if d >= 0.45:
        status = "STRONG"
    elif d >= 0.30:
        status = "MODERATE"
    else:
        status = "WEAK"

    print(f"{CLASS_NAMES[cls]:<12} → {status} (Dice = {d:.3f})")

# -------------------------
# FINAL CONCLUSION
# -------------------------

print("\n====================================================")
print("FINAL TEST CONCLUSION")
print("====================================================")

if np.mean(mean_dice) >= 0.35:
    print(
        "The U-Net baseline generalizes well on the test set.\n"
        "Most major anatomical structures are segmented reliably.\n"
        "Remaining errors are concentrated around small or visually ambiguous organs.\n"
        "This confirms the baseline is strong and suitable for architectural upgrades."
    )
else:
    print(
        "Test performance indicates limited generalization.\n"
        "Further improvements in architecture or data handling are required\n"
        "before moving to advanced models."
    )

print("\nQualitative predictions saved to:", VIS_DIR)
print("====================================================\n")