import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# =====================================================
# CONFIG
# =====================================================
SOURCE = r"E:/Python/OEP_Anatomy/DSAD_clean/multilabel"
TARGET = r"E:/Python/OEP_Anatomy_Final/DATASET"
TARGET_SIZE = (512, 512)

SPLITS = {
    "TRAIN": 0.80,
    "VALIDATION": 0.15,
    "TEST": 0.05
}

ORGANS = {
    "abdominal_wall": 1,
    "colon": 2,
    "liver": 3,
    "pancreas": 4,
    "small_intestine": 5,
    "spleen": 6,
    "stomach": 7,
}

random.seed(42)

# =====================================================
# CREATE FOLDER STRUCTURE
# =====================================================
for split in SPLITS:
    os.makedirs(f"{TARGET}/{split}/IMAGE", exist_ok=True)
    os.makedirs(f"{TARGET}/{split}/MASK", exist_ok=True)

# =====================================================
# HELPERS
# =====================================================
def resize_image(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

def resize_mask(mask):
    return cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

# =====================================================
# LOAD FILE LIST
# =====================================================
img_dir = os.path.join(SOURCE, "Images")
mask_dir = os.path.join(SOURCE, "Masks")

img_files = sorted([
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

random.shuffle(img_files)

N = len(img_files)
n_train = int(N * SPLITS["TRAIN"])
n_val   = int(N * SPLITS["VALIDATION"])

train_files = img_files[:n_train]
val_files   = img_files[n_train:n_train + n_val]
test_files  = img_files[n_train + n_val:]

SPLIT_FILES = {
    "TRAIN": train_files,
    "VALIDATION": val_files,
    "TEST": test_files
}

# =====================================================
# BUILD DATASET
# =====================================================
print("\n🚀 Building Dataset with Required Folder Structure...\n")

for split, files in SPLIT_FILES.items():
    print(f"Processing {split} set ({len(files)} images)")

    for img_name in tqdm(files):
        base = os.path.splitext(img_name)[0]

        # ---- Load image ----
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping corrupted image: {img_name}")
            continue

        H, W = img.shape[:2]
        final_mask = np.zeros((H, W), dtype=np.uint8)  # background = 0

        # ---- Merge organ masks ----
        for organ, cid in ORGANS.items():
            mask_path = os.path.join(mask_dir, f"{base}_{organ}.png")

            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            final_mask[mask > 0] = cid

        # ---- Resize ----
        img_resized = resize_image(img)
        mask_resized = resize_mask(final_mask)

        # ---- Save ----
        cv2.imwrite(f"{TARGET}/{split}/IMAGE/{img_name}", img_resized)
        cv2.imwrite(f"{TARGET}/{split}/MASK/{img_name}", mask_resized)

print("\n🎉 DONE!")
print("Final dataset structure created:")
print("DATASET/TRAIN/IMAGE & MASK")
print("DATASET/VALIDATION/IMAGE & MASK")
print("DATASET/TEST/IMAGE & MASK\n")