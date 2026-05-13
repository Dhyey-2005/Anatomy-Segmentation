import os
import cv2
import numpy as np
from tqdm import tqdm

DATASET_ROOT = "DSAD_clean"

# Organ names expected in multilabel
MULTI_ORGANS = [
    "abdominal_wall",
    "colon",
    "liver",
    "pancreas",
    "small_intestine",
    "spleen",
    "stomach"
]

def check_multilabel(anatomy_path):
    img_dir = os.path.join(anatomy_path, "Images")
    msk_dir = os.path.join(anatomy_path, "Masks")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    print(f"\nChecking MULTILABEL anatomy: {anatomy_path}")
    errors = 0
    correct = 0

    for img_name in tqdm(img_files):
        img_path = os.path.join(img_dir, img_name)

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Cannot open image → {img_name}")
            errors += 1
            continue

        # For each organ mask
        base = os.path.splitext(img_name)[0]

        for organ in MULTI_ORGANS:
            mask_name = f"{base}_{organ}.png"
            mask_path = os.path.join(msk_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"❌ Missing mask: {mask_name}")
                errors += 1
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"❌ Cannot read mask: {mask_name}")
                errors += 1
                continue

            # Check resolution
            if mask.shape != img.shape[:2]:
                print(f"❌ Size mismatch → {img_name} vs {mask_name}")
                errors += 1
                continue

            # Check binary values
            uniq = set(np.unique(mask))
            if not uniq.issubset({0, 255}):
                print(f"❌ Invalid pixel values {uniq} in {mask_name}")
                errors += 1
                continue

        correct += 1

    print("\n--------------------------")
    print(" MULTILABEL CHECK RESULT")
    print("--------------------------")
    print(f"✔ Correct images: {correct}")
    print(f"❌ Errors: {errors}\n")


def check_single_label(anatomy_path):
    img_dir = os.path.join(anatomy_path, "Images")
    msk_dir = os.path.join(anatomy_path, "Masks")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    msk_files = sorted([f for f in os.listdir(msk_dir) if f.endswith(".png")])

    if len(img_files) != len(msk_files):
        print(f"❌ Count mismatch in {anatomy_path}")
        return

    print(f"\nChecking anatomy: {os.path.basename(anatomy_path)}")
    errors = 0
    correct = 0

    for img_name, msk_name in tqdm(zip(img_files, msk_files), total=len(img_files)):
        img_path = os.path.join(img_dir, img_name)
        msk_path = os.path.join(msk_dir, msk_name)

        if img_name != msk_name:
            print(f"❌ Name mismatch → {img_name} vs {msk_name}")
            errors += 1
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"❌ Cannot open {img_name} or {msk_name}")
            errors += 1
            continue

        if img.shape[:2] != mask.shape:
            print(f"❌ Size mismatch → {img_name}")
            errors += 1
            continue

        uniq = set(np.unique(mask))
        if not uniq.issubset({0, 255}):
            print(f"❌ Invalid mask values {uniq} in {msk_name}")
            errors += 1
            continue

        correct += 1

    print("\n--------------------------")
    print(" CHECK RESULT:", os.path.basename(anatomy_path))
    print("--------------------------")
    print(f"✔ Correct masks: {correct}")
    print(f"❌ Errors found: {errors}\n")


if __name__ == "__main__":
    for anatomy in os.listdir(DATASET_ROOT):
        anatomy_path = os.path.join(DATASET_ROOT, anatomy)

        if not os.path.isdir(anatomy_path):
            continue

        if anatomy == "multilabel":
            check_multilabel(anatomy_path)
        else:
            check_single_label(anatomy_path)

    print("\n🎉 Dataset check completed!")