import os
import cv2
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

DATASET_ROOT = "DATASET"
SPLITS = ["TRAIN", "VALIDATION", "TEST"]

def get_image_mask_pairs(split):
    img_dir = os.path.join(DATASET_ROOT, split, "IMAGE")
    mask_dir = os.path.join(DATASET_ROOT, split, "MASK")

    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    img_set = set(images)
    mask_set = set(masks)

    missing_masks = img_set - mask_set
    extra_masks = mask_set - img_set

    pairs = list(img_set & mask_set)

    return img_dir, mask_dir, pairs, missing_masks, extra_masks


def dataset_audit():
    print("\n========== DATASET AUDIT ==========")

    audit_results = {}

    for split in SPLITS:
        img_dir, mask_dir, pairs, missing, extra = get_image_mask_pairs(split)

        print(f"\n[{split}]")
        print(f"Total image-mask pairs: {len(pairs)}")

        if missing:
            print(f"❌ Missing masks for {len(missing)} images")
        else:
            print("✅ No missing masks")

        if extra:
            print(f"❌ Extra masks without images: {len(extra)}")
        else:
            print("✅ No extra masks")

        audit_results[split] = {
            "count": len(pairs),
            "missing": len(missing),
            "extra": len(extra)
        }

    return audit_results


def analyze_masks_and_images():
    print("\n========== IMAGE & MASK ANALYSIS ==========")

    class_pixel_counts = defaultdict(int)
    resolutions = defaultdict(list)
    invalid_masks = []

    for split in SPLITS:
        img_dir, mask_dir, pairs, _, _ = get_image_mask_pairs(split)

        for fname in tqdm(pairs, desc=f"Analyzing {split}"):
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            if img is None or mask is None:
                invalid_masks.append(fname)
                continue

            if img.shape[:2] != mask.shape[:2]:
                invalid_masks.append(fname)
                continue

            if len(mask.shape) != 2:
                invalid_masks.append(fname)
                continue

            resolutions[split].append(img.shape[:2])

            unique_vals, counts = np.unique(mask, return_counts=True)
            for v, c in zip(unique_vals, counts):
                class_pixel_counts[int(v)] += int(c)

    return class_pixel_counts, resolutions, invalid_masks


def visualize_samples(num_samples=3):
    print("\n========== MASK VISUALIZATION ==========")

    for split in SPLITS:
        img_dir, mask_dir, pairs, _, _ = get_image_mask_pairs(split)

        if len(pairs) == 0:
            continue

        samples = random.sample(pairs, min(num_samples, len(pairs)))

        for fname in samples:
            img = cv2.imread(os.path.join(img_dir, fname))
            mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)

            overlay = img.copy()
            overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)

            plt.figure(figsize=(12, 4))
            plt.suptitle(f"{split} : {fname}")

            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Mask")
            plt.imshow(mask, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Overlay")
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.show()


def interpret_results(audit, class_pixels, resolutions, invalid_masks):
    print("\n========== DATA-DRIVEN INTERPRETATION ==========")

    total_pixels = sum(class_pixels.values())

    print("\nClass Distribution (by pixel percentage):")
    for cls, count in sorted(class_pixels.items()):
        pct = (count / total_pixels) * 100
        print(f"Class {cls}: {pct:.2f}%")

    if len(class_pixels) <= 2:
        print("\nInterpretation:")
        print("- Dataset appears to be binary segmentation.")
    else:
        print("\nInterpretation:")
        print("- Dataset is multi-class; class imbalance handling is required.")

    dominant_class = max(class_pixels, key=class_pixels.get)
    dominance_ratio = class_pixels[dominant_class] / total_pixels

    if dominance_ratio > 0.9:
        print("- Strong class imbalance detected.")
        print("  Recommendation: Use Dice or Dice + CE loss.")

    print("\nResolution consistency:")
    for split, res_list in resolutions.items():
        unique_res = set(res_list)
        if len(unique_res) == 1:
            print(f"- {split}: consistent resolution {unique_res.pop()}")
        else:
            print(f"- {split}: multiple resolutions detected {unique_res}")
            print("  Recommendation: resize or crop during preprocessing.")

    if invalid_masks:
        print(f"\n❌ Found {len(invalid_masks)} invalid image-mask pairs.")
        print("  Training should NOT begin until this is fixed.")
    else:
        print("\n✅ All masks passed structural validation.")

    print("\nFinal readiness assessment:")
    if invalid_masks or any(v["missing"] > 0 or v["extra"] > 0 for v in audit.values()):
        print("❌ Dataset is NOT training-ready.")
    else:
        print("✅ Dataset is TRAINING-READY.")


if __name__ == "__main__":
    audit = dataset_audit()
    class_pixels, resolutions, invalid_masks = analyze_masks_and_images()
    visualize_samples(num_samples=2)
    interpret_results(audit, class_pixels, resolutions, invalid_masks)