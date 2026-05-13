import os
import cv2
import numpy as np

ROOT = "DSAD_clean/multilabel/Masks"

# Organ list (exact names in your filenames)
ORGANS = [
    "abdominal_wall",
    "colon",
    "liver",
    "pancreas",
    "small_intestine",
    "spleen",
    "stomach"
]

# Dictionary to count valid masks
valid_counts = {org: 0 for org in ORGANS}

# Loop through all files in Masks folder
for fname in os.listdir(ROOT):
    fpath = os.path.join(ROOT, fname)

    # Check that this file corresponds to one of the organs
    for organ in ORGANS:
        if organ in fname.lower():      # match organ in filename
            mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print("Could not read:", fpath)
                continue

            # A mask is valid if it contains any white pixel > 0
            if np.any(mask > 0):
                valid_counts[organ] += 1

# Print final results
print("\n===== VALID MASK COUNTS =====")
for organ in ORGANS:
    print(f"{organ:20s} : {valid_counts[organ]}")
