import os
import cv2

# paths
img_dir = "DSAD_clean/multilabel/Images"
mask_dir = "DSAD_clean/multilabel/Masks"

# list all image files
img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
mask_files = set([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

total_images = len(img_files)
total_masks = len(mask_files)

missing_masks = []
corrupted_masks = []

for file in img_files:
    img_name = file
    mask_name = file  # mask should have same name

    mask_path = os.path.join(mask_dir, mask_name)

    # Case 1: Mask file missing
    if mask_name not in mask_files:
        missing_masks.append(mask_name)
        continue

    # Case 2: Mask exists but is corrupted / unreadable
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        corrupted_masks.append(mask_name)

# summary
print("====================================")
print("     MULTILABEL FOLDER SUMMARY      ")
print("====================================")
print(f"Total images:   {total_images}")
print(f"Total masks:    {total_masks}")
print(f"Missing masks:  {len(missing_masks)}")
print(f"Corrupted masks:{len(corrupted_masks)}")
print("------------------------------------")
print("Missing mask files:")
print(missing_masks[:50], "...") if len(missing_masks) > 50 else print(missing_masks)
print("------------------------------------")
print("Corrupted mask files:")
print(corrupted_masks[:50], "...") if len(corrupted_masks) > 50 else print(corrupted_masks)
print("====================================")
