import os
import shutil

SOURCE_ROOT = "DSAD"
TARGET_ROOT = "DSAD_clean"

ANATOMIES = [
    "abdominal_wall",
    "colon",
    "inferior_mesenteric_artery",
    "intestinal_veins",
    "liver",
    "multilabel",
    "pancreas",
    "small_intestine",
    "spleen",
    "stomach",
    "ureter",
    "vesicular_glands"
]

# -------------------------------------------------------
# MULTILABEL ORGAN ORDER (to rename masks consistently)
# -------------------------------------------------------
MULTI_ORGANS = [
    "abdominal_wall",
    "colon",
    "liver",
    "pancreas",
    "small_intestine",
    "spleen",
    "stomach"
]

# -------------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------------

for anatomy in ANATOMIES:
    source_anatomy_path = os.path.join(SOURCE_ROOT, anatomy)

    if not os.path.isdir(source_anatomy_path):
        print(f"WARNING: Missing anatomy folder → {anatomy}")
        continue

    target_images = os.path.join(TARGET_ROOT, anatomy, "Images")
    target_masks  = os.path.join(TARGET_ROOT, anatomy, "Masks")

    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_masks, exist_ok=True)

    print(f"\nProcessing anatomy: {anatomy}")

    count = 0

    # Loop through X directories
    for Xfolder in sorted(os.listdir(source_anatomy_path)):
        Xpath = os.path.join(source_anatomy_path, Xfolder)

        if not os.path.isdir(Xpath):
            continue

        # Loop through PNG images
        for file in sorted(os.listdir(Xpath)):
            if not file.lower().endswith(".png"):
                continue

            if "mask" in file.lower():
                continue

            img_path = os.path.join(Xpath, file)

            img_num = ''.join(filter(str.isdigit, file))
            new_base = f"img{count:03d}"

            # ----------------------------------------------------
            # SINGLE-LABEL ANATOMIES (normal folders)
            # ----------------------------------------------------
            if anatomy != "multilabel":
                mask_path = os.path.join(Xpath, "merged", f"mask{img_num}.png")

                if not os.path.exists(mask_path):
                    print(f"⚠ Missing merged mask: {mask_path}")
                    continue

                shutil.copy(img_path, os.path.join(target_images, new_base + ".png"))
                shutil.copy(mask_path, os.path.join(target_masks, new_base + ".png"))

                print(f"  Saved: {new_base}.png")

            # ----------------------------------------------------
            # MULTILABEL ANATOMY (7 masks per image)
            # ----------------------------------------------------
            else:
                merged_path = os.path.join(Xpath, "merged")

                shutil.copy(img_path, os.path.join(target_images, new_base + ".png"))

                for organ in MULTI_ORGANS:
                    mask_name = f"mask{img_num}_{organ}.png"
                    mask_path = os.path.join(merged_path, mask_name)

                    if not os.path.exists(mask_path):
                        print(f"⚠ Missing multi-mask: {mask_path}")
                        continue

                    shutil.copy(mask_path,
                                os.path.join(target_masks, f"{new_base}_{organ}.png"))
                
                print(f"  Saved multilabel set: {new_base}.png")

            count += 1

    print(f"Total images processed for {anatomy}: {count}")

print("\n🎉 All anatomies (including multilabel) processed successfully!")