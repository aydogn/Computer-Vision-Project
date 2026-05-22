"""
Kvasir-SEG resmi zip'inden train/validation/test split'i oluşturur.
Split: 800 train / 100 validation / 100 test (sabit seed ile tekrarlanabilir)
"""
import os
import shutil
import random

RAW_DIR = "kvasir-seg-raw/Kvasir-SEG"
OUT_DIR = "kvasir-seg"
SPLITS = {"train": 800, "validation": 100, "test": 100}
SEED = 42

def main():
    images_src = os.path.join(RAW_DIR, "images")
    masks_src  = os.path.join(RAW_DIR, "masks")

    all_files = sorted([
        f for f in os.listdir(images_src)
        if os.path.exists(os.path.join(masks_src, f))
    ])
    print(f"Toplam eslesen goruntu: {len(all_files)}")

    random.seed(SEED)
    random.shuffle(all_files)

    splits = {}
    idx = 0
    for split, count in SPLITS.items():
        splits[split] = all_files[idx:idx + count]
        idx += count

    for split, files in splits.items():
        img_dir  = os.path.join(OUT_DIR, split, "images")
        mask_dir = os.path.join(OUT_DIR, split, "masks")
        os.makedirs(img_dir,  exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(images_src, f), os.path.join(img_dir,  f))
            shutil.copy(os.path.join(masks_src,  f), os.path.join(mask_dir, f))
        print(f"  {split}: {len(files)} goruntu kopyalandi -> {img_dir}")

    print("\nHazir! Klasor yapisi:")
    for split in SPLITS:
        n = len(os.listdir(os.path.join(OUT_DIR, split, "images")))
        print(f"  kvasir-seg/{split}/images -> {n} dosya")

if __name__ == "__main__":
    main()
