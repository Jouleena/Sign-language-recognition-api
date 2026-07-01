"""
One-time migration script.
Converts old dataset files from shape (FRAMES_COUNT, 63) -> (FRAMES_COUNT, 126)
by zero-padding the Hand 2 slot (columns 63:126).

This matches the new two-hand format used by the updated collect_data.py:
    [0:63]   = Hand 1 (the hand that was already recorded)
    [63:126] = Hand 2 (zeros, since these old videos only had one hand)

Run this ONCE before retraining. It modifies files in place after
creating a backup copy of the whole dataset folder.
"""

import numpy as np
import os
import shutil

DATASET_PATH = "dataset"
BACKUP_PATH  = "dataset_backup_before_migration"
FRAMES_COUNT = 60
OLD_DIM      = 63
NEW_DIM      = 126


def migrate():
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: '{DATASET_PATH}' folder not found.")
        return

    # Backup first — never touch data without a safety copy
    if not os.path.exists(BACKUP_PATH):
        print(f"Backing up '{DATASET_PATH}' -> '{BACKUP_PATH}' ...")
        shutil.copytree(DATASET_PATH, BACKUP_PATH)
        print("Backup done.\n")
    else:
        print(f"Backup already exists at '{BACKUP_PATH}', skipping backup step.\n")

    words = sorted(os.listdir(DATASET_PATH))
    total_converted = 0
    total_skipped_already_new = 0
    total_skipped_bad_shape = 0

    for word in words:
        word_path = os.path.join(DATASET_PATH, word)
        if not os.path.isdir(word_path):
            continue

        files = [f for f in os.listdir(word_path) if f.endswith(".npy")]
        print(f"[{word}] {len(files)} files")

        for file in files:
            file_path = os.path.join(word_path, file)
            arr = np.load(file_path)

            if arr.shape == (FRAMES_COUNT, NEW_DIM):
                total_skipped_already_new += 1
                continue

            if arr.shape != (FRAMES_COUNT, OLD_DIM):
                print(f"  SKIP (unexpected shape {arr.shape}): {file}")
                total_skipped_bad_shape += 1
                continue

            padding = np.zeros((FRAMES_COUNT, NEW_DIM - OLD_DIM))
            new_arr = np.concatenate([arr, padding], axis=1)  # (60, 126)

            np.save(file_path, new_arr)
            total_converted += 1

    print("\n" + "=" * 40)
    print("MIGRATION SUMMARY")
    print("=" * 40)
    print(f"  Converted (63 -> 126):   {total_converted}")
    print(f"  Already 126, untouched:  {total_skipped_already_new}")
    print(f"  Skipped (bad shape):     {total_skipped_bad_shape}")
    print(f"\nOriginal data backed up in: '{BACKUP_PATH}'")


if __name__ == "__main__":
    migrate()
