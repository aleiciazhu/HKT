import pickle
import numpy as np
import os

# Load the dataset
dataset_path = "./dataset/humor/ur_funny.pkl"

if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at {dataset_path}")
    exit(1)

with open(dataset_path, "rb") as f:
    all_data = pickle.load(f)

# Check each split
for split_name in ["train", "dev", "test"]:
    data = all_data[split_name]
    print(f"\n{split_name.upper()} split:")
    print(f"  Number of samples: {len(data)}")

    if len(data) >= 2:
        # Get first two samples
        sample0 = data[0]
        sample1 = data[1]

        # Unpack the sample structure
        (p_words0, p_visual0, p_acoustic0, p_hcf0), (c_words0, c_visual0, c_acoustic0, c_hcf0), hid0, label0 = sample0
        (p_words1, p_visual1, p_acoustic1, p_hcf1), (c_words1, c_visual1, c_acoustic1, c_hcf1), hid1, label1 = sample1

        # Check if they're different
        print(f"\n  Sample 0:")
        print(f"    p_words: {' '.join(p_words0[:5]) if len(p_words0) > 0 else 'empty'}...")
        print(f"    p_visual shape: {p_visual0.shape}, sum: {p_visual0.sum():.4f}")
        print(f"    p_acoustic shape: {p_acoustic0.shape}, sum: {p_acoustic0.sum():.4f}")
        print(f"    label: {label0}")

        print(f"\n  Sample 1:")
        print(f"    p_words: {' '.join(p_words1[:5]) if len(p_words1) > 0 else 'empty'}...")
        print(f"    p_visual shape: {p_visual1.shape}, sum: {p_visual1.sum():.4f}")
        print(f"    p_acoustic shape: {p_acoustic1.shape}, sum: {p_acoustic1.sum():.4f}")
        print(f"    label: {label1}")

        # Check if features are identical
        visual_same = np.array_equal(p_visual0, p_visual1)
        acoustic_same = np.array_equal(p_acoustic0, p_acoustic1)

        print(f"\n  Are samples identical?")
        print(f"    Visual: {visual_same}")
        print(f"    Acoustic: {acoustic_same}")
        print(f"    Words same: {p_words0 == p_words1}")
