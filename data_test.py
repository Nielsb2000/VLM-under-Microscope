from datasets import load_dataset
from collections import Counter
import pickle
from pathlib import Path

# Cache file path
cache_file = Path("ubench_modalities.pkl")

# Load from cache or download
if cache_file.exists():
    print("Loading from cache...")
    with open(cache_file, 'rb') as f:
        modalities = pickle.load(f)
else:
    print("Downloading dataset (first time only)...")
    ds = load_dataset("jnirschl/uBench")
    test_ds = ds['test']
    modalities = test_ds['modality']
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(modalities, f)
    print("Saved to cache for next time!")

# Get modality column without decoding images
test_ds_len = len(modalities)

# Count electron microscopy images
em_count = modalities.count('electron microscopy')

print(f"✓ Number of electron microscopy images: {em_count}")
print(f"  ({em_count / test_ds_len * 100:.1f}% of {test_ds_len} total images)")

# Show all modality counts
print("\nAll modalities:")
modality_counts = Counter(modalities)
for modality, count in modality_counts.most_common():
    print(f"  {modality}: {count}")

