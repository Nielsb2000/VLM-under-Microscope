from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caseviz.io import load_runs
from caseviz.matching import matched_pairs

input_root = (ROOT / "outputs" / "case_study_1").resolve()
df = load_runs(input_root)

print("=== FULL TABLE ===")
print("shape:", df.shape)
print("columns:")
for c in df.columns:
    print(" -", c)

print("\n=== FULL TABLE ROWS MATCHING 749.tiff ===")
mask = pd.Series(False, index=df.index)
for col in df.columns:
    if df[col].dtype == object:
        mask |= df[col].astype(str).map(lambda x: Path(x).name == "749.tiff")

hit = df.loc[mask].copy()
print("match count:", len(hit))
if len(hit):
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(hit.to_string(index=False))

long_pairs, wide_pairs = matched_pairs(df)

print("\n=== MATCHED PAIRS ===")
print("long_pairs shape:", long_pairs.shape)
print("wide_pairs shape:", wide_pairs.shape)

print("\nlong_pairs columns:")
for c in long_pairs.columns:
    print(" -", c)

print("\nwide_pairs columns:")
for c in wide_pairs.columns:
    print(" -", c)

print("\n=== SHARED ID COLUMNS IN df AND long_pairs ===")
candidates = ["pair_id", "matched_id", "case_id", "image_id", "seed", "run_seed", "degradation_seed", "input_id"]
shared = [c for c in candidates if c in df.columns and c in long_pairs.columns]
print(shared if shared else "NONE")

print("\n=== long_pairs METHOD COUNTS ===")
if "method" in long_pairs.columns:
    print(long_pairs["method"].value_counts(dropna=False).to_string())

print("\n=== long_pairs ROWS MATCHING 749.tiff ===")
mask2 = pd.Series(False, index=long_pairs.index)
for col in long_pairs.columns:
    if long_pairs[col].dtype == object:
        mask2 |= long_pairs[col].astype(str).map(lambda x: Path(x).name == "749.tiff")

hit2 = long_pairs.loc[mask2].copy()
print("match count:", len(hit2))
if len(hit2):
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(hit2.to_string(index=False))
