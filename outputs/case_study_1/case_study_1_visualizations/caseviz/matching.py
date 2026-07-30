from __future__ import annotations

import pandas as pd

MATCH_COLUMNS = [
    "sample_seed",
    "randomization_seed",
    "sample_index",
    "random_brightness",
    "random_contrast",
]

FALLBACK_MATCH_COLUMNS = ["image", "randomization_seed", "random_brightness", "random_contrast"]


def make_match_key(df: pd.DataFrame) -> pd.Series:
    """Return a robust key for pairing direct/exploratory runs with same starting conditions."""
    cols = [c for c in MATCH_COLUMNS if c in df.columns and df[c].notna().any()]
    if len(cols) < 3:
        cols = [c for c in FALLBACK_MATCH_COLUMNS if c in df.columns and df[c].notna().any()]
    return df[cols].astype(str).agg("|".join, axis=1) if cols else pd.Series(range(len(df)), index=df.index).astype(str)


def matched_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paired direct-vs-exploratory table.

    Returns
    -------
    long_df:
        Input rows restricted to match groups containing both methods.
    wide_df:
        One row per matched case with direct/exploratory score and deltas.
    """
    if df.empty:
        return df.copy(), pd.DataFrame()
    usable = df[(df.get("status") == "completed") & df["final_score"].notna()].copy()
    usable["match_key"] = make_match_key(usable)
    counts = usable.groupby("match_key")["method"].nunique()
    keys = counts[counts >= 2].index
    long_df = usable[usable["match_key"].isin(keys)].copy()

    # If there are repeated runs per method per match, average for the paired overview.
    agg_cols = {
        "final_score": "mean",
        "absolute_improvement": "mean",
        "relative_improvement": "mean",
        "randomized_score": "mean",
        "filter_adjustments": "mean",
        "vlm_snapshots": "mean",
    }
    meta_cols = ["image", "sample_id", "sample_seed", "randomization_seed", "sample_index", "random_brightness", "random_contrast"]
    grouped = long_df.groupby(["match_key", "method"], dropna=False).agg({**agg_cols, **{c: "first" for c in meta_cols if c in long_df.columns}}).reset_index()
    wide = grouped.pivot(index="match_key", columns="method", values=list(agg_cols.keys()))
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    meta = grouped.groupby("match_key").agg({c: "first" for c in meta_cols if c in grouped.columns}).reset_index()
    wide_df = meta.merge(wide, on="match_key", how="left")
    if {"final_score_direct", "final_score_exploratory"}.issubset(wide_df.columns):
        wide_df["score_delta_exploratory_minus_direct"] = wide_df["final_score_exploratory"] - wide_df["final_score_direct"]
        wide_df["exploratory_better"] = wide_df["score_delta_exploratory_minus_direct"] < 0
    if {"absolute_improvement_direct", "absolute_improvement_exploratory"}.issubset(wide_df.columns):
        wide_df["improvement_delta_exploratory_minus_direct"] = wide_df["absolute_improvement_exploratory"] - wide_df["absolute_improvement_direct"]
    return long_df.reset_index(drop=True), wide_df.reset_index(drop=True)
