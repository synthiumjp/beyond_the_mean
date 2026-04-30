"""
Beyond the Mean: Supplementary Analyses
========================================
1. Pre-registered: Greedy comparison (Section 5.3.3)
2. Pre-registered: Stratified S_diff re-reporting of H1/H4 (Section 5.1.1)
3. Post-hoc: Churn rate, transition analysis, RCI magnitude, domain ratios

Usage:
    python run_btm_supplementary.py
    python run_btm_supplementary.py --greedy-dir D:\bcb_pilot\results
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

DEFAULT_DATA = r"D:\beyond_the_mean\data"
DEFAULT_OUTPUT = r"D:\beyond_the_mean\results"
DEFAULT_GREEDY = r"D:\bcb_pilot\results"

K = 10
RCI_THRESHOLD = 1.96


# ============================================================
# Data Loading (shared with main analysis)
# ============================================================

def load_trials(filepath: str) -> pd.DataFrame:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_item_accuracy(trials: pd.DataFrame) -> pd.DataFrame:
    valid = trials[~trials["is_missing"]].copy()
    grouped = valid.groupby(["item_index", "item_id", "domain"]).agg(
        n_correct=("is_correct", "sum"),
        n_valid=("is_correct", "count"),
    ).reset_index()
    grouped["p"] = grouped["n_correct"] / grouped["n_valid"]
    return grouped


def compute_rci_merged(items_v1, items_v2, r_xx_v1, r_xx_v2):
    """Compute merged RCI DataFrame."""
    merged = items_v1.merge(items_v2, on=["item_index", "item_id", "domain"], suffixes=("_v1", "_v2"))
    sd_v1 = merged["p_v1"].std()
    sd_v2 = merged["p_v2"].std()
    sem_v1 = sd_v1 * np.sqrt(1 - r_xx_v1)
    sem_v2 = sd_v2 * np.sqrt(1 - r_xx_v2)
    s_diff = np.sqrt(sem_v1**2 + sem_v2**2)
    merged["delta"] = merged["p_v2"] - merged["p_v1"]
    merged["rci"] = merged["delta"] / s_diff if s_diff > 0 else 0
    merged["rci_category"] = "no_reliable_change"
    merged.loc[merged["rci"] > RCI_THRESHOLD, "rci_category"] = "reliable_improvement"
    merged.loc[merged["rci"] < -RCI_THRESHOLD, "rci_category"] = "reliable_deterioration"
    merged.attrs["s_diff"] = s_diff
    merged.attrs["sem_v1"] = sem_v1
    merged.attrs["sem_v2"] = sem_v2
    return merged


def split_half_reliability(trials, n_splits=1000):
    valid = trials[~trials["is_missing"]].copy()
    pivot = valid.pivot_table(index="item_index", columns="k", values="is_correct", aggfunc="first")
    k_cols = sorted(pivot.columns.tolist())
    r_xx_values = []
    rng = np.random.RandomState(42)
    for _ in range(n_splits):
        perm = rng.permutation(k_cols)
        p1 = pivot[perm[:5]].mean(axis=1).values
        p2 = pivot[perm[5:]].mean(axis=1).values
        r_half = np.corrcoef(p1, p2)[0, 1]
        r_xx = (2 * r_half) / (1 + r_half)
        r_xx_values.append(r_xx)
    return np.median(r_xx_values)


# ============================================================
# 1. Greedy Comparison (Pre-registered, Section 5.3.3)
# ============================================================

def find_greedy_data(greedy_dir: str) -> tuple:
    """Find Study 3 greedy honest-baseline data for Llama pair.
    Tries several likely file patterns."""
    candidates_v1 = [
        "study3_honest_llama3-8b.jsonl",
        "honest_llama3-8b.jsonl",
        "H_llama3-8b.jsonl",
        "trials_H_llama3-8b.jsonl",
    ]
    candidates_v2 = [
        "study3_honest_llama3.1-8b.jsonl",
        "honest_llama3.1-8b.jsonl",
        "H_llama3.1-8b.jsonl",
        "trials_H_llama3.1-8b.jsonl",
    ]

    # Also scan directory for any files containing relevant keywords
    found_files = []
    if os.path.exists(greedy_dir):
        for f in os.listdir(greedy_dir):
            found_files.append(f)

    return found_files


def greedy_comparison(data_dir: str, greedy_dir: str):
    """Compare greedy (T=0) item changes with RCI classification."""

    print(f"\n{'#'*70}")
    print("# Pre-registered: Greedy Comparison (Section 5.3.3)")
    print(f"{'#'*70}")

    # Direct loading of Study 3 honest-baseline files
    v1_path = os.path.join(greedy_dir, "llama3-8b_H.jsonl")
    v2_path = os.path.join(greedy_dir, "llama3.1-8b_H.jsonl")

    if not os.path.exists(v1_path):
        print(f"  Not found: {v1_path}")
        print("  Skipping greedy comparison.")
        return
    if not os.path.exists(v2_path):
        print(f"  Not found: {v2_path}")
        print("  Skipping greedy comparison.")
        return

    print(f"  Loading {v1_path}")
    greedy_v1 = load_trials(v1_path)
    print(f"  Loading {v2_path}")
    greedy_v2 = load_trials(v2_path)
    print(f"  Llama-3-8B greedy: {len(greedy_v1)} records")
    print(f"  Llama-3.1-8B greedy: {len(greedy_v2)} records")

    # Compute greedy item-level correctness (binary, T=0)
    # Study 3 format: one row per item, uses item_id not item_index
    greedy_correct_v1 = greedy_v1[["item_id", "is_correct"]].copy()
    greedy_correct_v1.columns = ["item_id", "greedy_correct_v1"]
    greedy_correct_v2 = greedy_v2[["item_id", "is_correct"]].copy()
    greedy_correct_v2.columns = ["item_id", "greedy_correct_v2"]

    greedy = greedy_correct_v1.merge(greedy_correct_v2, on="item_id")

    # Drop items with None/NaN correctness (missing responses at T=0)
    before_drop = len(greedy)
    greedy = greedy.dropna(subset=["greedy_correct_v1", "greedy_correct_v2"])
    greedy["greedy_correct_v1"] = greedy["greedy_correct_v1"].astype(bool)
    greedy["greedy_correct_v2"] = greedy["greedy_correct_v2"].astype(bool)
    dropped = before_drop - len(greedy)
    if dropped > 0:
        print(f"  Dropped {dropped} items with missing greedy responses")

    greedy["greedy_change"] = "no_change"
    greedy.loc[(~greedy["greedy_correct_v1"]) & (greedy["greedy_correct_v2"]), "greedy_change"] = "gained"
    greedy.loc[(greedy["greedy_correct_v1"]) & (~greedy["greedy_correct_v2"]), "greedy_change"] = "lost"

    # Load T=0.7 RCI classifications
    trials_v1 = load_trials(os.path.join(data_dir, "trials_llama3-8b.jsonl"))
    trials_v2 = load_trials(os.path.join(data_dir, "trials_llama3_1-8b.jsonl"))
    r_v1 = split_half_reliability(trials_v1)
    r_v2 = split_half_reliability(trials_v2)
    items_v1 = compute_item_accuracy(trials_v1)
    items_v2 = compute_item_accuracy(trials_v2)
    merged = compute_rci_merged(items_v1, items_v2, r_v1, r_v2)

    # Merge greedy with RCI via item_id
    combined = merged.merge(greedy, on="item_id", how="inner")

    print(f"\n  Matched items: {len(combined)}")

    # Cross-tabulation
    ct = pd.crosstab(combined["greedy_change"], combined["rci_category"])
    print(f"\n  Greedy change vs RCI classification:")
    print(ct.to_string())

    # Agreement rate
    agree = 0
    total = len(combined)
    for _, row in combined.iterrows():
        g = row["greedy_change"]
        r = row["rci_category"]
        if g == "gained" and r == "reliable_improvement":
            agree += 1
        elif g == "lost" and r == "reliable_deterioration":
            agree += 1
        elif g == "no_change" and r == "no_reliable_change":
            agree += 1

    print(f"\n  Exact agreement: {agree}/{total} ({agree/total*100:.1f}%)")

    # Items that greedy says changed but RCI says no
    greedy_changed = combined[combined["greedy_change"] != "no_change"]
    rci_no_change_on_greedy_changed = greedy_changed[greedy_changed["rci_category"] == "no_reliable_change"]
    print(f"  Items greedy says changed, RCI says no change: {len(rci_no_change_on_greedy_changed)}/{len(greedy_changed)}")

    # Items that RCI says changed but greedy says no
    rci_changed = combined[combined["rci_category"] != "no_reliable_change"]
    greedy_no_change_on_rci_changed = rci_changed[rci_changed["greedy_change"] == "no_change"]
    print(f"  Items RCI says changed, greedy says no change: {len(greedy_no_change_on_rci_changed)}/{len(rci_changed)}")


# ============================================================
# 2. Stratified S_diff H1/H4 Re-reporting (Pre-registered 5.1.1)
# ============================================================

def stratified_h1_h4(data_dir: str):
    """Re-report H1/H4 using difficulty-stratified S_diff."""

    print(f"\n{'#'*70}")
    print("# Pre-registered: Stratified S_diff Re-reporting (Section 5.1.1)")
    print(f"{'#'*70}")

    for pair_name, pair_config in [
        ("Llama", {
            "v1_file": "trials_llama3-8b.jsonl",
            "v2_file": "trials_llama3_1-8b.jsonl",
            "label": "Llama 3 -> 3.1",
        }),
        ("Qwen", {
            "v1_file": "trials_qwen2_5-7b.jsonl",
            "v2_file": "trials_qwen3-8b.jsonl",
            "label": "Qwen 2.5 -> 3",
        }),
    ]:
        print(f"\n  --- {pair_config['label']} ---")

        trials_v1 = load_trials(os.path.join(data_dir, pair_config["v1_file"]))
        trials_v2 = load_trials(os.path.join(data_dir, pair_config["v2_file"]))
        r_v1 = split_half_reliability(trials_v1)
        r_v2 = split_half_reliability(trials_v2)
        items_v1 = compute_item_accuracy(trials_v1)
        items_v2 = compute_item_accuracy(trials_v2)
        merged = compute_rci_merged(items_v1, items_v2, r_v1, r_v2)

        # Apply exclusions
        mask = ~(((merged["p_v1"] == 0.0) & (merged["p_v2"] == 0.0)) |
                 ((merged["p_v1"] == 1.0) & (merged["p_v2"] == 1.0)))
        merged = merged[mask].copy()

        merged["difficulty"] = (merged["p_v1"] + merged["p_v2"]) / 2

        # Terciles
        try:
            merged["tercile"] = pd.qcut(merged["difficulty"], 3, labels=["hard", "medium", "easy"])
        except ValueError:
            merged["tercile"] = pd.qcut(merged["difficulty"], 3, labels=False, duplicates="drop")
            unique_bins = sorted(merged["tercile"].unique())
            labels = ["hard", "medium", "easy"][:len(unique_bins)]
            merged["tercile"] = merged["tercile"].map(dict(zip(unique_bins, labels)))

        # Compute stratified RCI per tercile
        rci_cats_stratified = []
        for _, row in merged.iterrows():
            t = row["tercile"]
            sub = merged[merged["tercile"] == t]
            sd_v1 = sub["p_v1"].std()
            sd_v2 = sub["p_v2"].std()
            sem_v1 = sd_v1 * np.sqrt(1 - r_v1)
            sem_v2 = sd_v2 * np.sqrt(1 - r_v2)
            s_diff_strat = np.sqrt(sem_v1**2 + sem_v2**2)
            if s_diff_strat > 0:
                rci_strat = row["delta"] / s_diff_strat
            else:
                rci_strat = 0
            if rci_strat > RCI_THRESHOLD:
                rci_cats_stratified.append("reliable_improvement")
            elif rci_strat < -RCI_THRESHOLD:
                rci_cats_stratified.append("reliable_deterioration")
            else:
                rci_cats_stratified.append("no_reliable_change")

        merged["rci_category_stratified"] = rci_cats_stratified

        n = len(merged)
        n_imp_s = (merged["rci_category_stratified"] == "reliable_improvement").sum()
        n_nc_s = (merged["rci_category_stratified"] == "no_reliable_change").sum()
        n_det_s = (merged["rci_category_stratified"] == "reliable_deterioration").sum()

        n_imp_g = (merged["rci_category"] == "reliable_improvement").sum()
        n_nc_g = (merged["rci_category"] == "no_reliable_change").sum()
        n_det_g = (merged["rci_category"] == "reliable_deterioration").sum()

        n_pos = (merged["delta"] > 0).sum()
        br_g = n_imp_g / n_pos if n_pos > 0 else 0
        br_s = n_imp_s / n_pos if n_pos > 0 else 0

        print(f"  Global S_diff:     improved={n_imp_g}, no_change={n_nc_g}, deteriorated={n_det_g}")
        print(f"  Stratified S_diff: improved={n_imp_s}, no_change={n_nc_s}, deteriorated={n_det_s}")
        print(f"  H1 global:     {'SUPPORTED' if n_nc_g/n > 0.5 else 'NOT SUPPORTED'} ({n_nc_g/n:.1%})")
        print(f"  H1 stratified: {'SUPPORTED' if n_nc_s/n > 0.5 else 'NOT SUPPORTED'} ({n_nc_s/n:.1%})")
        print(f"  H4 global:     breadth_ratio={br_g:.4f} {'SUPPORTED' if br_g < 0.5 else 'NOT SUPPORTED'}")
        print(f"  H4 stratified: breadth_ratio={br_s:.4f} {'SUPPORTED' if br_s < 0.5 else 'NOT SUPPORTED'}")

        # Check if conclusions differ
        h1_differs = (n_nc_g/n > 0.5) != (n_nc_s/n > 0.5)
        h4_differs = (br_g < 0.5) != (br_s < 0.5)
        if h1_differs or h4_differs:
            print(f"  *** CONCLUSIONS DIFFER between global and stratified ***")
        else:
            print(f"  Conclusions consistent across global and stratified.")


# ============================================================
# 3. Post-hoc Analyses
# ============================================================

def posthoc_analyses(data_dir: str):
    """Post-hoc exploratory analyses (clearly labelled)."""

    print(f"\n{'#'*70}")
    print("# POST-HOC ANALYSES (not pre-registered)")
    print(f"{'#'*70}")

    for pair_name, pair_config in [
        ("Llama", {
            "v1_file": "trials_llama3-8b.jsonl",
            "v2_file": "trials_llama3_1-8b.jsonl",
            "label": "Llama 3 -> 3.1",
        }),
        ("Qwen", {
            "v1_file": "trials_qwen2_5-7b.jsonl",
            "v2_file": "trials_qwen3-8b.jsonl",
            "label": "Qwen 2.5 -> 3",
        }),
    ]:
        print(f"\n  === {pair_config['label']} ===")

        trials_v1 = load_trials(os.path.join(data_dir, pair_config["v1_file"]))
        trials_v2 = load_trials(os.path.join(data_dir, pair_config["v2_file"]))
        r_v1 = split_half_reliability(trials_v1)
        r_v2 = split_half_reliability(trials_v2)
        items_v1 = compute_item_accuracy(trials_v1)
        items_v2 = compute_item_accuracy(trials_v2)
        merged_full = compute_rci_merged(items_v1, items_v2, r_v1, r_v2)

        # Exclusion mask (for post-exclusion analyses)
        mask_excl = ~(((merged_full["p_v1"] == 0.0) & (merged_full["p_v2"] == 0.0)) |
                      ((merged_full["p_v1"] == 1.0) & (merged_full["p_v2"] == 1.0)))
        merged_post = merged_full[mask_excl].copy()

        # --- 3a. Churn Rate ---
        n_post = len(merged_post)
        n_changed = (merged_post["rci_category"] != "no_reliable_change").sum()
        churn = n_changed / n_post
        print(f"\n  Churn rate (post-exclusion): {n_changed}/{n_post} ({churn:.1%})")

        # --- 3b. Domain improvement-deterioration ratios ---
        print(f"\n  Domain improvement/deterioration ratios:")
        for d in sorted(merged_post["domain"].unique()):
            d_data = merged_post[merged_post["domain"] == d]
            n_imp = (d_data["rci_category"] == "reliable_improvement").sum()
            n_det = (d_data["rci_category"] == "reliable_deterioration").sum()
            ratio = n_imp / n_det if n_det > 0 else float("inf")
            net = n_imp - n_det
            print(f"    {d}: {n_imp} improved / {n_det} deteriorated = {ratio:.2f} (net {'+' if net >= 0 else ''}{net})")

        # --- 3c. Transition Analysis (on FULL 2000 items) ---
        print(f"\n  Transition analysis (full 2000 items):")

        # Items always correct in v1
        always_correct_v1 = merged_full[merged_full["p_v1"] == 1.0]
        n_ac = len(always_correct_v1)
        n_ac_dropped = (always_correct_v1["p_v2"] < 1.0).sum()
        print(f"    Always correct in v1 (p=1.0): {n_ac}")
        print(f"      Dropped below 1.0 in v2: {n_ac_dropped} ({n_ac_dropped/n_ac*100:.1f}%)" if n_ac > 0 else "")

        # Items always wrong in v1
        always_wrong_v1 = merged_full[merged_full["p_v1"] == 0.0]
        n_aw = len(always_wrong_v1)
        n_aw_gained = (always_wrong_v1["p_v2"] > 0.0).sum()
        print(f"    Always wrong in v1 (p=0.0): {n_aw}")
        print(f"      Rose above 0.0 in v2: {n_aw_gained} ({n_aw_gained/n_aw*100:.1f}%)" if n_aw > 0 else "")

        # Items always correct in v2
        always_correct_v2 = merged_full[merged_full["p_v2"] == 1.0]
        n_ac2 = len(always_correct_v2)
        # Items always wrong in v2
        always_wrong_v2 = merged_full[merged_full["p_v2"] == 0.0]
        n_aw2 = len(always_wrong_v2)
        print(f"    Always correct in v2: {n_ac2}")
        print(f"    Always wrong in v2: {n_aw2}")

        # --- 3d. RCI Magnitude Distribution ---
        print(f"\n  RCI magnitude distribution (post-exclusion):")
        rci_vals = merged_post["rci"].values
        print(f"    Mean:   {np.mean(rci_vals):.3f}")
        print(f"    Median: {np.median(rci_vals):.3f}")
        print(f"    SD:     {np.std(rci_vals):.3f}")
        print(f"    Min:    {np.min(rci_vals):.3f}")
        print(f"    Max:    {np.max(rci_vals):.3f}")

        # Percentiles
        for pct in [5, 25, 75, 95]:
            print(f"    {pct}th pctl: {np.percentile(rci_vals, pct):.3f}")

        # How many exceed various thresholds
        for thresh in [3, 5, 7, 10]:
            n_above = np.sum(np.abs(rci_vals) > thresh)
            print(f"    |RCI| > {thresh}: {n_above} items")

        # --- 3e. p_v1 vs p_v2 summary for scatter plot ---
        print(f"\n  p_v1 vs p_v2 summary (post-exclusion):")
        corr = np.corrcoef(merged_post["p_v1"].values, merged_post["p_v2"].values)[0, 1]
        print(f"    Correlation: r = {corr:.4f}")
        print(f"    Items above diagonal (improved): {(merged_post['delta'] > 0).sum()}")
        print(f"    Items on diagonal (no change): {(merged_post['delta'] == 0).sum()}")
        print(f"    Items below diagonal (worsened): {(merged_post['delta'] < 0).sum()}")

    # --- 3f. Cross-pair churn comparison ---
    print(f"\n  --- Cross-Pair Summary ---")
    print(f"  Llama churn vs Qwen churn reported above.")
    print(f"  The 'capability trading' interpretation: version updates")
    print(f"  simultaneously improve and degrade performance on comparable")
    print(f"  numbers of items. The aggregate shows only the net residual.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Beyond the Mean: Supplementary Analyses")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--greedy-dir", default=DEFAULT_GREEDY,
                        help="Directory containing Study 3 greedy results")
    args = parser.parse_args()

    # Pre-registered: Stratified S_diff re-reporting
    stratified_h1_h4(args.data)

    # Pre-registered: Greedy comparison
    greedy_comparison(args.data, args.greedy_dir)

    # Post-hoc analyses
    posthoc_analyses(args.data)

    print(f"\n{'='*70}")
    print("SUPPLEMENTARY ANALYSES COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
