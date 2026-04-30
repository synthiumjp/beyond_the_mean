"""
Beyond the Mean: Effect Size and Difficulty-Bin Analysis
=========================================================
Post-hoc analyses requested by reviewers.
1. Raw delta (Δp) distribution statistics
2. Churn rate by baseline difficulty bin
"""

import json
import numpy as np
import pandas as pd
import os

DEFAULT_DATA = r"D:\beyond_the_mean\data"

def load_trials(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)

def compute_item_accuracy(trials):
    valid = trials[~trials["is_missing"]].copy()
    grouped = valid.groupby(["item_index", "item_id", "domain"]).agg(
        n_correct=("is_correct", "sum"),
        n_valid=("is_correct", "count"),
    ).reset_index()
    grouped["p"] = grouped["n_correct"] / grouped["n_valid"]
    return grouped

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

def analyze_pair(data_dir, v1_file, v2_file, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    trials_v1 = load_trials(os.path.join(data_dir, v1_file))
    trials_v2 = load_trials(os.path.join(data_dir, v2_file))

    r_v1 = split_half_reliability(trials_v1)
    r_v2 = split_half_reliability(trials_v2)

    items_v1 = compute_item_accuracy(trials_v1)
    items_v2 = compute_item_accuracy(trials_v2)

    merged = items_v1.merge(items_v2, on=["item_index", "item_id", "domain"], suffixes=("_v1", "_v2"))
    merged["delta"] = merged["p_v2"] - merged["p_v1"]
    merged["abs_delta"] = merged["delta"].abs()

    # Compute RCI
    sd_v1 = merged["p_v1"].std()
    sd_v2 = merged["p_v2"].std()
    sem_v1 = sd_v1 * np.sqrt(1 - r_v1)
    sem_v2 = sd_v2 * np.sqrt(1 - r_v2)
    s_diff = np.sqrt(sem_v1**2 + sem_v2**2)
    merged["rci"] = merged["delta"] / s_diff

    merged["rci_cat"] = "no_reliable_change"
    merged.loc[merged["rci"] > 1.96, "rci_cat"] = "reliable_improvement"
    merged.loc[merged["rci"] < -1.96, "rci_cat"] = "reliable_deterioration"

    # Exclusion
    excl_mask = ~(((merged["p_v1"] == 0.0) & (merged["p_v2"] == 0.0)) |
                  ((merged["p_v1"] == 1.0) & (merged["p_v2"] == 1.0)))
    post = merged[excl_mask].copy()

    # ==========================================
    # 1. RAW DELTA DISTRIBUTION (post-exclusion)
    # ==========================================
    print(f"\n--- Raw Delta Distribution (post-exclusion, n={len(post)}) ---")
    deltas = post["delta"].values
    abs_deltas = post["abs_delta"].values
    print(f"  Mean delta:     {np.mean(deltas):.4f}")
    print(f"  Mean |delta|:   {np.mean(abs_deltas):.4f}")
    print(f"  Median |delta|: {np.median(abs_deltas):.4f}")
    print(f"  SD delta:       {np.std(deltas):.4f}")

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        n_above = np.sum(abs_deltas >= thresh)
        pct = n_above / len(abs_deltas) * 100
        print(f"  |delta| >= {thresh}: {n_above} ({pct:.1f}%)")

    # Among reliably changed items only
    changed = post[post["rci_cat"] != "no_reliable_change"]
    if len(changed) > 0:
        print(f"\n  Among reliably changed items (n={len(changed)}):")
        ch_abs = changed["abs_delta"].values
        print(f"    Mean |delta|:   {np.mean(ch_abs):.4f}")
        print(f"    Median |delta|: {np.median(ch_abs):.4f}")
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            n_above = np.sum(ch_abs >= thresh)
            pct = n_above / len(ch_abs) * 100
            print(f"    |delta| >= {thresh}: {n_above} ({pct:.1f}%)")

    # ==========================================
    # 2. CHURN BY BASELINE DIFFICULTY BIN
    # ==========================================
    print(f"\n--- Churn by Baseline Difficulty (p_v1 bins, post-exclusion) ---")
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    for (lo, hi), lab in zip(bins, labels):
        sub = post[(post["p_v1"] >= lo) & (post["p_v1"] < hi)]
        n = len(sub)
        if n == 0:
            print(f"  {lab}: n=0")
            continue
        n_changed = (sub["rci_cat"] != "no_reliable_change").sum()
        n_imp = (sub["rci_cat"] == "reliable_improvement").sum()
        n_det = (sub["rci_cat"] == "reliable_deterioration").sum()
        churn = n_changed / n * 100
        mean_delta = sub["delta"].mean()
        mean_abs = sub["abs_delta"].mean()
        print(f"  {lab}: n={n:4d} | churn={churn:5.1f}% | "
              f"imp={n_imp:3d} det={n_det:3d} | "
              f"mean_delta={mean_delta:+.3f} | mean_|delta|={mean_abs:.3f}")

    # ==========================================
    # 3. FULL BENCHMARK VIEW (all 2000 items)
    # ==========================================
    print(f"\n--- Full Benchmark View (all 2000 items) ---")
    n_total = len(merged)
    n_floor_ceil = n_total - len(post)
    n_changed_full = (merged["rci_cat"] != "no_reliable_change").sum()
    n_imp_full = (merged["rci_cat"] == "reliable_improvement").sum()
    n_det_full = (merged["rci_cat"] == "reliable_deterioration").sum()
    n_nc_full = (merged["rci_cat"] == "no_reliable_change").sum()

    print(f"  Total items: {n_total}")
    print(f"  Floor/ceiling (stable): {n_floor_ceil} ({n_floor_ceil/n_total*100:.1f}%)")
    print(f"  Reliable improvement: {n_imp_full} ({n_imp_full/n_total*100:.1f}%)")
    print(f"  No reliable change: {n_nc_full} ({n_nc_full/n_total*100:.1f}%)")
    print(f"  Reliable deterioration: {n_det_full} ({n_det_full/n_total*100:.1f}%)")
    print(f"  Full-benchmark churn: {n_changed_full/n_total*100:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()

    analyze_pair(args.data,
                 "trials_llama3-8b.jsonl", "trials_llama3_1-8b.jsonl",
                 "Llama 3 -> 3.1")
    analyze_pair(args.data,
                 "trials_qwen2_5-7b.jsonl", "trials_qwen3-8b.jsonl",
                 "Qwen 2.5 -> 3")

if __name__ == "__main__":
    main()
