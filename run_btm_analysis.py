"""
Beyond the Mean: Analysis Script
=================================
Implements all pre-registered analyses for the Reliable Change Index
study on MMLU-Pro model-version comparisons.

Pre-registration: OSF [insert DOI]
Author: Jon-Paul Cacioli (ORCID: 0009-0000-7054-2014)

Usage:
    python run_btm_analysis.py
    python run_btm_analysis.py --data D:\beyond_the_mean\data --output D:\beyond_the_mean\results
"""

import argparse
import json
import os
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# Configuration
# ============================================================

DEFAULT_DATA = r"D:\beyond_the_mean\data"
DEFAULT_OUTPUT = r"D:\beyond_the_mean\results"

K = 10
RCI_THRESHOLD = 1.96
RCI_THRESHOLD_CONSERVATIVE = 2.58
N_SPLITS = 1000        # random split-halves for reliability
N_PERMUTATIONS = 1000  # empirical null calibration
N_BOOTSTRAP = 10000    # bootstrap for breadth ratio CI

MODEL_PAIRS = {
    "llama": {
        "v1": "llama3-8b",
        "v2": "llama3.1-8b",
        "v1_file": "trials_llama3-8b.jsonl",
        "v2_file": "trials_llama3_1-8b.jsonl",
        "label": "Llama 3 -> 3.1 (minor update)",
    },
    "qwen": {
        "v1": "qwen2.5-7b",
        "v2": "qwen3-8b",
        "v1_file": "trials_qwen2_5-7b.jsonl",
        "v2_file": "trials_qwen3-8b.jsonl",
        "label": "Qwen 2.5 -> 3 (generational update)",
    },
}


# ============================================================
# Data Loading
# ============================================================

def load_trials(filepath: str) -> pd.DataFrame:
    """Load trial-level JSONL data."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_item_accuracy(trials: pd.DataFrame) -> pd.DataFrame:
    """Compute per-item accuracy from trial data.
    Returns DataFrame with item_index, domain, p (accuracy), n_valid."""
    valid = trials[~trials["is_missing"]].copy()
    grouped = valid.groupby(["item_index", "domain"]).agg(
        n_correct=("is_correct", "sum"),
        n_valid=("is_correct", "count"),
    ).reset_index()
    grouped["p"] = grouped["n_correct"] / grouped["n_valid"]
    return grouped


# ============================================================
# Reliability Estimation (Section 3.2)
# ============================================================

def split_half_reliability(trials: pd.DataFrame, n_splits: int = N_SPLITS) -> dict:
    """Compute reliability via random split-halves with Spearman-Brown correction.
    Returns dict with median r_xx, CI, and all r_xx values."""

    # Get per-item, per-k correctness matrix
    valid = trials[~trials["is_missing"]].copy()

    # Build item x k matrix of correctness
    pivot = valid.pivot_table(
        index="item_index", columns="k", values="is_correct", aggfunc="first"
    )

    n_items = len(pivot)
    k_cols = sorted(pivot.columns.tolist())
    assert len(k_cols) == K, f"Expected {K} samples per item, got {len(k_cols)}"

    r_xx_values = []
    rng = np.random.RandomState(42)

    for _ in range(n_splits):
        # Random split into two halves
        perm = rng.permutation(k_cols)
        half1 = perm[:5]
        half2 = perm[5:]

        # Compute accuracy for each half
        p_half1 = pivot[half1].mean(axis=1).values
        p_half2 = pivot[half2].mean(axis=1).values

        # Split-half correlation
        r_half = np.corrcoef(p_half1, p_half2)[0, 1]

        # Spearman-Brown correction
        r_xx = (2 * r_half) / (1 + r_half) if (1 + r_half) != 0 else 0
        r_xx_values.append(r_xx)

    r_xx_values = np.array(r_xx_values)

    return {
        "median_r_xx": np.median(r_xx_values),
        "ci_lower": np.percentile(r_xx_values, 2.5),
        "ci_upper": np.percentile(r_xx_values, 97.5),
        "mean_r_xx": np.mean(r_xx_values),
        "sd_r_xx": np.std(r_xx_values),
        "all_r_xx": r_xx_values,
    }


def compute_icc(trials: pd.DataFrame) -> float:
    """Compute ICC (two-way random, single measures) across K samples.
    Uses one-way random effects model as approximation."""
    valid = trials[~trials["is_missing"]].copy()
    pivot = valid.pivot_table(
        index="item_index", columns="k", values="is_correct", aggfunc="first"
    )

    data = pivot.values.astype(float)
    n, k = data.shape

    # Grand mean
    grand_mean = np.nanmean(data)

    # Between-items sum of squares
    item_means = np.nanmean(data, axis=1)
    ss_between = k * np.sum((item_means - grand_mean) ** 2)

    # Within-items sum of squares
    ss_within = np.nansum((data - item_means[:, None]) ** 2)

    # Mean squares
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))

    # ICC(1,1)
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    return icc


# ============================================================
# RCI Computation (Sections 3.3 - 3.5)
# ============================================================

def compute_rci(
    items_v1: pd.DataFrame,
    items_v2: pd.DataFrame,
    r_xx_v1: float,
    r_xx_v2: float,
) -> pd.DataFrame:
    """Compute per-item RCI for a model pair.
    Returns merged DataFrame with RCI values and classifications."""

    merged = items_v1.merge(
        items_v2,
        on=["item_index", "domain"],
        suffixes=("_v1", "_v2"),
    )

    # SEM per model (Section 3.3)
    sd_v1 = merged["p_v1"].std()
    sd_v2 = merged["p_v2"].std()
    sem_v1 = sd_v1 * np.sqrt(1 - r_xx_v1)
    sem_v2 = sd_v2 * np.sqrt(1 - r_xx_v2)

    # S_diff (Section 3.4)
    s_diff = np.sqrt(sem_v1**2 + sem_v2**2)

    # RCI per item (Section 3.5)
    merged["delta"] = merged["p_v2"] - merged["p_v1"]
    merged["rci"] = merged["delta"] / s_diff if s_diff > 0 else 0

    # Classification
    merged["rci_category"] = "no_reliable_change"
    merged.loc[merged["rci"] > RCI_THRESHOLD, "rci_category"] = "reliable_improvement"
    merged.loc[merged["rci"] < -RCI_THRESHOLD, "rci_category"] = "reliable_deterioration"

    # Conservative classification
    merged["rci_category_conservative"] = "no_reliable_change"
    merged.loc[merged["rci"] > RCI_THRESHOLD_CONSERVATIVE, "rci_category_conservative"] = "reliable_improvement"
    merged.loc[merged["rci"] < -RCI_THRESHOLD_CONSERVATIVE, "rci_category_conservative"] = "reliable_deterioration"

    # Store parameters
    merged.attrs["sd_v1"] = sd_v1
    merged.attrs["sd_v2"] = sd_v2
    merged.attrs["sem_v1"] = sem_v1
    merged.attrs["sem_v2"] = sem_v2
    merged.attrs["s_diff"] = s_diff
    merged.attrs["min_delta"] = RCI_THRESHOLD * s_diff

    return merged


def apply_exclusions(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply pre-registered item exclusion rules (Section 2.5).
    Returns (filtered_df, exclusion_report)."""

    n_before = len(merged)

    # (a) Items with <6 valid responses for either model
    mask_valid = (merged["n_valid_v1"] >= 6) & (merged["n_valid_v2"] >= 6)

    # (b) Floor/ceiling in both versions
    mask_floor_ceil = ~(
        ((merged["p_v1"] == 0.0) & (merged["p_v2"] == 0.0)) |
        ((merged["p_v1"] == 1.0) & (merged["p_v2"] == 1.0))
    )

    mask = mask_valid & mask_floor_ceil
    filtered = merged[mask].copy()

    excluded_valid = (~mask_valid).sum()
    excluded_floor_ceil = (mask_valid & ~mask_floor_ceil).sum()
    n_after = len(filtered)

    report = {
        "n_before": n_before,
        "n_after": n_after,
        "excluded_low_valid": int(excluded_valid),
        "excluded_floor_ceiling": int(excluded_floor_ceil),
        "excluded_total": n_before - n_after,
    }

    return filtered, report


# ============================================================
# Primary Analyses (Section 5.1)
# ============================================================

def wilson_ci(count: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = count / total
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p * (1 - p) / total) + z**2 / (4 * total**2)) / denom
    return (max(0, centre - margin), min(1, centre + margin))


def analyze_rci_classification(merged: pd.DataFrame, label: str) -> dict:
    """H1 and H4: RCI classification and breadth ratio."""
    n = len(merged)
    s_diff = merged.attrs["s_diff"]
    min_delta = merged.attrs["min_delta"]

    # Category counts
    n_improved = (merged["rci_category"] == "reliable_improvement").sum()
    n_no_change = (merged["rci_category"] == "no_reliable_change").sum()
    n_deteriorated = (merged["rci_category"] == "reliable_deterioration").sum()

    # Conservative counts
    n_improved_cons = (merged["rci_category_conservative"] == "reliable_improvement").sum()
    n_no_change_cons = (merged["rci_category_conservative"] == "no_reliable_change").sum()
    n_deteriorated_cons = (merged["rci_category_conservative"] == "reliable_deterioration").sum()

    # Proportions with Wilson CIs
    prop_improved = n_improved / n
    prop_no_change = n_no_change / n
    prop_deteriorated = n_deteriorated / n

    ci_improved = wilson_ci(n_improved, n)
    ci_no_change = wilson_ci(n_no_change, n)
    ci_deteriorated = wilson_ci(n_deteriorated, n)

    # Aggregate accuracy difference
    mean_delta = merged["delta"].mean()
    delta_ci = stats.t.interval(
        0.95, df=n-1,
        loc=mean_delta,
        scale=merged["delta"].std() / np.sqrt(n)
    )

    # Breadth ratio (H4)
    n_any_positive = (merged["delta"] > 0).sum()
    breadth_ratio = n_improved / n_any_positive if n_any_positive > 0 else 0

    # Bootstrap CI for breadth ratio
    rng = np.random.RandomState(42)
    boot_ratios = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        sample = merged.iloc[idx]
        s_imp = (sample["rci_category"] == "reliable_improvement").sum()
        s_pos = (sample["delta"] > 0).sum()
        boot_ratios.append(s_imp / s_pos if s_pos > 0 else 0)
    boot_ratios = np.array(boot_ratios)
    breadth_ci = (np.percentile(boot_ratios, 2.5), np.percentile(boot_ratios, 97.5))

    # Minimum detectable delta in samples
    min_delta_samples = int(np.ceil(min_delta * K))

    result = {
        "pair": label,
        "n_items": n,
        "s_diff": round(s_diff, 4),
        "min_delta_p": round(min_delta, 4),
        "min_delta_samples": min_delta_samples,
        "sem_v1": round(merged.attrs["sem_v1"], 4),
        "sem_v2": round(merged.attrs["sem_v2"], 4),
        "aggregate_delta": round(mean_delta, 4),
        "aggregate_delta_ci": (round(delta_ci[0], 4), round(delta_ci[1], 4)),
        "n_improved": int(n_improved),
        "n_no_change": int(n_no_change),
        "n_deteriorated": int(n_deteriorated),
        "prop_improved": round(prop_improved, 4),
        "prop_no_change": round(prop_no_change, 4),
        "prop_deteriorated": round(prop_deteriorated, 4),
        "ci_improved": (round(ci_improved[0], 4), round(ci_improved[1], 4)),
        "ci_no_change": (round(ci_no_change[0], 4), round(ci_no_change[1], 4)),
        "ci_deteriorated": (round(ci_deteriorated[0], 4), round(ci_deteriorated[1], 4)),
        "n_improved_conservative": int(n_improved_cons),
        "n_no_change_conservative": int(n_no_change_cons),
        "n_deteriorated_conservative": int(n_deteriorated_cons),
        "h1_supported": prop_no_change > 0.50,
        "n_any_positive_delta": int(n_any_positive),
        "breadth_ratio": round(breadth_ratio, 4),
        "breadth_ratio_ci": (round(breadth_ci[0], 4), round(breadth_ci[1], 4)),
        "h4_supported": breadth_ratio < 0.50,
    }

    return result


# ============================================================
# Empirical Null Calibration (Section 5.1.3)
# ============================================================

def empirical_null(
    trials_v1: pd.DataFrame,
    trials_v2: pd.DataFrame,
    r_xx_v1: float,
    r_xx_v2: float,
    n_perms: int = N_PERMUTATIONS,
) -> dict:
    """Permutation test: shuffle version labels at block level."""

    # Get per-item accuracy for each model
    items_v1 = compute_item_accuracy(trials_v1)
    items_v2 = compute_item_accuracy(trials_v2)

    # Stack all trials by item
    trials_v1_copy = trials_v1.copy()
    trials_v2_copy = trials_v2.copy()
    trials_v1_copy["version"] = "v1"
    trials_v2_copy["version"] = "v2"

    rng = np.random.RandomState(42)
    null_improved = []
    null_deteriorated = []

    for _ in range(n_perms):
        # For each item, randomly swap which block is v1 vs v2
        swap_mask = rng.randint(0, 2, size=2000).astype(bool)

        perm_v1 = items_v1.copy()
        perm_v2 = items_v2.copy()

        # Swap p values for items where swap_mask is True
        merged_temp = perm_v1.merge(perm_v2, on=["item_index", "domain"], suffixes=("_v1", "_v2"))
        swapped_p_v1 = np.where(swap_mask[:len(merged_temp)], merged_temp["p_v2"].values, merged_temp["p_v1"].values)
        swapped_p_v2 = np.where(swap_mask[:len(merged_temp)], merged_temp["p_v1"].values, merged_temp["p_v2"].values)

        # Recompute SEM and S_diff under permutation
        sd1 = np.std(swapped_p_v1, ddof=1)
        sd2 = np.std(swapped_p_v2, ddof=1)
        sem1 = sd1 * np.sqrt(1 - r_xx_v1)
        sem2 = sd2 * np.sqrt(1 - r_xx_v2)
        s_diff_perm = np.sqrt(sem1**2 + sem2**2)

        if s_diff_perm > 0:
            rci_perm = (swapped_p_v2 - swapped_p_v1) / s_diff_perm
            null_improved.append(np.sum(rci_perm > RCI_THRESHOLD))
            null_deteriorated.append(np.sum(rci_perm < -RCI_THRESHOLD))
        else:
            null_improved.append(0)
            null_deteriorated.append(0)

    null_improved = np.array(null_improved)
    null_deteriorated = np.array(null_deteriorated)

    return {
        "null_improved_mean": round(np.mean(null_improved), 2),
        "null_improved_95th": int(np.percentile(null_improved, 95)),
        "null_deteriorated_mean": round(np.mean(null_deteriorated), 2),
        "null_deteriorated_95th": int(np.percentile(null_deteriorated, 95)),
        "null_improved_all": null_improved,
        "null_deteriorated_all": null_deteriorated,
    }


# ============================================================
# Secondary Analyses (Section 5.2)
# ============================================================

def domain_analysis(merged: pd.DataFrame) -> dict:
    """H2: Domain x RCI category chi-squared."""
    domains = sorted(merged["domain"].unique())
    categories = ["reliable_improvement", "no_reliable_change", "reliable_deterioration"]

    # Contingency table
    ct = pd.crosstab(merged["domain"], merged["rci_category"])
    for cat in categories:
        if cat not in ct.columns:
            ct[cat] = 0
    ct = ct[categories]

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = len(merged)
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0

    # Per-domain proportions
    domain_props = {}
    for d in domains:
        d_data = merged[merged["domain"] == d]
        n_d = len(d_data)
        n_imp = (d_data["rci_category"] == "reliable_improvement").sum()
        n_det = (d_data["rci_category"] == "reliable_deterioration").sum()
        n_nc = (d_data["rci_category"] == "no_reliable_change").sum()
        domain_props[d] = {
            "n": n_d,
            "improved": int(n_imp),
            "no_change": int(n_nc),
            "deteriorated": int(n_det),
            "prop_improved": round(n_imp / n_d, 4) if n_d > 0 else 0,
            "prop_deteriorated": round(n_det / n_d, 4) if n_d > 0 else 0,
            "mean_accuracy_v1": round(d_data["p_v1"].mean(), 4),
            "mean_accuracy_v2": round(d_data["p_v2"].mean(), 4),
            "mean_delta": round(d_data["delta"].mean(), 4),
        }

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "dof": dof,
        "cramers_v": round(cramers_v, 4),
        "h2_supported": p < 0.05,
        "domain_props": domain_props,
    }


def cross_pair_comparison(results_llama: dict, results_qwen: dict) -> dict:
    """H3: Two-proportion z-test on reliable-change rates."""
    n1 = results_llama["n_items"]
    n2 = results_qwen["n_items"]
    x1 = results_llama["n_improved"] + results_llama["n_deteriorated"]
    x2 = results_qwen["n_improved"] + results_qwen["n_deteriorated"]

    p1 = x1 / n1
    p2 = x2 / n2
    p_pooled = (x1 + x2) / (n1 + n2)

    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2)) if p_pooled > 0 and p_pooled < 1 else 1
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "llama_reliable_change_rate": round(p1, 4),
        "qwen_reliable_change_rate": round(p2, 4),
        "z_statistic": round(z, 4),
        "p_value": round(p_val, 6),
        "h3_supported": p_val < 0.05,
    }


def item_characteristics_regression(merged: pd.DataFrame) -> dict:
    """Logistic regression: reliable_change ~ difficulty + domain."""
    merged = merged.copy()
    merged["difficulty"] = (merged["p_v1"] + merged["p_v2"]) / 2
    merged["reliable_change"] = (merged["rci_category"] != "no_reliable_change").astype(int)

    # Check if we have enough events
    n_events = merged["reliable_change"].sum()
    if n_events < 10:
        return {"skipped": True, "reason": f"Only {n_events} events, insufficient for regression"}

    # Domain dummies
    domains = sorted(merged["domain"].unique())
    ref_domain = domains[0]
    for d in domains[1:]:
        merged[f"domain_{d}"] = (merged["domain"] == d).astype(int)

    # Simple logistic via statsmodels-free approach
    from scipy.optimize import minimize

    X_cols = ["difficulty"] + [f"domain_{d}" for d in domains[1:]]
    X = merged[X_cols].values
    X = np.column_stack([np.ones(len(X)), X])  # intercept
    y = merged["reliable_change"].values

    def neg_log_lik(beta):
        z = X @ beta
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize(neg_log_lik, np.zeros(X.shape[1]), method="BFGS")
    betas = result.x

    # Odds ratios
    or_dict = {"intercept": round(np.exp(betas[0]), 4)}
    or_dict["difficulty"] = round(np.exp(betas[1]), 4)
    for i, d in enumerate(domains[1:]):
        or_dict[f"domain_{d}_vs_{ref_domain}"] = round(np.exp(betas[2 + i]), 4)

    return {
        "skipped": False,
        "odds_ratios": or_dict,
        "coefficients": {name: round(b, 4) for name, b in zip(["intercept", "difficulty"] + [f"domain_{d}" for d in domains[1:]], betas)},
        "reference_domain": ref_domain,
        "n_events": int(n_events),
        "converged": result.success,
    }


# ============================================================
# Exploratory Analyses (Section 5.3)
# ============================================================

def reliability_vs_k(trials: pd.DataFrame) -> dict:
    """Reliability as a function of K (Section 5.3.1)."""
    valid = trials[~trials["is_missing"]].copy()
    pivot = valid.pivot_table(
        index="item_index", columns="k", values="is_correct", aggfunc="first"
    )
    k_cols = sorted(pivot.columns.tolist())

    results = {}
    for k_subset in [4, 6, 8, 10]:
        if k_subset > len(k_cols):
            continue
        cols = k_cols[:k_subset]
        sub = pivot[cols]

        r_xx_vals = []
        rng = np.random.RandomState(42)
        half_size = k_subset // 2

        for _ in range(200):  # fewer splits for speed
            perm = rng.permutation(cols)
            h1 = perm[:half_size]
            h2 = perm[half_size:2*half_size]
            p1 = sub[h1].mean(axis=1).values
            p2 = sub[h2].mean(axis=1).values
            r_half = np.corrcoef(p1, p2)[0, 1]
            r_xx = (2 * r_half) / (1 + r_half)
            r_xx_vals.append(r_xx)

        results[k_subset] = {
            "median_r_xx": round(np.median(r_xx_vals), 4),
            "ci": (round(np.percentile(r_xx_vals, 2.5), 4), round(np.percentile(r_xx_vals, 97.5), 4)),
        }

    # Spearman-Brown extrapolation to target reliabilities
    if 10 in results:
        r10 = results[10]["median_r_xx"]
        # r_xx for K=n is: r_n = (n/10 * r10) / (1 + (n/10 - 1) * r10)
        # Solve for n given target r: n = 10 * r_target * (1 - r10) / (r10 * (1 - r_target))
        for target in [0.80, 0.90]:
            if r10 > 0 and r10 < 1:
                k_needed = 10 * target * (1 - r10) / (r10 * (1 - target))
                results[f"k_for_r{int(target*100)}"] = int(np.ceil(k_needed))

    return results


def cross_pair_item_correlation(merged_llama: pd.DataFrame, merged_qwen: pd.DataFrame) -> dict:
    """Cross-pair item-level RCI correlation (Section 5.3.2)."""
    combined = merged_llama[["item_index", "rci"]].merge(
        merged_qwen[["item_index", "rci"]],
        on="item_index",
        suffixes=("_llama", "_qwen"),
    )
    r, p = stats.pearsonr(combined["rci_llama"], combined["rci_qwen"])
    return {
        "pearson_r": round(r, 4),
        "p_value": round(p, 6),
        "n_items": len(combined),
    }


def stratified_sensitivity(merged: pd.DataFrame) -> dict:
    """Heteroscedasticity sensitivity: difficulty-stratified S_diff (Section 5.3.4)."""
    merged = merged.copy()
    merged["difficulty"] = (merged["p_v1"] + merged["p_v2"]) / 2

    # Terciles (drop duplicates if many items share same difficulty)
    try:
        merged["tercile"] = pd.qcut(merged["difficulty"], 3, labels=["hard", "medium", "easy"])
    except ValueError:
        merged["tercile"] = pd.qcut(merged["difficulty"], 3, labels=False, duplicates="drop")
        # Map numeric bins to labels
        label_map = {v: l for v, l in zip(sorted(merged["tercile"].unique()), ["hard", "medium", "easy"])}
        merged["tercile"] = merged["tercile"].map(label_map)
        # If only 2 bins resulted, fill missing label
        if merged["tercile"].isna().any():
            merged["tercile"] = merged["tercile"].fillna("medium")

    results = {}
    for tercile in ["hard", "medium", "easy"]:
        sub = merged[merged["tercile"] == tercile]
        sd_v1 = sub["p_v1"].std()
        sd_v2 = sub["p_v2"].std()
        # Use global reliability (best available)
        r_xx_v1 = merged.attrs.get("r_xx_v1", 0.8)
        r_xx_v2 = merged.attrs.get("r_xx_v2", 0.8)
        sem_v1 = sd_v1 * np.sqrt(1 - r_xx_v1)
        sem_v2 = sd_v2 * np.sqrt(1 - r_xx_v2)
        s_diff_strat = np.sqrt(sem_v1**2 + sem_v2**2)

        # Reclassify with stratified S_diff
        if s_diff_strat > 0:
            rci_strat = sub["delta"].values / s_diff_strat
            n_imp = int(np.sum(rci_strat > RCI_THRESHOLD))
            n_det = int(np.sum(rci_strat < -RCI_THRESHOLD))
        else:
            n_imp = 0
            n_det = 0

        results[tercile] = {
            "n_items": len(sub),
            "sd_v1": round(sd_v1, 4),
            "sd_v2": round(sd_v2, 4),
            "s_diff": round(s_diff_strat, 4),
            "min_delta": round(RCI_THRESHOLD * s_diff_strat, 4),
            "n_improved": int(n_imp),
            "n_deteriorated": int(n_det),
            "mean_difficulty": round(sub["difficulty"].mean(), 4),
        }

    return results


# ============================================================
# Reporting
# ============================================================

def print_results(pair_name: str, pair_config: dict, reliability_v1: dict,
                  reliability_v2: dict, icc_v1: float, icc_v2: float,
                  rci_results: dict, rci_results_pre: dict,
                  exclusion_report: dict, null_results: dict,
                  domain_results: dict, regression_results: dict,
                  reliability_k: dict, stratified: dict):
    """Print formatted results for one model pair."""

    print(f"\n{'#'*70}")
    print(f"# {pair_config['label']}")
    print(f"{'#'*70}")

    # Reliability
    print(f"\n--- Reliability ---")
    print(f"  {pair_config['v1']}:")
    print(f"    Split-half r_xx: {reliability_v1['median_r_xx']:.4f} "
          f"[{reliability_v1['ci_lower']:.4f}, {reliability_v1['ci_upper']:.4f}]")
    print(f"    ICC: {icc_v1:.4f}")
    print(f"  {pair_config['v2']}:")
    print(f"    Split-half r_xx: {reliability_v2['median_r_xx']:.4f} "
          f"[{reliability_v2['ci_lower']:.4f}, {reliability_v2['ci_upper']:.4f}]")
    print(f"    ICC: {icc_v2:.4f}")

    # Exclusions
    print(f"\n--- Exclusions ---")
    print(f"  Before: {exclusion_report['n_before']}")
    print(f"  After:  {exclusion_report['n_after']}")
    print(f"  Excluded (low valid): {exclusion_report['excluded_low_valid']}")
    print(f"  Excluded (floor/ceiling): {exclusion_report['excluded_floor_ceiling']}")

    # Measurement parameters
    print(f"\n--- Measurement Parameters ---")
    print(f"  SEM v1: {rci_results['sem_v1']}")
    print(f"  SEM v2: {rci_results['sem_v2']}")
    print(f"  S_diff: {rci_results['s_diff']}")
    print(f"  Min detectable delta: {rci_results['min_delta_p']} p-units "
          f"({rci_results['min_delta_samples']} additional correct out of {K})")

    # RCI Classification (post-exclusion)
    print(f"\n--- RCI Classification (post-exclusion, n={rci_results['n_items']}) ---")
    print(f"  Aggregate delta: {rci_results['aggregate_delta']} "
          f"{rci_results['aggregate_delta_ci']}")
    print(f"  Reliable improvement:   {rci_results['n_improved']} "
          f"({rci_results['prop_improved']:.1%}) {rci_results['ci_improved']}")
    print(f"  No reliable change:     {rci_results['n_no_change']} "
          f"({rci_results['prop_no_change']:.1%}) {rci_results['ci_no_change']}")
    print(f"  Reliable deterioration: {rci_results['n_deteriorated']} "
          f"({rci_results['prop_deteriorated']:.1%}) {rci_results['ci_deteriorated']}")

    # Pre-exclusion
    print(f"\n  Pre-exclusion (n={rci_results_pre['n_items']}):")
    print(f"    Reliable improvement:   {rci_results_pre['n_improved']} ({rci_results_pre['prop_improved']:.1%})")
    print(f"    No reliable change:     {rci_results_pre['n_no_change']} ({rci_results_pre['prop_no_change']:.1%})")
    print(f"    Reliable deterioration: {rci_results_pre['n_deteriorated']} ({rci_results_pre['prop_deteriorated']:.1%})")

    # Conservative threshold
    print(f"\n  Conservative threshold (|RCI| > 2.58):")
    print(f"    Reliable improvement:   {rci_results['n_improved_conservative']}")
    print(f"    No reliable change:     {rci_results['n_no_change_conservative']}")
    print(f"    Reliable deterioration: {rci_results['n_deteriorated_conservative']}")

    # Hypotheses
    print(f"\n--- Hypothesis Tests ---")
    print(f"  H1 (majority no change): {'SUPPORTED' if rci_results['h1_supported'] else 'NOT SUPPORTED'} "
          f"({rci_results['prop_no_change']:.1%} > 50%)")
    print(f"  H4 (breadth ratio < 0.50): {'SUPPORTED' if rci_results['h4_supported'] else 'NOT SUPPORTED'} "
          f"(ratio = {rci_results['breadth_ratio']:.4f} {rci_results['breadth_ratio_ci']})")

    # Empirical null
    print(f"\n--- Empirical Null Calibration ---")
    print(f"  Null improved (mean): {null_results['null_improved_mean']}")
    print(f"  Null improved (95th pctl): {null_results['null_improved_95th']}")
    print(f"  Observed improved: {rci_results['n_improved']}")
    print(f"  Exceeds null: {rci_results['n_improved'] > null_results['null_improved_95th']}")
    print(f"  Null deteriorated (mean): {null_results['null_deteriorated_mean']}")
    print(f"  Null deteriorated (95th pctl): {null_results['null_deteriorated_95th']}")
    print(f"  Observed deteriorated: {rci_results['n_deteriorated']}")
    print(f"  Exceeds null: {rci_results['n_deteriorated'] > null_results['null_deteriorated_95th']}")

    # Domain analysis
    print(f"\n--- Domain Analysis (H2) ---")
    print(f"  Chi-squared: {domain_results['chi2']}, p = {domain_results['p_value']}")
    print(f"  Cramer's V: {domain_results['cramers_v']}")
    print(f"  H2 supported: {domain_results['h2_supported']}")
    for d, props in domain_results["domain_props"].items():
        print(f"  {d}: improved={props['improved']}, "
              f"deteriorated={props['deteriorated']}, "
              f"delta={props['mean_delta']}")

    # Regression
    print(f"\n--- Item Characteristics Regression ---")
    if regression_results.get("skipped"):
        print(f"  Skipped: {regression_results['reason']}")
    else:
        print(f"  Events: {regression_results['n_events']}")
        print(f"  Odds ratios: {regression_results['odds_ratios']}")

    # Reliability vs K
    print(f"\n--- Reliability vs K ---")
    for k_val, r_data in reliability_k.items():
        if isinstance(k_val, int):
            print(f"  K={k_val}: r_xx = {r_data['median_r_xx']:.4f} {r_data['ci']}")
    if "k_for_r80" in reliability_k:
        print(f"  K needed for r_xx=.80: {reliability_k['k_for_r80']}")
    if "k_for_r90" in reliability_k:
        print(f"  K needed for r_xx=.90: {reliability_k['k_for_r90']}")

    # Stratified sensitivity
    print(f"\n--- Difficulty-Stratified Sensitivity ---")
    for tercile, data in stratified.items():
        print(f"  {tercile}: S_diff={data['s_diff']}, min_delta={data['min_delta']}, "
              f"improved={data['n_improved']}, deteriorated={data['n_deteriorated']}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Beyond the Mean: RCI Analysis")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Data directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_results = {}

    for pair_name, pair_config in MODEL_PAIRS.items():
        print(f"\nLoading data for {pair_config['label']}...")

        # Load trials
        trials_v1 = load_trials(os.path.join(args.data, pair_config["v1_file"]))
        trials_v2 = load_trials(os.path.join(args.data, pair_config["v2_file"]))

        print(f"  v1: {len(trials_v1)} trials")
        print(f"  v2: {len(trials_v2)} trials")

        # Reliability (Section 3.2)
        print("Computing reliability...")
        rel_v1 = split_half_reliability(trials_v1)
        rel_v2 = split_half_reliability(trials_v2)
        icc_v1 = compute_icc(trials_v1)
        icc_v2 = compute_icc(trials_v2)

        # Item-level accuracy
        items_v1 = compute_item_accuracy(trials_v1)
        items_v2 = compute_item_accuracy(trials_v2)

        # RCI (pre-exclusion)
        print("Computing RCI...")
        merged_pre = compute_rci(items_v1, items_v2, rel_v1["median_r_xx"], rel_v2["median_r_xx"])
        rci_results_pre = analyze_rci_classification(merged_pre, pair_config["label"])

        # Apply exclusions
        merged_post, exclusion_report = apply_exclusions(merged_pre)
        # Recompute RCI on post-exclusion set
        merged_post_rci = compute_rci(
            items_v1[items_v1["item_index"].isin(merged_post["item_index"])],
            items_v2[items_v2["item_index"].isin(merged_post["item_index"])],
            rel_v1["median_r_xx"],
            rel_v2["median_r_xx"],
        )
        # Store reliability in attrs for stratified analysis
        merged_post_rci.attrs["r_xx_v1"] = rel_v1["median_r_xx"]
        merged_post_rci.attrs["r_xx_v2"] = rel_v2["median_r_xx"]

        rci_results_post = analyze_rci_classification(merged_post_rci, pair_config["label"])

        # Empirical null (Section 5.1.3)
        print("Running empirical null calibration (1000 permutations)...")
        null_results = empirical_null(trials_v1, trials_v2, rel_v1["median_r_xx"], rel_v2["median_r_xx"])

        # Domain analysis (H2, Section 5.2.1)
        print("Running domain analysis...")
        domain_results = domain_analysis(merged_post_rci)

        # Item characteristics regression (Section 5.2.3)
        print("Running item characteristics regression...")
        regression_results = item_characteristics_regression(merged_post_rci)

        # Reliability vs K (Section 5.3.1)
        print("Computing reliability vs K...")
        rel_k_v1 = reliability_vs_k(trials_v1)

        # Stratified sensitivity (Section 5.3.4)
        print("Running stratified sensitivity...")
        stratified = stratified_sensitivity(merged_post_rci)

        # Print results
        print_results(
            pair_name, pair_config,
            rel_v1, rel_v2, icc_v1, icc_v2,
            rci_results_post, rci_results_pre,
            exclusion_report, null_results,
            domain_results, regression_results,
            rel_k_v1, stratified,
        )

        # Store for cross-pair analysis
        all_results[pair_name] = {
            "merged": merged_post_rci,
            "rci_results": rci_results_post,
            "reliability_v1": rel_v1,
            "reliability_v2": rel_v2,
        }

    # Cross-pair analyses
    print(f"\n{'#'*70}")
    print(f"# Cross-Pair Analyses")
    print(f"{'#'*70}")

    # H3: Cross-pair comparison (Section 5.2.2)
    h3 = cross_pair_comparison(
        all_results["llama"]["rci_results"],
        all_results["qwen"]["rci_results"],
    )
    print(f"\n--- H3: Cross-Pair Divergence ---")
    print(f"  Llama reliable change rate: {h3['llama_reliable_change_rate']}")
    print(f"  Qwen reliable change rate:  {h3['qwen_reliable_change_rate']}")
    print(f"  z = {h3['z_statistic']}, p = {h3['p_value']}")
    print(f"  H3 supported: {h3['h3_supported']}")

    # Cross-pair item correlation (Section 5.3.2)
    cross_corr = cross_pair_item_correlation(
        all_results["llama"]["merged"],
        all_results["qwen"]["merged"],
    )
    print(f"\n--- Cross-Pair Item-Level RCI Correlation ---")
    print(f"  Pearson r = {cross_corr['pearson_r']}, p = {cross_corr['p_value']}")
    print(f"  n = {cross_corr['n_items']}")

    # Save all results to JSON
    output_file = os.path.join(args.output, "btm_results.json")
    save_results = {
        "llama": {k: v for k, v in all_results["llama"]["rci_results"].items()},
        "qwen": {k: v for k, v in all_results["qwen"]["rci_results"].items()},
        "h3": h3,
        "cross_correlation": cross_corr,
    }
    with open(output_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
