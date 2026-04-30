"""
Beyond the Mean: Figure Generation
====================================
Generates publication-quality figures for the paper.

Requires: matplotlib, numpy, pandas
Usage: python generate_figures.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

DEFAULT_DATA = r"D:\beyond_the_mean\data"
DEFAULT_OUTPUT = r"D:\beyond_the_mean\figures"

K = 10
RCI_THRESHOLD = 1.96


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


def load_pair(data_dir, v1_file, v2_file):
    trials_v1 = load_trials(os.path.join(data_dir, v1_file))
    trials_v2 = load_trials(os.path.join(data_dir, v2_file))
    r_v1 = split_half_reliability(trials_v1)
    r_v2 = split_half_reliability(trials_v2)
    items_v1 = compute_item_accuracy(trials_v1)
    items_v2 = compute_item_accuracy(trials_v2)
    merged = items_v1.merge(items_v2, on=["item_index", "item_id", "domain"], suffixes=("_v1", "_v2"))
    merged["delta"] = merged["p_v2"] - merged["p_v1"]
    sd_v1 = merged["p_v1"].std()
    sd_v2 = merged["p_v2"].std()
    sem_v1 = sd_v1 * np.sqrt(1 - r_v1)
    sem_v2 = sd_v2 * np.sqrt(1 - r_v2)
    s_diff = np.sqrt(sem_v1**2 + sem_v2**2)
    merged["rci"] = merged["delta"] / s_diff
    merged["rci_cat"] = "No reliable change"
    merged.loc[merged["rci"] > RCI_THRESHOLD, "rci_cat"] = "Reliable improvement"
    merged.loc[merged["rci"] < -RCI_THRESHOLD, "rci_cat"] = "Reliable deterioration"
    excl = ~(((merged["p_v1"] == 0.0) & (merged["p_v2"] == 0.0)) |
             ((merged["p_v1"] == 1.0) & (merged["p_v2"] == 1.0)))
    return merged, merged[excl].copy(), s_diff


# ============================================================
# Figure 1: p_v1 vs p_v2 scatter with RCI bands
# ============================================================

def figure1_scatter(merged_post, s_diff, title, filename, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

    colors = {
        "Reliable improvement": "#1D9E75",
        "No reliable change": "#888780",
        "Reliable deterioration": "#E24B4A",
    }

    for cat in ["No reliable change", "Reliable improvement", "Reliable deterioration"]:
        sub = merged_post[merged_post["rci_cat"] == cat]
        ax.scatter(sub["p_v1"], sub["p_v2"], c=colors[cat], s=12, alpha=0.5,
                   label=f"{cat} (n={len(sub)})", edgecolors='none', zorder=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8, zorder=1)

    min_delta = RCI_THRESHOLD * s_diff
    ax.plot([0, 1], [min_delta, 1 + min_delta], color="#1D9E75", alpha=0.2, linewidth=0.8, linestyle=':', zorder=1)
    ax.plot([0, 1], [-min_delta, 1 - min_delta], color="#E24B4A", alpha=0.2, linewidth=0.8, linestyle=':', zorder=1)

    ax.set_xlabel("Accuracy v1 (proportion correct, K=10)")
    ax.set_ylabel("Accuracy v2 (proportion correct, K=10)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"  Saved {filename}")


# ============================================================
# Figure 2: RCI distribution histograms
# ============================================================

def figure2_rci_histogram(merged_post_llama, merged_post_qwen, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, data, title in [
        (ax1, merged_post_llama, "Llama 3 \u2192 3.1"),
        (ax2, merged_post_qwen, "Qwen 2.5 \u2192 3"),
    ]:
        rci = data["rci"].values
        rci_clipped = np.clip(rci, -12, 12)

        ax.hist(rci_clipped, bins=50, color="#B5D4F4", edgecolor="#185FA5", linewidth=0.5, zorder=2)
        ax.axvline(x=1.96, color="#1D9E75", linestyle='--', linewidth=1, alpha=0.8, label="|RCI| = 1.96")
        ax.axvline(x=-1.96, color="#E24B4A", linestyle='--', linewidth=1, alpha=0.8)

        n_imp = (data["rci_cat"] == "Reliable improvement").sum()
        n_det = (data["rci_cat"] == "Reliable deterioration").sum()
        n_nc = (data["rci_cat"] == "No reliable change").sum()

        ax.set_xlabel("RCI value")
        ax.set_ylabel("Number of items")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        stats_text = f"Improved: {n_imp}\nNo change: {n_nc}\nDeteriorated: {n_det}"
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#D3D1C7', alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_rci_distribution.png"))
    plt.close()
    print("  Saved fig2_rci_distribution.png")


# ============================================================
# Figure 3: Domain churn heatmap
# ============================================================

def figure3_domain_heatmap(merged_post_llama, merged_post_qwen, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    domains = ["economics", "law", "physics", "psychology"]

    for ax, data, title in [
        (ax1, merged_post_llama, "Llama 3 \u2192 3.1"),
        (ax2, merged_post_qwen, "Qwen 2.5 \u2192 3"),
    ]:
        matrix = []
        labels = []
        for d in domains:
            sub = data[data["domain"] == d]
            n = len(sub)
            n_imp = (sub["rci_cat"] == "Reliable improvement").sum()
            n_nc = (sub["rci_cat"] == "No reliable change").sum()
            n_det = (sub["rci_cat"] == "Reliable deterioration").sum()
            matrix.append([n_imp/n*100, n_nc/n*100, n_det/n*100])
            labels.append(d.capitalize())

        matrix = np.array(matrix)
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=60)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Improved", "No change", "Deteriorated"], fontsize=8)
        ax.set_yticks(range(len(domains)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')

        for i in range(len(domains)):
            for j in range(3):
                val = matrix[i, j]
                color = 'white' if val > 40 else 'black'
                ax.text(j, i, f"{val:.0f}%", ha='center', va='center', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_domain_heatmap.png"))
    plt.close()
    print("  Saved fig3_domain_heatmap.png")


# ============================================================
# Figure 4: Churn by baseline difficulty
# ============================================================

def figure4_difficulty_churn(merged_post_llama, merged_post_qwen, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bin_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    for ax, data, title in [
        (ax1, merged_post_llama, "Llama 3 \u2192 3.1"),
        (ax2, merged_post_qwen, "Qwen 2.5 \u2192 3"),
    ]:
        imp_pcts = []
        det_pcts = []
        nc_pcts = []

        for (lo, hi) in bins:
            sub = data[(data["p_v1"] >= lo) & (data["p_v1"] < hi)]
            n = len(sub) if len(sub) > 0 else 1
            n_imp = (sub["rci_cat"] == "Reliable improvement").sum()
            n_det = (sub["rci_cat"] == "Reliable deterioration").sum()
            n_nc = (sub["rci_cat"] == "No reliable change").sum()
            imp_pcts.append(n_imp / n * 100)
            det_pcts.append(n_det / n * 100)
            nc_pcts.append(n_nc / n * 100)

        x = np.arange(len(bin_labels))
        width = 0.25

        ax.bar(x - width, imp_pcts, width, color="#1D9E75", label="Improved", zorder=2)
        ax.bar(x, nc_pcts, width, color="#888780", label="No change", zorder=2)
        ax.bar(x + width, det_pcts, width, color="#E24B4A", label="Deteriorated", zorder=2)

        ax.set_xlabel("Baseline accuracy (p in v1)")
        ax.set_ylabel("Percentage of items")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_difficulty_churn.png"))
    plt.close()
    print("  Saved fig4_difficulty_churn.png")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading data...")
    merged_llama, post_llama, s_diff_llama = load_pair(
        args.data, "trials_llama3-8b.jsonl", "trials_llama3_1-8b.jsonl")
    merged_qwen, post_qwen, s_diff_qwen = load_pair(
        args.data, "trials_qwen2_5-7b.jsonl", "trials_qwen3-8b.jsonl")

    print("Generating figures...")
    figure1_scatter(post_llama, s_diff_llama, "Llama 3 \u2192 3.1 (n=952)", "fig1a_scatter_llama.png", args.output)
    figure1_scatter(post_qwen, s_diff_qwen, "Qwen 2.5 \u2192 3 (n=652)", "fig1b_scatter_qwen.png", args.output)
    figure2_rci_histogram(post_llama, post_qwen, args.output)
    figure3_domain_heatmap(post_llama, post_qwen, args.output)
    figure4_difficulty_churn(post_llama, post_qwen, args.output)

    print("\nAll figures saved to", args.output)


if __name__ == "__main__":
    main()
