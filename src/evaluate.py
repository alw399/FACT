import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from matplotlib import patches as mpatches

from train import evaluate_model_metrics


def evaluate_baselines(model, results_dir="results", device="mps"):
    """
    Evaluates a model across train/val/test splits and three synthetic baselines
    (Permuted, Gaussian, Uniform), comparing profile JSD distributions and
    log-count correlations.

    Args:
        model:       Trained model with .train_loader, .val_loader, .test_loader,
                     and .min_profile attributes.
        results_dir: Directory to save the JSD plot (default: "results").
        device:      Device to run evaluation on (default: "mps").

    Returns:
        df_corr (pd.DataFrame):       DataFrame with Pearson and Spearman correlations
                                      for each split, with columns:
                                      ['Split', 'Pearson', 'Spearman'].
        loader_metrics (dict):        Raw metrics dict keyed by split name, for
                                      downstream plotting (e.g. plot_counts_correlation).
    """
    loaders = {
        "Train": model.train_loader,
        "Validation": model.val_loader,
        "Test": model.test_loader,
    }

    # Get a reference batch for shapes/values
    sample_batch = next(iter(model.test_loader))
    *x_ref, y_ref = sample_batch

    # 1. Permuted Baseline (shuffled inputs)
    perm_idx = torch.randperm(x_ref[0].size(0))
    perm_x = [x[perm_idx] for x in x_ref]
    perm_loader = DataLoader(
        torch.utils.data.TensorDataset(*perm_x, y_ref), batch_size=32
    )

    # 2. Gaussian Baseline
    gauss_x = [torch.randn_like(x) for x in x_ref]
    gauss_loader = DataLoader(
        torch.utils.data.TensorDataset(*gauss_x, y_ref), batch_size=32
    )

    # 3. Uniform Baseline (values between 0 and 1)
    unif_x = [torch.rand_like(x) for x in x_ref]
    unif_loader = DataLoader(
        torch.utils.data.TensorDataset(*unif_x, y_ref), batch_size=32
    )

    loaders["Uniform"] = unif_loader
    loaders["Permuted"] = perm_loader
    loaders["Gaussian"] = gauss_loader

    correlations = []
    loader_metrics = {}
    plt.figure(figsize=(10, 6))

    for split_name, loader in loaders.items():
        if loader is None:
            continue

        print(f"Evaluating {split_name} split...")
        metrics = evaluate_model_metrics(
            model,
            loader,
            device=device,
            min_profile=model.min_profile,
            smooth_true=True,
        )

        loader_metrics[split_name] = metrics

        pcorr, _ = pearsonr(metrics["log_counts_true"], metrics["log_counts_pred"])
        scorr, _ = spearmanr(metrics["log_counts_true"], metrics["log_counts_pred"])

        correlations.append(
            {"Split": split_name, "Pearson": pcorr, "Spearman": scorr}
        )

        sns.kdeplot(
            data=metrics["profile_jsd"], label=split_name, fill=True, alpha=0.3
        )

    plt.title("Jensen-Shannon Distance Distribution by Split")
    plt.xlabel("Jensen-Shannon Distance")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/jsd.png")
    plt.show()

    df_corr = pd.DataFrame(correlations)

    return df_corr, loader_metrics


def plot_counts_correlation(loader_metrics, model_name, results_dir="results"):
    """
    Plots a grid of true vs predicted log counts scatter plots, one panel per
    split, with a y=x reference line and Pearson r annotation.

    Args:
        loader_metrics (dict):  Dict of split_name -> metrics, as returned by
                                evaluate_baselines.
        model_name (str):       Model name used in the figure title.
        results_dir (str):      Directory to save the plot (default: "results").
    """
    n = len(loader_metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 7 * nrows / 2))
    fig.patch.set_facecolor("#f8f9fa")
    axs = axs.flatten()

    for i, (split_name, metrics) in enumerate(loader_metrics.items()):
        ax = axs[i]
        x = np.array(metrics["log_counts_true"])
        y = np.array(metrics["log_counts_pred"])

        ax.scatter(x, y, alpha=0.4, s=12, color="#1976d2", edgecolors="none", rasterized=True)

        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, color="#e53935", lw=1.2, ls="--", zorder=3, label="y = x")

        r, _ = pearsonr(x, y)
        ax.text(
            0.05, 0.93, f"r = {r:.3f}",
            transform=ax.transAxes,
            fontsize=9, color="#212121",
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
        )

        ax.set_facecolor("#ffffff")
        ax.set_title(split_name, fontsize=11, color="#212121", pad=6)
        ax.set_xlabel("True log counts", fontsize=9, color="#555555")
        ax.set_ylabel("Predicted log counts", fontsize=9, color="#555555")
        ax.tick_params(colors="#555555", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#dddddd")
        ax.grid(True, color="#eeeeee", lw=0.7, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    fig.suptitle(
        f"{model_name}\nLog Counts: True vs Predicted",
        fontsize=13, color="#212121", y=1,
    )
    plt.tight_layout()
    plt.savefig(f"{results_dir}/counts_correlation.png")
    plt.show()


def evaluate_baselines_sig(loader_metrics, results_dir="results", alpha=0.05):
    """
    Produces a statistically rigorous boxplot of profile JSD distributions across
    splits, with pairwise Mann-Whitney U tests and Benjamini-Hochberg FDR correction.
    Significant pairs are annotated with brackets and corrected p-values.
 
    Uses the loader_metrics dict already computed by evaluate_baselines, so no
    additional model evaluation is required.
 
    Args:
        loader_metrics (dict):  Dict of split_name -> metrics, as returned by
                                evaluate_baselines.
        results_dir (str):      Directory to save the plot (default: "results").
        alpha (float):          FDR significance threshold (default: 0.05).
 
    Returns:
        df_stats (pd.DataFrame): Pairwise test results with columns:
                                 ['Group A', 'Group B', 'U statistic',
                                  'p (raw)', 'p (BH corrected)', 'Significant'].
    """
    # ------------------------------------------------------------------ #
    # 1. Collect JSD arrays and run all pairwise Mann-Whitney U tests      #
    # ------------------------------------------------------------------ #
    split_names = list(loader_metrics.keys())
    jsd_arrays = {k: np.array(v["profile_jsd"]) for k, v in loader_metrics.items()}
 
    baseline_names = {"Permuted", "Gaussian", "Uniform"}
    real_splits = [n for n in split_names if n not in baseline_names]
 
    real_splits = [n for n in split_names if n not in baseline_names]
    pairs = [
        (real, base)
        for real in real_splits
        for base in split_names
        if base in baseline_names
    ]
    raw_pvals, u_stats = [], []
    for a, b in pairs:
        u, p = mannwhitneyu(jsd_arrays[a], jsd_arrays[b], alternative="two-sided")
        u_stats.append(u)
        raw_pvals.append(p)
 
    # Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_bh")
 
    df_stats = pd.DataFrame({
        "Group A": [a for a, _ in pairs],
        "Group B": [b for _, b in pairs],
        "U statistic": u_stats,
        "p (raw)": raw_pvals,
        "p (BH corrected)": pvals_corrected,
        "Significant": reject,
    })
 
    # ------------------------------------------------------------------ #
    # 2. Build a long-form DataFrame for seaborn                           #
    # ------------------------------------------------------------------ #
    records = []
    for name, arr in jsd_arrays.items():
        for val in arr:
            records.append({"Split": name, "JSD": val})
    df_long = pd.DataFrame(records)
 
    # ------------------------------------------------------------------ #
    # 3. Plot                                                              #
    # ------------------------------------------------------------------ #
    # Colour palette: real splits get a blue family, baselines get greys
    palette = {
        name: "#1976d2" if name not in baseline_names else "#9e9e9e"
        for name in split_names
    }
 
    fig, ax = plt.subplots(figsize=(max(6, len(split_names) * 1), 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
 
    sns.boxplot(
        data=df_long,
        x="Split", y="JSD",
        hue="Split",
        legend=False,
        order=split_names,
        palette=palette,
        width=0.5,
        linewidth=1.2,
        flierprops=dict(marker="o", markersize=2, alpha=0.3, markeredgewidth=0),
        ax=ax,
    )
    sns.stripplot(
        data=df_long,
        x="Split", y="JSD",
        order=split_names,
        color="black", alpha=0.08, size=2, jitter=True, ax=ax,
    )
 
    # ------------------------------------------------------------------ #
    # 4. Annotate significant pairs with brackets                          #
    # ------------------------------------------------------------------ #
    x_positions = {name: i for i, name in enumerate(split_names)}
    y_max = df_long["JSD"].max()
    y_range = df_long["JSD"].max() - df_long["JSD"].min()
    bracket_step = y_range * 0.08   # vertical spacing between brackets
    bracket_height = y_range * 0.02
 
    sig_pairs = df_stats[df_stats["Significant"]]
    for level, (_, row) in enumerate(sig_pairs.iterrows()):
        x1, x2 = x_positions[row["Group A"]], x_positions[row["Group B"]]
        y = y_max + bracket_step * (level + 1)
        p_corr = row["p (BH corrected)"]
 
        # significance stars
        if p_corr < 0.001:
            label = "***"
        elif p_corr < 0.01:
            label = "**"
        else:
            label = f"* p={p_corr:.3f}"
 
        ax.plot([x1, x1, x2, x2],
                [y, y + bracket_height, y + bracket_height, y],
                lw=1.0, color="#424242")
        ax.text((x1 + x2) / 2, y + bracket_height, label,
                ha="center", va="bottom", fontsize=8, color="#424242")
 
    # Extend y-axis to fit brackets
    n_sig = len(sig_pairs)
    ax.set_ylim(
        df_long["JSD"].min() - y_range * 0.05,
        y_max + bracket_step * (n_sig + 1.5),
    )
    ax.set_xlim(-0.5, len(split_names) - 0.5)
 
    # Legend for split type
    real_patch = mpatches.Patch(color="#1976d2", label="Real splits")
    base_patch = mpatches.Patch(color="#9e9e9e", label="Baselines")
    ax.legend(handles=[real_patch, base_patch], fontsize=9, framealpha=0.8)
 
    ax.set_title("Jensen-Shannon Distance by Split\n(Mann-Whitney U, BH-corrected)",
                 fontsize=12, color="#212121", pad=8)
    ax.set_xlabel("Split", fontsize=10, color="#555555")
    ax.set_ylabel("Jensen-Shannon Distance", fontsize=10, color="#555555")
    ax.tick_params(colors="#555555", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    ax.grid(axis="y", color="#eeeeee", lw=0.7, zorder=0)
 
    plt.tight_layout()
    plt.savefig(f"{results_dir}/jsd_boxplot_sig.png", dpi=150)
    plt.show()
 
    return df_stats
 