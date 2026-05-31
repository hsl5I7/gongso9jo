"""
AUROC analysis for the two saved 3-model ensembles at weights (0.78, 0.07, 0.15).

Inputs (already weight-applied):
    outputs/ensemble/baseline/predictions_test_pattern_ty{YYYY}.csv
    outputs/ensemble/youtube/predictions_test_pattern_yt12_ty{YYYY}.csv
    cols: date, ticker, prob_익절, label_binary

Outputs (outputs/ensemble/auroc/):
    pooled_roc.png                 baseline vs youtube ensemble overlaid
    per_fold_roc_baseline.png      8-year ROC curves for baseline
    per_fold_roc_youtube.png       8-year ROC curves for youtube
    per_fold_auroc.csv             year × variant AUC matrix (+ pooled row)
    per_fold_auroc_bars.png        grouped bar chart per year
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[2]
ENS_ROOT = ROOT / "outputs" / "ensemble"
OUT_DIR = ENS_ROOT / "auroc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "baseline": ENS_ROOT / "baseline",
    "youtube": ENS_ROOT / "youtube",
}

WEIGHTS = (0.78, 0.07, 0.15)   # for plot subtitles

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Load                                                                        #
# --------------------------------------------------------------------------- #
def _year_from_stem(stem: str) -> int:
    # works for both `predictions_test_pattern_ty2018` and
    # `predictions_test_pattern_yt12_ty2018`
    return int(stem.rsplit("ty", 1)[-1].split("_", 1)[0])


def load_variant(variant_dir: Path) -> dict[int, pd.DataFrame]:
    files = sorted(variant_dir.glob("predictions_test_pattern_*.csv"))
    assert files, f"no CSVs under {variant_dir}"
    out: dict[int, pd.DataFrame] = {}
    for f in files:
        df = pd.read_csv(f)
        # tolerate BOM
        df.columns = [c.lstrip("﻿") for c in df.columns]
        df["label_binary"] = df["label_binary"].astype(int)
        df["prob_익절"] = df["prob_익절"].astype(float)
        out[_year_from_stem(f.stem)] = df[["prob_익절", "label_binary"]]
    return out


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def per_fold_auc(folds: dict[int, pd.DataFrame]) -> pd.Series:
    return pd.Series(
        {
            yr: (
                roc_auc_score(df["label_binary"], df["prob_익절"])
                if df["label_binary"].nunique() == 2
                else np.nan
            )
            for yr, df in sorted(folds.items())
        },
        name="AUC",
    )


def pooled_auc(folds: dict[int, pd.DataFrame]) -> float:
    all_df = pd.concat(folds.values(), ignore_index=True)
    return float(roc_auc_score(all_df["label_binary"], all_df["prob_익절"]))


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _suptitle(fig, title: str) -> None:
    fig.suptitle(
        f"{title}    (weights: w_xgb={WEIGHTS[0]}, w_lstm={WEIGHTS[1]}, w_cnn={WEIGHTS[2]})",
        fontsize=12, fontweight="bold",
    )


def plot_pooled_roc(data: dict[str, dict[int, pd.DataFrame]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    colors = {"baseline": "#1f77b4", "youtube": "#d62728"}
    for variant, folds in data.items():
        df = pd.concat(folds.values(), ignore_index=True)
        fpr, tpr, _ = roc_curve(df["label_binary"], df["prob_익절"])
        auc = roc_auc_score(df["label_binary"], df["prob_익절"])
        ax.plot(fpr, tpr, color=colors[variant], lw=2.2,
                label=f"{variant}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled ROC — both ensembles", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    _suptitle(fig, "AUROC analysis — 3-model ensemble")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "pooled_roc.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


def plot_per_fold_roc(variant: str, folds: dict[int, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    cmap = plt.get_cmap("viridis")
    years = sorted(folds.keys())
    for i, yr in enumerate(years):
        df = folds[yr]
        if df["label_binary"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df["label_binary"], df["prob_익절"])
        auc = roc_auc_score(df["label_binary"], df["prob_익절"])
        ax.plot(fpr, tpr, color=cmap(i / max(len(years) - 1, 1)), lw=1.6,
                label=f"ty{yr}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Per-fold ROC — {variant}", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _suptitle(fig, f"AUROC analysis — 3-model ensemble ({variant})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / f"per_fold_roc_{variant}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


def plot_per_fold_bars(per_fold: pd.DataFrame) -> None:
    """Grouped bar chart: each year has a baseline + youtube bar; pooled row shown as separate group."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    years = [str(y) for y in per_fold.index if isinstance(y, (int, np.integer))]
    x = np.arange(len(years))
    width = 0.38
    base_vals = [per_fold.loc[int(y), "baseline"] for y in years]
    yt_vals = [per_fold.loc[int(y), "youtube"] for y in years]
    b1 = ax.bar(x - width / 2, base_vals, width, label="baseline", color="#1f77b4")
    b2 = ax.bar(x + width / 2, yt_vals, width, label="youtube", color="#d62728")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.003,
                    f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    # pooled reference lines
    if "pooled" in per_fold.index:
        ax.axhline(per_fold.loc["pooled", "baseline"], color="#1f77b4", ls=":",
                   lw=1.2, alpha=0.7,
                   label=f"baseline pooled = {per_fold.loc['pooled', 'baseline']:.4f}")
        ax.axhline(per_fold.loc["pooled", "youtube"], color="#d62728", ls=":",
                   lw=1.2, alpha=0.7,
                   label=f"youtube  pooled = {per_fold.loc['pooled', 'youtube']:.4f}")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="chance (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_xlabel("Test year (fold)")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-fold AUROC — baseline vs youtube ensemble", fontsize=11)
    ymin = min(min(base_vals), min(yt_vals)) - 0.02
    ax.set_ylim(max(ymin, 0.40), max(max(base_vals), max(yt_vals)) + 0.04)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    _suptitle(fig, "AUROC analysis — per-fold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "per_fold_auroc_bars.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"Loading ensembles from {ENS_ROOT}")
    data = {name: load_variant(path) for name, path in VARIANTS.items()}
    for name, folds in data.items():
        n = sum(len(df) for df in folds.values())
        print(f"  {name:<8}  years={sorted(folds.keys())}  total_rows={n:,}")

    # --- AUC table (per-fold + pooled) ---
    table = pd.DataFrame({name: per_fold_auc(folds) for name, folds in data.items()})
    table.index.name = "test_year"
    pooled = pd.Series({name: pooled_auc(folds) for name, folds in data.items()}, name="pooled")
    table = pd.concat([table, pooled.to_frame().T])
    table.loc["mean_per_fold"] = table.loc[[i for i in table.index if isinstance(i, (int, np.integer))]].mean()

    print("\n[Per-fold AUROC]")
    print(table.round(4).to_string())
    csv_out = OUT_DIR / "per_fold_auroc.csv"
    table.round(6).to_csv(csv_out)
    print(f"\n  → {csv_out}")

    # --- Plots ---
    print("\nRendering plots …")
    plot_pooled_roc(data)
    for name, folds in data.items():
        plot_per_fold_roc(name, folds)
    plot_per_fold_bars(table)


if __name__ == "__main__":
    main()
