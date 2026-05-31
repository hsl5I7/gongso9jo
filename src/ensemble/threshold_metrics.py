"""
Confusion matrix + Acc / Prec / Rec / F1 at threshold = 0.50 for the saved
*uncalibrated* 3-model ensembles (weights 0.78 / 0.07 / 0.15).

Inputs:
    outputs/ensemble/baseline/predictions_test_pattern_ty{YYYY}.csv
    outputs/ensemble/youtube/predictions_test_pattern_yt12_ty{YYYY}.csv
        cols: date, ticker, prob_익절, label_binary

Outputs (outputs/ensemble/threshold_metrics/):
    metrics_table.csv                long: variant × {per fold, pooled} × counts + 4 metrics
    confusion_pooled.png             1×2 heatmap (baseline | youtube), pooled
    confusion_per_fold_baseline.png  2×4 grid of per-year heatmaps
    confusion_per_fold_youtube.png   2×4 grid of per-year heatmaps
    metrics_bars.png                 per-fold grouped bars, one panel per metric
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ENS_ROOT = ROOT / "outputs" / "ensemble"
OUT_DIR = ENS_ROOT / "threshold_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_DIRS = {
    "baseline": ENS_ROOT / "baseline",
    "youtube": ENS_ROOT / "youtube",
}
WEIGHTS = (0.78, 0.07, 0.15)
THRESHOLD = 0.50
METRICS = ["Accuracy", "Precision", "Recall", "F1"]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Load                                                                        #
# --------------------------------------------------------------------------- #
def _year_from_stem(stem: str) -> int:
    return int(stem.rsplit("ty", 1)[-1].split("_", 1)[0])


def load_variant(variant_dir: Path) -> dict[int, pd.DataFrame]:
    files = sorted(variant_dir.glob("predictions_test_pattern_*.csv"))
    assert files, f"no CSVs under {variant_dir}"
    out: dict[int, pd.DataFrame] = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lstrip("﻿") for c in df.columns]
        df["label_binary"] = df["label_binary"].astype(int)
        df["prob_익절"] = df["prob_익절"].astype(float)
        out[_year_from_stem(f.stem)] = df[["prob_익절", "label_binary"]]
    return out


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def metrics_from_cm(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": n, "n_pos": tp + fn, "base_rate": (tp + fn) / n if n else 0.0,
            "n_pred_pos": tp + fp, "pred_pos_rate": (tp + fp) / n if n else 0.0,
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


def evaluate_variant(folds: dict[int, pd.DataFrame], thr: float) -> pd.DataFrame:
    rows = []
    # per-fold
    for yr, df in sorted(folds.items()):
        y = df["label_binary"].to_numpy()
        pred = (df["prob_익절"].to_numpy() >= thr).astype(int)
        cm = confusion(y, pred)
        m = metrics_from_cm(cm)
        rows.append({"fold": yr, **cm, **m})
    # pooled
    full = pd.concat(folds.values(), ignore_index=True)
    y = full["label_binary"].to_numpy()
    pred = (full["prob_익절"].to_numpy() >= thr).astype(int)
    cm = confusion(y, pred)
    m = metrics_from_cm(cm)
    rows.append({"fold": "pooled", **cm, **m})
    return pd.DataFrame(rows).set_index("fold")


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _draw_cm(ax, cm: dict[str, int], title: str, base_rate: float | None = None) -> None:
    """2×2 confusion-matrix heatmap. rows = actual, cols = predicted."""
    mat = np.array([[cm["TN"], cm["FP"]],
                    [cm["FN"], cm["TP"]]])
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    n = mat.sum()
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            ax.text(j, i, f"{v:,}\n({v / n * 100:.1f}%)" if n else "",
                    ha="center", va="center",
                    color="white" if v > mat.max() / 2 else "#222",
                    fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title(title, fontsize=10)
    if base_rate is not None:
        ax.set_xlabel(f"base_rate={base_rate:.3f}", fontsize=8)
    return im


def plot_pooled_confusion(per_variant: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (variant, df) in zip(axes, per_variant.items()):
        row = df.loc["pooled"]
        cm = {k: int(row[k]) for k in ("TP", "FP", "FN", "TN")}
        _draw_cm(ax, cm, f"{variant} — pooled (Acc={row['Accuracy']:.3f}, "
                          f"Prec={row['Precision']:.3f}, Rec={row['Recall']:.3f}, "
                          f"F1={row['F1']:.3f})",
                 base_rate=row["base_rate"])
    fig.suptitle(
        f"Confusion matrix — uncalibrated ensemble  "
        f"(weights {WEIGHTS[0]}/{WEIGHTS[1]}/{WEIGHTS[2]}, threshold={THRESHOLD:.2f})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT_DIR / "confusion_pooled.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


def plot_per_fold_confusion(variant: str, df: pd.DataFrame) -> None:
    rows = [r for r in df.index if isinstance(r, (int, np.integer))]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, yr in zip(axes.flat, rows):
        row = df.loc[yr]
        cm = {k: int(row[k]) for k in ("TP", "FP", "FN", "TN")}
        _draw_cm(ax, cm, f"ty{yr}  (F1={row['F1']:.3f})",
                 base_rate=row["base_rate"])
    fig.suptitle(
        f"Confusion matrix per fold — {variant}  "
        f"(uncalibrated, threshold={THRESHOLD:.2f})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / f"confusion_per_fold_{variant}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


def plot_metric_bars(per_variant: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = {"baseline": "#1f77b4", "youtube": "#d62728"}
    years = sorted(int(y) for y in next(iter(per_variant.values())).index
                   if isinstance(y, (int, np.integer)))
    x = np.arange(len(years)); w = 0.38

    for ax, m in zip(axes.flat, METRICS):
        base_vals = [per_variant["baseline"].loc[y, m] for y in years]
        yt_vals = [per_variant["youtube"].loc[y, m] for y in years]
        b1 = ax.bar(x - w / 2, base_vals, w, label="baseline", color=colors["baseline"])
        b2 = ax.bar(x + w / 2, yt_vals, w, label="youtube", color=colors["youtube"])
        for bars in (b1, b2):
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.003,
                        f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(per_variant["baseline"].loc["pooled", m], color=colors["baseline"],
                   ls=":", lw=1.1, alpha=0.65,
                   label=f"base pooled={per_variant['baseline'].loc['pooled', m]:.3f}")
        ax.axhline(per_variant["youtube"].loc["pooled", m], color=colors["youtube"],
                   ls=":", lw=1.1, alpha=0.65,
                   label=f"yt pooled={per_variant['youtube'].loc['pooled', m]:.3f}")
        ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years])
        ax.set_title(m, fontsize=11)
        ax.set_ylabel(m)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="best", fontsize=7, framealpha=0.9)

    fig.suptitle(
        f"Per-fold metrics — uncalibrated ensemble  "
        f"(weights {WEIGHTS[0]}/{WEIGHTS[1]}/{WEIGHTS[2]}, threshold={THRESHOLD:.2f})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "metrics_bars.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"Loading ensembles from {ENS_ROOT}")
    data = {name: load_variant(path) for name, path in VARIANT_DIRS.items()}
    for name, folds in data.items():
        n = sum(len(df) for df in folds.values())
        print(f"  {name:<8}  years={sorted(folds.keys())}  total_rows={n:,}")

    print(f"\nEvaluating at threshold={THRESHOLD:.2f}")
    per_variant = {name: evaluate_variant(folds, THRESHOLD)
                   for name, folds in data.items()}

    # --- CSV: long format ---
    parts = []
    for name, df in per_variant.items():
        d = df.reset_index().assign(variant=name, threshold=THRESHOLD)
        parts.append(d)
    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df[["variant", "threshold", "fold", "n", "n_pos", "base_rate",
                      "n_pred_pos", "pred_pos_rate",
                      "TP", "FP", "FN", "TN",
                      "Accuracy", "Precision", "Recall", "F1"]]
    csv_out = OUT_DIR / "metrics_table.csv"
    out_df.round(6).to_csv(csv_out, index=False)
    print(f"  → {csv_out}")

    # --- Console print ---
    for name, df in per_variant.items():
        print(f"\n[{name}]  threshold={THRESHOLD:.2f}")
        show = df[["n", "base_rate", "TP", "FP", "FN", "TN",
                   "Accuracy", "Precision", "Recall", "F1"]].copy()
        show[["base_rate", "Accuracy", "Precision", "Recall", "F1"]] = (
            show[["base_rate", "Accuracy", "Precision", "Recall", "F1"]].round(4)
        )
        print(show.to_string())

    # --- Plots ---
    print("\nRendering plots …")
    plot_pooled_confusion(per_variant)
    for name, df in per_variant.items():
        plot_per_fold_confusion(name, df)
    plot_metric_bars(per_variant)


if __name__ == "__main__":
    main()
