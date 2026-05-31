"""
Fine-grid (step=0.01) ensemble-weight search for 3 models (XGB + LSTM + CNN),
for both `baseline` and `youtube` feature variants.

The 3-simplex {w_xgb + w_lstm + w_cnn = 1, w ∈ {0, 0.01, ..., 1.00}} has
C(102, 2) = 5,151 lattice points. For each point we evaluate:

    AUC                              (threshold-free)
    Accuracy, Precision, Recall, F1   at THRESHOLD = 0.50

For each variant we save:
    ensemble_output/optimal_sweep_{variant}.parquet   long-format sweep
    ensemble_output/optimal_best_{variant}.csv        argmax-per-metric weights
    ensemble_output/plots/optimal_3d_{variant}.png    5-panel 3D surface
    ensemble_output/plots/optimal_3d_{variant}_{metric}.png   per-metric 3D

Source loaders / merge are imported from ensemble_3model.py so behaviour stays
identical to the headline ensemble.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection
from scipy.stats import rankdata

from ensemble_3model import VARIANTS, load_cnn, load_lstm, load_xgb


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "ensemble"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

STEP = 0.01            # 0.01 grid on the simplex
THRESHOLD = 0.50       # for ACC / Prec / Rec / F1
METRICS = ["AUC", "Accuracy", "Precision", "Recall", "F1"]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Data prep — same join logic as ensemble_3model.run_variant                  #
# --------------------------------------------------------------------------- #
def build_merged(variant: str) -> pd.DataFrame:
    v = VARIANTS[variant]
    xgb = load_xgb(v).rename(columns={"label_binary": "lbl_xgb", "test_year": "ty_xgb"})
    lstm = load_lstm(v).rename(columns={"label_binary": "lbl_lstm", "test_year": "ty_lstm"})
    cnn = load_cnn(v).rename(columns={"label_binary": "lbl_cnn", "test_year": "ty_cnn"})

    m = xgb.merge(lstm, on=["date", "ticker"], how="inner")
    m = m.merge(cnn, on=["date", "ticker"], how="inner")
    m["label_binary"] = (
        m["lbl_xgb"].combine_first(m["lbl_lstm"]).combine_first(m["lbl_cnn"])
    )
    m["test_year"] = m["ty_xgb"]
    return m[["date", "ticker", "test_year", "label_binary",
              "p_xgb", "p_lstm", "p_cnn"]]


# --------------------------------------------------------------------------- #
# Fast AUC via Mann–Whitney U on precomputed score array                      #
# --------------------------------------------------------------------------- #
def fast_auc(p: np.ndarray, y: np.ndarray, n_pos: int, n_neg: int) -> float:
    r = rankdata(p, method="average")
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_metrics(p: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float, float, float]:
    pred = p >= thr
    tp = int(np.logical_and(pred, y == 1).sum())
    fp = int(np.logical_and(pred, y == 0).sum())
    fn = int(np.logical_and(~pred, y == 1).sum())
    tn = int(np.logical_and(~pred, y == 0).sum())
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #
def sweep(merged: pd.DataFrame, step: float = STEP, thr: float = THRESHOLD) -> pd.DataFrame:
    px = merged["p_xgb"].to_numpy(dtype=np.float64)
    pl = merged["p_lstm"].to_numpy(dtype=np.float64)
    pc = merged["p_cnn"].to_numpy(dtype=np.float64)
    y = merged["label_binary"].astype(int).to_numpy()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"  n={len(y):,}  pos={n_pos:,}  neg={n_neg:,}  base_rate={n_pos/len(y):.4f}")

    grid = int(round(1.0 / step))   # 100 → ints 0..100
    combos = (grid + 1) * (grid + 2) // 2  # 5151 for step=0.01

    rows = np.empty((combos, 8), dtype=np.float64)
    i = 0
    t0 = time.time()
    next_log = 500
    for a in range(grid + 1):           # w_xgb steps
        wx = a / grid
        for b in range(grid + 1 - a):   # w_lstm steps
            wl = b / grid
            wc = 1.0 - wx - wl
            p = wx * px + wl * pl + wc * pc
            auc = fast_auc(p, y, n_pos, n_neg)
            acc, prec, rec, f1 = threshold_metrics(p, y, thr)
            rows[i] = (wx, wl, wc, auc, acc, prec, rec, f1)
            i += 1
            if i >= next_log:
                elapsed = time.time() - t0
                eta = elapsed * (combos - i) / max(i, 1)
                print(f"    {i:>5}/{combos}  elapsed={elapsed:5.1f}s  eta={eta:5.1f}s")
                next_log += 500
    print(f"  swept {i} combos in {time.time() - t0:.1f}s")

    return pd.DataFrame(
        rows,
        columns=["w_xgb", "w_lstm", "w_cnn", "AUC", "Accuracy", "Precision", "Recall", "F1"],
    )


def best_per_metric(sweep_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in METRICS:
        idx = sweep_df[m].idxmax()
        row = sweep_df.loc[idx].copy()
        out.append({
            "metric": m,
            "best_value": float(row[m]),
            "w_xgb": float(row["w_xgb"]),
            "w_lstm": float(row["w_lstm"]),
            "w_cnn": float(row["w_cnn"]),
            "AUC@best": float(row["AUC"]),
            "Accuracy@best": float(row["Accuracy"]),
            "Precision@best": float(row["Precision"]),
            "Recall@best": float(row["Recall"]),
            "F1@best": float(row["F1"]),
        })
    return pd.DataFrame(out).set_index("metric")


# --------------------------------------------------------------------------- #
# 3D plots                                                                    #
# --------------------------------------------------------------------------- #
def _plot_one(ax, sweep_df: pd.DataFrame, metric: str, title: str) -> None:
    x = sweep_df["w_xgb"].values
    y = sweep_df["w_lstm"].values
    z = sweep_df[metric].values
    tri = ax.plot_trisurf(
        x, y, z,
        cmap=cm.viridis, edgecolor="none", linewidth=0, antialiased=True, alpha=0.95,
    )
    # mark argmax
    idx = int(np.argmax(z))
    ax.scatter([x[idx]], [y[idx]], [z[idx]], color="red", s=60, marker="*",
               edgecolor="black", linewidth=0.6, zorder=10,
               label=f"max={z[idx]:.4f}\n(w_xgb={x[idx]:.2f}, w_lstm={y[idx]:.2f}, w_cnn={1-x[idx]-y[idx]:.2f})")
    ax.set_xlabel("w_xgb")
    ax.set_ylabel("w_lstm")
    ax.set_zlabel(metric)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax.view_init(elev=24, azim=-58)
    return tri


def plot_panel(sweep_df: pd.DataFrame, variant: str) -> None:
    fig = plt.figure(figsize=(22, 11))
    fig.suptitle(
        f"3-model ensemble — weight sweep (step={STEP}, thr={THRESHOLD:.2f}) — {variant}",
        fontsize=14, fontweight="bold",
    )
    for i, m in enumerate(METRICS):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        _plot_one(ax, sweep_df, m, m)
    # legend / explainer in last cell
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    note = (
        "Simplex constraint:  w_xgb + w_lstm + w_cnn = 1\n"
        f"Step:                 {STEP}\n"
        f"Threshold (acc/prec/rec/f1): {THRESHOLD:.2f}\n"
        f"Grid points:          {len(sweep_df):,}\n\n"
        "★ = optimum for the panel's metric.\n"
        "z-axis and color both encode the metric value.\n"
        "(w_cnn implied: 1 − w_xgb − w_lstm)"
    )
    ax6.text(0.02, 0.98, note, ha="left", va="top", fontsize=11,
             family="monospace",
             bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f5f5f5", "edgecolor": "#999"})
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOT_DIR / f"optimal_3d_{variant}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  → {out}")


def plot_each(sweep_df: pd.DataFrame, variant: str) -> None:
    for m in METRICS:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        _plot_one(ax, sweep_df, m, f"{variant} — {m}  (thr={THRESHOLD:.2f}, step={STEP})")
        fig.tight_layout()
        out = PLOT_DIR / f"optimal_3d_{variant}_{m}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  → {out}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run(variant: str) -> None:
    print(f"\n{'=' * 76}\nVariant: {variant}\n{'=' * 76}")
    merged = build_merged(variant)
    print(f"  merged rows={len(merged):,}  years={sorted(merged['test_year'].unique())}")

    df = sweep(merged)

    sweep_path = OUT_DIR / f"optimal_sweep_{variant}.parquet"
    df.to_parquet(sweep_path, index=False)
    print(f"  → {sweep_path}  ({len(df):,} rows)")

    best = best_per_metric(df)
    best_path = OUT_DIR / f"optimal_best_{variant}.csv"
    best.round(6).to_csv(best_path)
    print(f"  → {best_path}")
    print(f"\n  Optimal weights per metric ({variant}):")
    print(best[["best_value", "w_xgb", "w_lstm", "w_cnn"]].round(4).to_string())

    print("\n  Rendering 3D plots …")
    plot_panel(df, variant)
    plot_each(df, variant)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "youtube", "both"], default="both")
    args = ap.parse_args()
    targets = ["baseline", "youtube"] if args.variant == "both" else [args.variant]
    for v in targets:
        run(v)


if __name__ == "__main__":
    main()
