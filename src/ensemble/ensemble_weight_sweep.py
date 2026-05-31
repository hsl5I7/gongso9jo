"""
3-model ensemble weight sweep + visualization.

Sweeps the 3-simplex (w_xgb + w_lstm + w_cnn = 1) with step 0.1
(=> 66 weight combinations) and, for each fold (2018..2025) and threshold
in {0.50, 0.60, 0.65}, computes:

  AUC (threshold-independent)
  ACC, Precision, Recall, F1   (binary @ threshold)

Note on metrics:
  An ensemble of pre-computed predictions has no training phase; the only
  meaningful split is the per-fold *test* set. There is no train/val accuracy
  for the ensemble itself. (Underlying XGB/LSTM/CNN are already trained
  upstream — see their own history.csv files for train/val curves.)

Inputs (from ensemble_xgb_lstm.py): the same 3-way merged parquet.

Outputs (ensemble_output/):
  weight_sweep_metrics.csv          long-format: fold × weight × threshold × metric
  best_weights_per_fold.csv         best weight per (fold, metric, threshold)
  plots/fold_{YYYY}_simplex.png     5×3 grid of ternary heatmaps per fold
  plots/all_folds_auc.png           AUC simplex grid (one panel per fold)
  plots/best_weight_summary.png     per-fold best weights as bar charts
  plots/metrics_across_folds.png    line plot: representative weight combos × folds
  plots/mean_across_folds_simplex.png   mean (over folds) of each metric — simplex
  plots/metric_vs_weight_marginal.png   marginal effect of each weight axis on metrics
                                        (5 metrics × 3 axes, per-fold lines + bold mean)
  plots/metric_3d_mean.png          3D scatter: axes = (w_xgb, w_lstm, w_cnn),
                                    color = mean metric value over folds (5 panels)
  plots/auc_3d_per_fold.png         3D scatter per fold for AUC (8 panels)

Note on train_acc / val_acc:
  An ensemble of pre-computed test-set predictions has no training step, so
  "train_acc" and "val_acc" are not defined for the ensemble itself. The only
  evaluation surface is the per-fold *test* prediction. Train/val accuracy of
  the underlying base models is upstream (LSTM history.csv has train_loss /
  val_auc per epoch but no train_acc/val_acc; XGB/CNN don't save those at all).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
XGB_PATH = ROOT / "to_team_ensemble" / "predictions_all.parquet"
LSTM_BASE = ROOT / "lstm" / "models" / "lstm" / "general_v2" / "lookback_30"
CNN_DIR = ROOT / "outputs" / "cnn" / "baseline"
OUT_DIR = ROOT / "outputs" / "ensemble"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

STEP = 0.1
THRESHOLDS = (0.50, 0.60, 0.65)


# --------------------------------------------------------------------------- #
# Loaders (mirror ensemble_xgb_lstm.py)                                       #
# --------------------------------------------------------------------------- #
def load_xgb() -> pd.DataFrame:
    df = pd.read_parquet(XGB_PATH).rename(columns={"p_pos": "p_xgb"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    return df[["date", "ticker", "p_xgb", "label_binary", "test_year"]]


def load_lstm() -> pd.DataFrame:
    parts = []
    for f in sorted(LSTM_BASE.glob("fold_*")):
        p = f / "test_predictions.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        prob = [c for c in d.columns if c.startswith("prob_")][0]
        d = d.rename(columns={prob: "p_lstm"})
        d["test_year"] = int(f.name.split("test")[-1])
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out[["date", "ticker", "p_lstm", "label_binary", "test_year"]]


def load_cnn() -> pd.DataFrame:
    parts = []
    for f in sorted(CNN_DIR.glob("predictions_test_pattern_ty*.csv")):
        d = pd.read_csv(f)
        prob = [c for c in d.columns if c.startswith("prob_")][0]
        d = d.rename(columns={prob: "p_cnn"})
        d["test_year"] = int(f.stem.split("ty")[-1])
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out[["date", "ticker", "p_cnn", "label_binary", "test_year"]]


def build_merged() -> pd.DataFrame:
    xgb = load_xgb().rename(columns={"label_binary": "lbl", "test_year": "ty"})
    lstm = load_lstm().drop(columns=["label_binary", "test_year"])
    cnn = load_cnn().drop(columns=["label_binary", "test_year"])
    m = xgb.merge(lstm, on=["date", "ticker"], how="inner")
    m = m.merge(cnn, on=["date", "ticker"], how="inner")
    m = m.rename(columns={"lbl": "label_binary", "ty": "test_year"})
    m = m[m["label_binary"].notna()].copy()
    m["label_binary"] = m["label_binary"].astype(int)
    return m


# --------------------------------------------------------------------------- #
# Weight grid (3-simplex, step 0.1)                                           #
# --------------------------------------------------------------------------- #
def simplex_weights(step: float = STEP) -> list[tuple[float, float, float]]:
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((i * step, j * step, k * step))
    return out


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict:
    """Returns a flat dict: AUC, plus ACC/Prec/Rec/F1 at each threshold."""
    out: dict = {}
    out["AUC"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    for thr in thresholds:
        yhat = (p >= thr).astype(int)
        if yhat.sum() == 0:
            prec = 0.0
        else:
            prec = precision_score(y, yhat, zero_division=0)
        rec = recall_score(y, yhat, zero_division=0)
        acc = accuracy_score(y, yhat)
        f1 = f1_score(y, yhat, zero_division=0)
        out[f"ACC@{thr:.2f}"] = acc
        out[f"Prec@{thr:.2f}"] = prec
        out[f"Rec@{thr:.2f}"] = rec
        out[f"F1@{thr:.2f}"] = f1
    return out


def sweep(merged: pd.DataFrame) -> pd.DataFrame:
    weights = simplex_weights(STEP)
    fold_years = sorted(merged["test_year"].unique())
    print(f"sweep: {len(weights)} weight combos × {len(fold_years)} folds = "
          f"{len(weights) * len(fold_years)} evaluations")

    rows = []
    for ty in fold_years:
        g = merged[merged["test_year"] == ty]
        y = g["label_binary"].to_numpy()
        p_xgb = g["p_xgb"].to_numpy()
        p_lstm = g["p_lstm"].to_numpy()
        p_cnn = g["p_cnn"].to_numpy()
        for wx, wl, wc in weights:
            p = wx * p_xgb + wl * p_lstm + wc * p_cnn
            m = compute_metrics(y, p)
            m.update({
                "test_year": int(ty),
                "n": len(g),
                "w_xgb": round(wx, 2),
                "w_lstm": round(wl, 2),
                "w_cnn": round(wc, 2),
            })
            rows.append(m)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ternary (simplex) heatmap                                                   #
# --------------------------------------------------------------------------- #
def bary_to_xy(w_xgb: np.ndarray, w_lstm: np.ndarray, w_cnn: np.ndarray):
    """Map (w_xgb, w_lstm, w_cnn) → 2D triangle. Corners:
       XGB=(0,0), CNN=(1,0), LSTM=(0.5, sqrt(3)/2).
    """
    x = w_cnn + 0.5 * w_lstm
    y = (np.sqrt(3.0) / 2.0) * w_lstm
    return x, y


def draw_simplex(
    ax: plt.Axes,
    weights: pd.DataFrame,
    values: np.ndarray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> "plt.cm.ScalarMappable":
    xs, ys = bary_to_xy(
        weights["w_xgb"].to_numpy(),
        weights["w_lstm"].to_numpy(),
        weights["w_cnn"].to_numpy(),
    )
    tri = Triangulation(xs, ys)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1e-6
    tcf = ax.tricontourf(tri, values, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.tricontour(tri, values, levels=10, colors="white", linewidths=0.3, alpha=0.4)

    # corner labels
    ax.text(0.0, -0.04, "XGB", ha="center", va="top", fontsize=8, weight="bold")
    ax.text(1.0, -0.04, "CNN", ha="center", va="top", fontsize=8, weight="bold")
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, "LSTM", ha="center", va="bottom",
            fontsize=8, weight="bold")

    # mark best point
    best = int(np.nanargmax(values))
    ax.plot(xs[best], ys[best], marker="*", color="red", markersize=9,
            markeredgecolor="white", markeredgewidth=0.7)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.10)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return tcf


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def plot_fold(df_fold: pd.DataFrame, test_year: int) -> Path:
    """5×3 grid: row 1 = AUC (shown once), rows 2..5 = ACC/Prec/Rec/F1 × 3 thresholds."""
    fig = plt.figure(figsize=(12, 17))
    gs = fig.add_gridspec(5, 3, hspace=0.30, wspace=0.10)

    # Row 1, col 0: AUC. Cols 1,2 blank (with note).
    ax_auc = fig.add_subplot(gs[0, 0])
    tcf = draw_simplex(ax_auc, df_fold, df_fold["AUC"].to_numpy(),
                       title=f"AUC  (max={df_fold['AUC'].max():.4f})")
    cb = fig.colorbar(tcf, ax=ax_auc, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    for c in (1, 2):
        ax = fig.add_subplot(gs[0, c])
        ax.text(0.5, 0.5, "(threshold-independent)\n→ see AUC at left",
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_axis_off()

    metrics = ["ACC", "Prec", "Rec", "F1"]
    for r, met in enumerate(metrics, start=1):
        for c, thr in enumerate(THRESHOLDS):
            col = f"{met}@{thr:.2f}"
            ax = fig.add_subplot(gs[r, c])
            vals = df_fold[col].to_numpy()
            tcf = draw_simplex(ax, df_fold, vals,
                               title=f"{met} @ thr={thr:.2f}  (max={vals.max():.4f})")
            cb = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Fold test_year={test_year}  (n={int(df_fold['n'].iloc[0]):,})  — "
        f"weight simplex sweep (step={STEP})",
        fontsize=12, y=0.995,
    )
    out_path = PLOT_DIR / f"fold_{test_year}_simplex.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_auc_grid(df: pd.DataFrame) -> Path:
    """All folds' AUC on one figure for at-a-glance comparison."""
    folds = sorted(df["test_year"].unique())
    n = len(folds)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.8 * rows))
    axes = np.atleast_2d(axes)

    # shared color scale
    vmin = df["AUC"].min()
    vmax = df["AUC"].max()

    last_tcf = None
    for k, ty in enumerate(folds):
        ax = axes[k // cols, k % cols]
        sub = df[df["test_year"] == ty]
        last_tcf = draw_simplex(
            ax, sub, sub["AUC"].to_numpy(),
            title=f"{ty}  (max={sub['AUC'].max():.4f})",
            vmin=vmin, vmax=vmax,
        )
    # hide extras
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].set_axis_off()

    if last_tcf is not None:
        fig.subplots_adjust(right=0.90)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(last_tcf, cax=cax, label="AUC")
    fig.suptitle("AUC over weight simplex — all folds (shared color scale)",
                 fontsize=13, y=0.995)
    out = PLOT_DIR / "all_folds_auc.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metrics_across_folds(df: pd.DataFrame) -> Path:
    """For a handful of representative weight combos, plot how each metric
    varies across the 8 folds. Easier to read than a simplex when comparing
    fold-to-fold stability of a chosen ratio.
    """
    # representative weight combos to compare
    combos = [
        ("XGB only",        (1.0, 0.0, 0.0)),
        ("LSTM only",       (0.0, 1.0, 0.0)),
        ("CNN only",        (0.0, 0.0, 1.0)),
        ("equal (1/3 each)",(0.3, 0.3, 0.4)),  # closest grid point to (1/3,1/3,1/3)
        ("XGB heavy 0.5/0.3/0.2", (0.5, 0.3, 0.2)),
        ("XGB heavy 0.6/0.2/0.2", (0.6, 0.2, 0.2)),
        ("LSTM heavy 0.2/0.6/0.2", (0.2, 0.6, 0.2)),
        ("CNN heavy 0.2/0.2/0.6",  (0.2, 0.2, 0.6)),
    ]
    metrics = [
        ("AUC",       "AUC"),
        ("Accuracy",  "ACC@0.50"),
        ("Precision", "Prec@0.50"),
        ("Recall",    "Rec@0.50"),
        ("F1-score",  "F1@0.50"),
    ]

    folds = sorted(df["test_year"].unique())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3.0 * len(metrics)),
                             sharex=True)
    for ax, (label, col) in zip(axes, metrics):
        for name, (wx, wl, wc) in combos:
            sub = df[
                (np.isclose(df["w_xgb"], wx))
                & (np.isclose(df["w_lstm"], wl))
                & (np.isclose(df["w_cnn"], wc))
            ].sort_values("test_year")
            if sub.empty:
                continue
            ax.plot(sub["test_year"].astype(int), sub[col], marker="o",
                    label=name, linewidth=1.4)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_title(f"{label} @ test set, threshold=0.50 (n/a for AUC)",
                     fontsize=9, loc="left")

    axes[-1].set_xlabel("test_year (fold)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.45),
                   ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Per-fold test metrics across representative weight combos",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = PLOT_DIR / "metrics_across_folds.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mean_simplex(df: pd.DataFrame) -> Path:
    """One figure, 5 simplex panels: mean of each metric across folds.
    Helps spot the weight combo that generalizes best.
    """
    cols = ["AUC", "ACC@0.50", "Prec@0.50", "Rec@0.50", "F1@0.50"]
    titles = ["AUC", "Accuracy@0.50", "Precision@0.50", "Recall@0.50", "F1@0.50"]

    mean = (
        df.groupby(["w_xgb", "w_lstm", "w_cnn"], as_index=False)[cols].mean()
    )

    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4.2))
    for ax, col, title in zip(axes, cols, titles):
        vals = mean[col].to_numpy()
        tcf = draw_simplex(
            ax, mean, vals,
            title=f"{title}\nmean over folds (max={vals.max():.4f})",
        )
        cb = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle("Mean across 8 folds — weight simplex (★ = best mean)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = PLOT_DIR / "mean_across_folds_simplex.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metric_vs_weight_marginal(df: pd.DataFrame) -> Path:
    """5 metrics × 3 weight axes grid.

    For each weight axis A ∈ {w_xgb, w_lstm, w_cnn} and each fixed value
    k ∈ {0.0, 0.1, ..., 1.0}, average the metric over *all* (other-weights)
    combos with that value of A on the simplex. This is the "marginal effect"
    of axis A on the metric.

    Plots, per cell:
      - 8 thin colored lines: one per fold
      - 1 thick black line: mean across folds

    Why marginal averaging:
      Fixing only w_xgb leaves w_lstm and w_cnn free along the (1 - w_xgb)
      edge. Averaging over those captures "what does turning the XGB knob
      do, on average across the rest of the mix?"
    """
    metrics_cols = [
        ("AUC",       "AUC"),
        ("Accuracy",  "ACC@0.50"),
        ("Precision", "Prec@0.50"),
        ("Recall",    "Rec@0.50"),
        ("F1-score",  "F1@0.50"),
    ]
    axes_info = [
        ("w_xgb",  "XGB weight"),
        ("w_lstm", "LSTM weight"),
        ("w_cnn",  "CNN weight"),
    ]
    folds = sorted(df["test_year"].unique())
    cmap = plt.get_cmap("tab10")

    nrows = len(metrics_cols)
    ncols = len(axes_info)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 2.8 * nrows),
                             sharex=True)

    for r, (mlabel, mcol) in enumerate(metrics_cols):
        for c, (wcol, wlabel) in enumerate(axes_info):
            ax = axes[r, c]
            # per-fold thin lines
            for fi, ty in enumerate(folds):
                sub = df[df["test_year"] == ty]
                # marginal mean: at each value of wcol, average over all
                # complementary combos
                marg = (sub.groupby(wcol, as_index=False)[mcol]
                        .mean()
                        .sort_values(wcol))
                ax.plot(marg[wcol], marg[mcol],
                        color=cmap(fi % 10), alpha=0.55,
                        linewidth=1.0, label=str(int(ty)))
            # mean across folds, thick black
            marg_all = (df.groupby(wcol, as_index=False)[mcol]
                        .mean()
                        .sort_values(wcol))
            ax.plot(marg_all[wcol], marg_all[mcol], color="black",
                    linewidth=2.4, label="mean", zorder=10)
            # mark argmax of the mean line
            best_i = int(marg_all[mcol].idxmax())
            ax.scatter([marg_all.loc[best_i, wcol]],
                       [marg_all.loc[best_i, mcol]],
                       marker="*", s=120, color="red",
                       edgecolor="white", linewidth=0.7, zorder=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xlim(-0.02, 1.02)
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=10)
            if r == 0:
                ax.set_title(wlabel, fontsize=11)
            if r == nrows - 1:
                ax.set_xlabel("weight value")

    # one shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=9, fontsize=8, frameon=False)

    fig.suptitle(
        "Marginal effect of each model's weight on test metrics  "
        "(thin = per fold, bold black = mean over folds, ★ = mean argmax)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out = PLOT_DIR / "metric_vs_weight_marginal.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_3d_scatter(
    ax,
    w_xgb: np.ndarray,
    w_lstm: np.ndarray,
    w_cnn: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_simplex_edges: bool = True,
):
    """Render a 3D scatter on the (w_xgb, w_lstm, w_cnn) cube.

    Points lie on the 2-simplex (the triangle w_xgb+w_lstm+w_cnn=1).
    Color encodes `values`. Best point gets a red star.
    """
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1e-6

    sc = ax.scatter(
        w_xgb, w_lstm, w_cnn,
        c=values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=46, depthshade=True, edgecolor="black", linewidth=0.25,
    )

    if show_simplex_edges:
        # the three corners of the simplex
        corners = np.array([
            [1.0, 0.0, 0.0],   # XGB
            [0.0, 1.0, 0.0],   # LSTM
            [0.0, 0.0, 1.0],   # CNN
        ])
        for i in range(3):
            j = (i + 1) % 3
            ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color="gray", linewidth=0.8, alpha=0.5,
            )
        # corner labels
        ax.text(1.02, -0.02, -0.02, "XGB",  fontsize=9, weight="bold")
        ax.text(-0.02, 1.02, -0.02, "LSTM", fontsize=9, weight="bold")
        ax.text(-0.02, -0.02, 1.02, "CNN",  fontsize=9, weight="bold")

    # mark the best point
    best_i = int(np.nanargmax(values))
    ax.scatter(
        [w_xgb[best_i]], [w_lstm[best_i]], [w_cnn[best_i]],
        marker="*", s=240, color="red", edgecolor="white",
        linewidth=0.9, zorder=20,
    )

    ax.set_xlabel("w_xgb",  fontsize=9, labelpad=2)
    ax.set_ylabel("w_lstm", fontsize=9, labelpad=2)
    ax.set_zlabel("w_cnn",  fontsize=9, labelpad=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.set_title(title, fontsize=10, pad=6)
    # nicer viewing angle (looks "down" at the simplex face)
    ax.view_init(elev=22, azim=35)
    return sc


def plot_metric_3d_mean(df: pd.DataFrame) -> Path:
    """One figure, 5 3D panels: mean of each metric across folds."""
    cols = [
        ("AUC",          "AUC"),
        ("Accuracy@0.5", "ACC@0.50"),
        ("Precision@0.5","Prec@0.50"),
        ("Recall@0.5",   "Rec@0.50"),
        ("F1@0.5",       "F1@0.50"),
    ]
    mean = df.groupby(
        ["w_xgb", "w_lstm", "w_cnn"], as_index=False
    )[[c for _, c in cols]].mean()

    fig = plt.figure(figsize=(5.0 * len(cols), 4.6))
    for i, (label, col) in enumerate(cols, start=1):
        ax = fig.add_subplot(1, len(cols), i, projection="3d")
        vals = mean[col].to_numpy()
        sc = _draw_3d_scatter(
            ax,
            mean["w_xgb"].to_numpy(),
            mean["w_lstm"].to_numpy(),
            mean["w_cnn"].to_numpy(),
            vals,
            title=f"{label}   max={vals.max():.4f}",
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.10, shrink=0.7)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        "3D weight space — mean over 8 folds (★ = best mean; points on the simplex w_xgb+w_lstm+w_cnn=1)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out = PLOT_DIR / "metric_3d_mean.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_auc_3d_per_fold(df: pd.DataFrame) -> Path:
    """3D AUC scatter, one panel per fold (shared color scale)."""
    folds = sorted(df["test_year"].unique())
    n = len(folds)
    cols = 4
    rows = (n + cols - 1) // cols

    vmin = df["AUC"].min()
    vmax = df["AUC"].max()

    fig = plt.figure(figsize=(4.5 * cols, 4.2 * rows))
    last_sc = None
    for k, ty in enumerate(folds):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        sub = df[df["test_year"] == ty]
        last_sc = _draw_3d_scatter(
            ax,
            sub["w_xgb"].to_numpy(),
            sub["w_lstm"].to_numpy(),
            sub["w_cnn"].to_numpy(),
            sub["AUC"].to_numpy(),
            title=f"{int(ty)}   max AUC={sub['AUC'].max():.4f}",
            vmin=vmin, vmax=vmax,
        )

    if last_sc is not None:
        fig.subplots_adjust(right=0.91)
        cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        fig.colorbar(last_sc, cax=cax, label="AUC")

    fig.suptitle(
        "AUC in 3D weight space — one panel per fold (shared color scale, ★ = per-fold best)",
        fontsize=13, y=0.995,
    )
    out = PLOT_DIR / "auc_3d_per_fold.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_best_weights(best: pd.DataFrame) -> Path:
    """Bar chart of best-AUC weights per fold."""
    sub = best[best["metric"] == "AUC"].sort_values("test_year")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(sub))
    width = 0.27
    ax.bar(x - width, sub["w_xgb"].to_numpy(), width, label="w_xgb",
           color="#1f77b4")
    ax.bar(x, sub["w_lstm"].to_numpy(), width, label="w_lstm",
           color="#ff7f0e")
    ax.bar(x + width, sub["w_cnn"].to_numpy(), width, label="w_cnn",
           color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["test_year"].astype(int).tolist())
    ax.set_xlabel("test_year (fold)")
    ax.set_ylabel("weight")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right")

    # annotate each bar group with the best AUC value
    for i, auc in enumerate(sub["value"].to_numpy()):
        ax.text(i, 1.03, f"AUC={auc:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Best-AUC weight per fold (step=0.1)", fontsize=12, pad=22)
    fig.tight_layout()
    out = PLOT_DIR / "best_weight_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 78)
    print("Ensemble weight sweep — XGB + LSTM(general) + CNN (baselines only)")
    print("=" * 78)
    print("NOTE: ensemble is a static weighted average of pre-computed predictions —")
    print("      no training step → train_acc / val_acc do not apply to the ensemble.")
    print("      All metrics below are computed on the per-fold *test* set.")
    print()
    print("Loading inputs...")
    merged = build_merged()
    print(f"  merged rows={len(merged):,}  folds={sorted(merged['test_year'].unique())}")

    print("\nRunning weight sweep...")
    df = sweep(merged)
    df_path = OUT_DIR / "weight_sweep_metrics.csv"
    df.to_csv(df_path, index=False)
    print(f"  saved long-format metrics → {df_path}  ({len(df):,} rows)")

    # ---- best weights per (fold, metric, threshold) ----
    metric_cols = ["AUC"] + [
        f"{m}@{t:.2f}" for m in ("ACC", "Prec", "Rec", "F1") for t in THRESHOLDS
    ]
    best_rows = []
    for ty, g in df.groupby("test_year"):
        for col in metric_cols:
            idx = g[col].idxmax()
            r = df.loc[idx]
            best_rows.append({
                "test_year": int(ty),
                "metric": col,
                "value": float(r[col]),
                "w_xgb": float(r["w_xgb"]),
                "w_lstm": float(r["w_lstm"]),
                "w_cnn": float(r["w_cnn"]),
            })
    best = pd.DataFrame(best_rows)
    best_path = OUT_DIR / "best_weights_per_fold.csv"
    best.to_csv(best_path, index=False)
    print(f"  saved best-weights table → {best_path}")

    # ---- plots ----
    print("\nDrawing plots...")
    for ty in sorted(df["test_year"].unique()):
        path = plot_fold(df[df["test_year"] == ty].copy(), int(ty))
        print(f"  {path}")
    p1 = plot_auc_grid(df)
    print(f"  {p1}")
    p2 = plot_best_weights(best)
    print(f"  {p2}")
    p3 = plot_metrics_across_folds(df)
    print(f"  {p3}")
    p4 = plot_mean_simplex(df)
    print(f"  {p4}")
    p5 = plot_metric_vs_weight_marginal(df)
    print(f"  {p5}")
    p6 = plot_metric_3d_mean(df)
    print(f"  {p6}")
    p7 = plot_auc_3d_per_fold(df)
    print(f"  {p7}")

    # ---- quick console summary ----
    print("\n[Best AUC per fold]")
    sub = best[best["metric"] == "AUC"][["test_year", "w_xgb", "w_lstm", "w_cnn", "value"]]
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
