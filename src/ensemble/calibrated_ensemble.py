"""
Forward-only isotonic calibration of XGB / LSTM / CNN per fold, with a
bootstrap-derived 90% CI on the calibration mapping, then weighted ensemble
at (0.78, 0.07, 0.15) — produced for both `baseline` and `youtube` variants.

Procedure (per variant):
  1. Load raw per-model predictions via ensemble_3model loaders.
  2. For each fold (test_year Y):
       calibration set = predictions on all years < Y (forward-only, no leak).
       For Y = first available year there is no past → use raw scores.
       Fit `B` bootstrap-resampled IsotonicRegression calibrators per model.
       Also fit one calibrator on the full calibration set → point estimate.
       For each test row, get B calibrated values per model → 5/95 percentile.
  3. Ensemble (point and per-bootstrap-sample):
       p_ens = w_xgb * p_xgb_cal + w_lstm * p_lstm_cal + w_cnn * p_cnn_cal.
       Take 5/95 percentile of the B ensemble values → row-wise 90% CI.

Outputs (per variant):
  outputs/ensemble/{variant}_calibrated/predictions_test_pattern_<stem>_ty{YYYY}.csv
      cols: date, ticker, prob_익절, prob_익절_lo90, prob_익절_hi90, label_binary
  outputs/ensemble/auroc_calibrated/
      pooled_roc.png, per_fold_roc_{variant}.png,
      per_fold_auroc.csv, per_fold_auroc_bars.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve

from ensemble_3model import VARIANTS, VariantPaths, load_cnn, load_lstm, load_xgb


ROOT = Path(__file__).resolve().parents[2]
ENS_ROOT = ROOT / "outputs" / "ensemble"
AUROC_DIR = ENS_ROOT / "auroc_calibrated"
AUROC_DIR.mkdir(parents=True, exist_ok=True)

FNAME_TEMPLATES = {
    "baseline": "predictions_test_pattern_ty{year}.csv",
    "youtube": "predictions_test_pattern_yt12_ty{year}.csv",
}

WEIGHTS = (0.78, 0.07, 0.15)         # (w_xgb, w_lstm, w_cnn)
MODEL_KEYS = ("p_xgb", "p_lstm", "p_cnn")
CI_LO, CI_HI = 5.0, 95.0             # 90 % CI percentiles
B_BOOTSTRAP = 200                    # bootstrap calibrators per (model, fold)
RNG_SEED = 20260529

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Data prep                                                                   #
# --------------------------------------------------------------------------- #
def merge_three(v: VariantPaths) -> pd.DataFrame:
    """XGB ∩ LSTM ∩ CNN inner-join on (date, ticker) — same as ensemble_3model."""
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
# Calibration                                                                 #
# --------------------------------------------------------------------------- #
def _fit_iso(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def calibrate_fold(
    cal_p: np.ndarray, cal_y: np.ndarray, test_p: np.ndarray,
    *, b: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (p_point, p_boot) where:
      p_point : (n_test,)        — isotonic fit on full calibration set
      p_boot  : (b, n_test)      — b bootstrap calibrators applied to test
    Caller takes percentiles on p_boot for the CI.
    """
    p_point = _fit_iso(cal_p, cal_y).transform(test_p)

    n = cal_p.size
    p_boot = np.empty((b, test_p.size), dtype=np.float64)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        p_boot[i] = _fit_iso(cal_p[idx], cal_y[idx]).transform(test_p)
    return p_point, p_boot


def build_calibrated_ensemble(
    merged: pd.DataFrame, weights: tuple[float, float, float] = WEIGHTS,
    *, b: int = B_BOOTSTRAP, rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """For each test_year, forward-only bootstrap-isotonic calibrate each model,
    then weighted-ensemble (point + per-row 90 % CI)."""
    rng = rng if rng is not None else np.random.default_rng(RNG_SEED)
    w = np.asarray(weights, dtype=np.float64)
    assert np.isclose(w.sum(), 1.0), f"weights sum != 1: {w.sum()}"

    years = sorted(merged["test_year"].unique())
    parts: list[pd.DataFrame] = []

    for y in years:
        t0 = time.time()
        cur = merged[merged["test_year"] == y]
        past = merged[merged["test_year"] < y]

        # Test predictions per model
        test_p = {k: cur[k].to_numpy(dtype=np.float64) for k in MODEL_KEYS}
        n_test = len(cur)

        if past.empty:
            # First fold: no past → raw scores, zero-width CI
            p_point = {k: test_p[k].copy() for k in MODEL_KEYS}
            p_boot = {k: np.broadcast_to(test_p[k], (b, n_test)).copy() for k in MODEL_KEYS}
            note = "raw (no past)"
        else:
            cal_y = past["label_binary"].astype(int).to_numpy()
            p_point: dict[str, np.ndarray] = {}
            p_boot: dict[str, np.ndarray] = {}
            for k in MODEL_KEYS:
                cal_p = past[k].to_numpy(dtype=np.float64)
                pt, bt = calibrate_fold(cal_p, cal_y, test_p[k], b=b, rng=rng)
                p_point[k] = pt
                p_boot[k]  = bt
            note = f"iso on {len(past):,} past rows"

        # Ensemble: point + per-bootstrap row → 5/95 percentile
        ens_point = sum(w[i] * p_point[MODEL_KEYS[i]] for i in range(3))
        ens_boot = np.zeros((b, n_test), dtype=np.float64)
        for i, k in enumerate(MODEL_KEYS):
            ens_boot += w[i] * p_boot[k]
        ens_lo = np.percentile(ens_boot, CI_LO, axis=0)
        ens_hi = np.percentile(ens_boot, CI_HI, axis=0)

        out = cur[["date", "ticker", "test_year", "label_binary"]].copy()
        out["prob_익절"] = ens_point
        out["prob_익절_lo90"] = ens_lo
        out["prob_익절_hi90"] = ens_hi
        # Per-model calibrated points are handy for downstream debugging
        for k in MODEL_KEYS:
            out[f"{k}_cal"] = p_point[k]
        parts.append(out)
        print(f"    ty{y} n={n_test:,}  ci_width_med={np.median(ens_hi - ens_lo):.4f}  "
              f"({note})  [{time.time() - t0:.1f}s]")

    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# AUROC analysis                                                              #
# --------------------------------------------------------------------------- #
def per_fold_auc(df: pd.DataFrame) -> pd.Series:
    rows = {}
    for y, g in df.groupby("test_year"):
        if g["label_binary"].nunique() == 2:
            rows[int(y)] = roc_auc_score(g["label_binary"], g["prob_익절"])
        else:
            rows[int(y)] = np.nan
    return pd.Series(rows, name="AUC")


def pooled_auc(df: pd.DataFrame) -> float:
    return float(roc_auc_score(df["label_binary"], df["prob_익절"]))


def plot_pooled_roc(data: dict[str, pd.DataFrame], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    colors = {"baseline": "#1f77b4", "youtube": "#d62728"}
    for variant, df in data.items():
        fpr, tpr, _ = roc_curve(df["label_binary"], df["prob_익절"])
        auc = pooled_auc(df)
        ax.plot(fpr, tpr, color=colors[variant], lw=2.2,
                label=f"{variant}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled ROC — calibrated ensembles", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.suptitle(
        f"AUROC — calibrated 3-model ensemble  (weights: "
        f"{WEIGHTS[0]}/{WEIGHTS[1]}/{WEIGHTS[2]})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


def plot_per_fold_roc(variant: str, df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    cmap = plt.get_cmap("viridis")
    years = sorted(df["test_year"].unique())
    for i, y in enumerate(years):
        g = df[df["test_year"] == y]
        if g["label_binary"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["label_binary"], g["prob_익절"])
        auc = roc_auc_score(g["label_binary"], g["prob_익절"])
        ax.plot(fpr, tpr, color=cmap(i / max(len(years) - 1, 1)), lw=1.6,
                label=f"ty{int(y)}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Per-fold ROC — {variant} (calibrated)", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"AUROC — calibrated 3-model ensemble  ({variant})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


def plot_per_fold_bars(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    years = [str(y) for y in table.index if isinstance(y, (int, np.integer))]
    x = np.arange(len(years))
    w = 0.38
    base_vals = [table.loc[int(y), "baseline"] for y in years]
    yt_vals = [table.loc[int(y), "youtube"] for y in years]
    b1 = ax.bar(x - w / 2, base_vals, w, label="baseline (calibrated)", color="#1f77b4")
    b2 = ax.bar(x + w / 2, yt_vals, w, label="youtube (calibrated)", color="#d62728")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.003,
                    f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    if "pooled" in table.index:
        ax.axhline(table.loc["pooled", "baseline"], color="#1f77b4", ls=":",
                   lw=1.2, alpha=0.7,
                   label=f"baseline pooled = {table.loc['pooled', 'baseline']:.4f}")
        ax.axhline(table.loc["pooled", "youtube"], color="#d62728", ls=":",
                   lw=1.2, alpha=0.7,
                   label=f"youtube  pooled = {table.loc['pooled', 'youtube']:.4f}")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="chance (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_xlabel("Test year (fold)"); ax.set_ylabel("AUROC")
    ax.set_title("Per-fold AUROC — calibrated baseline vs calibrated youtube", fontsize=11)
    ymin = min(min(base_vals), min(yt_vals)) - 0.02
    ax.set_ylim(max(ymin, 0.40), max(max(base_vals), max(yt_vals)) + 0.04)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("AUROC — calibrated ensemble, per fold", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  → {out}")


# --------------------------------------------------------------------------- #
# Save                                                                        #
# --------------------------------------------------------------------------- #
def write_per_year_csvs(variant: str, ens: pd.DataFrame) -> None:
    out_dir = ENS_ROOT / f"{variant}_calibrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = FNAME_TEMPLATES[variant]
    for y, g in ens.groupby("test_year"):
        path = out_dir / tmpl.format(year=int(y))
        cols = ["date", "ticker", "prob_익절", "prob_익절_lo90", "prob_익절_hi90", "label_binary"]
        out = g[cols].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
        print(f"    → {path}  rows={len(out):,}  ci_width_med="
              f"{np.median(out['prob_익절_hi90'] - out['prob_익절_lo90']):.4f}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run(variant: str, b: int) -> pd.DataFrame:
    v = VARIANTS[variant]
    print(f"\n{'=' * 76}\nVariant: {variant}   weights={WEIGHTS}   B={b}\n{'=' * 76}")
    merged = merge_three(v)
    print(f"  merged rows={len(merged):,}  years={sorted(merged['test_year'].unique())}")

    rng = np.random.default_rng(RNG_SEED + hash(variant) % 10_000)
    ens = build_calibrated_ensemble(merged, weights=WEIGHTS, b=b, rng=rng)

    print(f"\n  → writing per-year CSVs")
    write_per_year_csvs(variant, ens)
    return ens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "youtube", "both"], default="both")
    ap.add_argument("--bootstrap", type=int, default=B_BOOTSTRAP,
                    help="number of bootstrap calibrators per model per fold")
    args = ap.parse_args()
    targets = ["baseline", "youtube"] if args.variant == "both" else [args.variant]

    data: dict[str, pd.DataFrame] = {}
    for name in targets:
        data[name] = run(name, args.bootstrap)

    if len(data) == 2:
        # --- AUROC summary table + plots ---
        print(f"\n{'=' * 76}\nAUROC analysis\n{'=' * 76}")
        table = pd.DataFrame({name: per_fold_auc(df) for name, df in data.items()})
        table.index.name = "test_year"
        pooled = pd.Series({name: pooled_auc(df) for name, df in data.items()}, name="pooled")
        table = pd.concat([table, pooled.to_frame().T])
        int_rows = [i for i in table.index if isinstance(i, (int, np.integer))]
        table.loc["mean_per_fold"] = table.loc[int_rows].mean()
        print(table.round(4).to_string())
        out_csv = AUROC_DIR / "per_fold_auroc.csv"
        table.round(6).to_csv(out_csv); print(f"\n  → {out_csv}")

        print("\nRendering plots …")
        plot_pooled_roc(data, AUROC_DIR / "pooled_roc.png")
        for name, df in data.items():
            plot_per_fold_roc(name, df, AUROC_DIR / f"per_fold_roc_{name}.png")
        plot_per_fold_bars(table, AUROC_DIR / "per_fold_auroc_bars.png")


if __name__ == "__main__":
    main()
