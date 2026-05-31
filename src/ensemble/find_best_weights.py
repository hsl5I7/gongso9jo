"""
Fine-grid ensemble weight search (step=0.01).

Sweeps the full 3-simplex (w_xgb + w_lstm + w_cnn = 1, weights ∈ {0, 0.01, ..., 1.00})
→ 5151 weight combos × 8 folds = 41,208 AUC evaluations.

For the best combos we additionally compute ACC/Prec/Rec/F1 at threshold 0.50.

Inputs match ensemble_xgb_lstm.py (XGB / LSTM general / CNN baseline).

Outputs (ensemble_output/):
  fine_sweep_auc.parquet            long-format: (test_year, w_xgb, w_lstm, w_cnn, AUC)
  fine_best_weights_per_fold.csv    best AUC weight per fold + Acc/Prec/Rec/F1 at thr=0.5
  fine_best_weight_overall.csv      best weight by mean AUC across folds (+ full metrics)
  fine_best_pooled.csv              best weight by AUC on pooled data (all folds together)

The script prints the winning weights at the end.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEP = 0.01
THRESHOLD = 0.50  # for Acc/Prec/Rec/F1 on the winning combos


# --------------------------------------------------------------------------- #
# Loaders (mirror ensemble_weight_sweep.py)                                   #
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
# Weight grid & fast AUC                                                      #
# --------------------------------------------------------------------------- #
def simplex_weights(step: float = STEP) -> np.ndarray:
    """All (wx, wl, wc) on the simplex with the given step. Shape (n_combos, 3)."""
    n = int(round(1.0 / step))
    rows = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            rows.append((i * step, j * step, k * step))
    return np.array(rows, dtype=np.float64)


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC via the rank-sum (Mann-Whitney U) formula.

    Matches sklearn.metrics.roc_auc_score for tie-free inputs.
    For our probabilities ties are extremely rare → discrepancy is <1e-6.
    """
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="quicksort")
    y_sorted = y[order]
    # rank i = position+1 in ascending sort; sum of ranks of positives
    sum_rank_pos = float((np.arange(1, n + 1) * y_sorted).sum())
    return (sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float = THRESHOLD) -> dict:
    yhat = (p >= thr).astype(int)
    return {
        f"ACC@{thr:.2f}":  accuracy_score(y, yhat),
        f"Prec@{thr:.2f}": precision_score(y, yhat, zero_division=0),
        f"Rec@{thr:.2f}":  recall_score(y, yhat, zero_division=0),
        f"F1@{thr:.2f}":   f1_score(y, yhat, zero_division=0),
    }


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #
def sweep_auc(merged: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Compute AUC for every (fold, weight combo). Returns long-format DF."""
    folds = sorted(merged["test_year"].unique())
    n_combos = len(weights)
    print(f"Sweep: {n_combos} weight combos × {len(folds)} folds = "
          f"{n_combos * len(folds):,} AUC evaluations")

    rows: list[tuple] = []
    for ty in folds:
        g = merged[merged["test_year"] == ty]
        y = g["label_binary"].to_numpy()
        P = np.column_stack([
            g["p_xgb"].to_numpy(),
            g["p_lstm"].to_numpy(),
            g["p_cnn"].to_numpy(),
        ])  # (n, 3)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        # batch score: (n, 3) @ (3, n_combos) → (n, n_combos)
        # do it in chunks to keep memory sane
        chunk = 512
        aucs = np.empty(n_combos)
        t_start = time.perf_counter()
        for off in range(0, n_combos, chunk):
            W = weights[off:off + chunk]  # (k, 3)
            S = P @ W.T                   # (n, k)
            for j in range(S.shape[1]):
                s = S[:, j]
                order = np.argsort(s, kind="quicksort")
                y_sorted = y[order]
                sum_rank_pos = float((np.arange(1, len(y) + 1) * y_sorted).sum())
                aucs[off + j] = (
                    (sum_rank_pos - n_pos * (n_pos + 1) / 2.0)
                    / (n_pos * n_neg)
                )
        elapsed = time.perf_counter() - t_start
        print(f"  fold {ty}  n={len(y):>6,}  done in {elapsed:.1f}s   "
              f"max AUC={aucs.max():.4f}")

        for (wx, wl, wc), a in zip(weights, aucs):
            rows.append((int(ty), wx, wl, wc, float(a)))

    df = pd.DataFrame(rows, columns=["test_year", "w_xgb", "w_lstm", "w_cnn", "AUC"])
    # round weights to 2dp for clean CSV
    df["w_xgb"] = df["w_xgb"].round(2)
    df["w_lstm"] = df["w_lstm"].round(2)
    df["w_cnn"] = df["w_cnn"].round(2)
    return df


# --------------------------------------------------------------------------- #
# Reports                                                                     #
# --------------------------------------------------------------------------- #
def best_per_fold(df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """For each fold, find max-AUC weight and also report Acc/Prec/Rec/F1 at thr=0.5."""
    rows = []
    for ty in sorted(df["test_year"].unique()):
        sub = df[df["test_year"] == ty]
        i = int(sub["AUC"].idxmax())
        wx = float(sub.at[i, "w_xgb"])
        wl = float(sub.at[i, "w_lstm"])
        wc = float(sub.at[i, "w_cnn"])
        auc = float(sub.at[i, "AUC"])

        g = merged[merged["test_year"] == ty]
        y = g["label_binary"].to_numpy()
        p = (wx * g["p_xgb"] + wl * g["p_lstm"] + wc * g["p_cnn"]).to_numpy()
        # cross-check with sklearn AUC (catches ties / drift)
        auc_sk = roc_auc_score(y, p)
        rec = {
            "test_year": int(ty),
            "w_xgb": wx, "w_lstm": wl, "w_cnn": wc,
            "AUC_fast": auc, "AUC_sklearn": auc_sk,
            **threshold_metrics(y, p),
            "n": int(len(y)),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def best_by_mean_auc(df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Best weight by mean AUC across folds (the most common 'overall' criterion)."""
    mean_auc = (
        df.groupby(["w_xgb", "w_lstm", "w_cnn"], as_index=False)["AUC"].mean()
    )
    i = int(mean_auc["AUC"].idxmax())
    wx = float(mean_auc.at[i, "w_xgb"])
    wl = float(mean_auc.at[i, "w_lstm"])
    wc = float(mean_auc.at[i, "w_cnn"])
    mean = float(mean_auc.at[i, "AUC"])

    # Per-fold breakdown with this weight
    rows = []
    for ty in sorted(merged["test_year"].unique()):
        g = merged[merged["test_year"] == ty]
        y = g["label_binary"].to_numpy()
        p = (wx * g["p_xgb"] + wl * g["p_lstm"] + wc * g["p_cnn"]).to_numpy()
        rows.append({
            "test_year": int(ty),
            "w_xgb": wx, "w_lstm": wl, "w_cnn": wc,
            "AUC": roc_auc_score(y, p),
            **threshold_metrics(y, p),
            "n": int(len(y)),
        })
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {
        "test_year": "ALL_mean",
        "w_xgb": wx, "w_lstm": wl, "w_cnn": wc,
        "AUC": mean,
        "ACC@0.50": out["ACC@0.50"].mean(),
        "Prec@0.50": out["Prec@0.50"].mean(),
        "Rec@0.50": out["Rec@0.50"].mean(),
        "F1@0.50": out["F1@0.50"].mean(),
        "n": int(out["n"].sum()),
    }
    return out


def best_on_pooled(merged: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Pool all folds together, then find best weight by overall AUC."""
    y = merged["label_binary"].to_numpy()
    P = np.column_stack([
        merged["p_xgb"].to_numpy(),
        merged["p_lstm"].to_numpy(),
        merged["p_cnn"].to_numpy(),
    ])
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    aucs = np.empty(len(weights))
    chunk = 256
    t0 = time.perf_counter()
    for off in range(0, len(weights), chunk):
        W = weights[off:off + chunk]
        S = P @ W.T
        for j in range(S.shape[1]):
            s = S[:, j]
            order = np.argsort(s, kind="quicksort")
            y_sorted = y[order]
            sum_rank_pos = float((np.arange(1, len(y) + 1) * y_sorted).sum())
            aucs[off + j] = (
                (sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            )
    print(f"Pooled sweep: {time.perf_counter() - t0:.1f}s")

    i = int(np.argmax(aucs))
    wx, wl, wc = float(weights[i, 0]), float(weights[i, 1]), float(weights[i, 2])
    p = (wx * merged["p_xgb"] + wl * merged["p_lstm"] + wc * merged["p_cnn"]).to_numpy()
    return pd.DataFrame([{
        "w_xgb": round(wx, 2), "w_lstm": round(wl, 2), "w_cnn": round(wc, 2),
        "AUC_pooled": roc_auc_score(y, p),
        **threshold_metrics(y, p),
        "n": int(len(y)),
    }])


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 78)
    print(f"Fine-grid ensemble weight search (step={STEP})")
    print("=" * 78)
    print("Loading merged predictions...")
    merged = build_merged()
    print(f"  rows={len(merged):,}  folds={sorted(merged['test_year'].unique())}")

    weights = simplex_weights(STEP)
    print(f"  weight grid: {len(weights):,} combos on the 3-simplex")

    t0 = time.perf_counter()
    df = sweep_auc(merged, weights)
    print(f"\nTotal sweep time: {time.perf_counter() - t0:.1f}s")

    auc_path = OUT_DIR / "fine_sweep_auc.parquet"
    df.to_parquet(auc_path, index=False)
    print(f"→ saved AUC table: {auc_path}  ({len(df):,} rows)")

    print("\n[Best AUC per fold]")
    per_fold = best_per_fold(df, merged)
    print(per_fold.round(4).to_string(index=False))
    per_fold_path = OUT_DIR / "fine_best_weights_per_fold.csv"
    per_fold.round(6).to_csv(per_fold_path, index=False)
    print(f"→ saved: {per_fold_path}")

    print("\n[Best by mean AUC across folds — 'overall' winning weight]")
    overall = best_by_mean_auc(df, merged)
    print(overall.round(4).to_string(index=False))
    overall_path = OUT_DIR / "fine_best_weight_overall.csv"
    overall.round(6).to_csv(overall_path, index=False)
    print(f"→ saved: {overall_path}")

    print("\n[Best by AUC on pooled data (all folds concatenated)]")
    pooled = best_on_pooled(merged, weights)
    print(pooled.round(4).to_string(index=False))
    pooled_path = OUT_DIR / "fine_best_pooled.csv"
    pooled.round(6).to_csv(pooled_path, index=False)
    print(f"→ saved: {pooled_path}")


if __name__ == "__main__":
    main()
