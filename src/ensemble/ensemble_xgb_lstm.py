"""
XGBoost + LSTM + CNN baseline ensemble (walk-forward, 3-model).

Inputs (baselines only — no yt_features variants):
  - XGBoost:  to_team_ensemble/predictions_all.parquet
              cols: date, ticker, p_pos, label_binary, test_year
              test_year: 2018..2026

  - LSTM:     lstm/models/lstm/general_v2/lookback_30/fold_*/test_predictions.parquet
              cols: date, ticker, prob_익절, label_binary
              fold 00..13 → test_year 2012..2025
              (general_yt is excluded — it uses yt_features)

  - CNN:      cnn_baseline_output/predictions_test_pattern_ty{YYYY}.csv
              cols: date, ticker, prob_익절, label_binary
              ty 2018..2026

Pipeline:
  1. Load and normalize the three sources to a common schema
     (date: datetime64[ns], ticker: 6-digit zero-padded str).
  2. Inner-join all three on (date, ticker). Overlap: 2018..2025.
  3. Verify label_binary agrees across sources; use XGB label as truth.
  4. Compute ensemble columns:
       - p_avg          : mean of (p_xgb, p_lstm, p_cnn)
       - p_avg_rank     : rank-normalize each per fold, then mean
       - pair baselines : p_xl, p_xc, p_lc (mean of pairs)
       - weighted grid  : a few sensible 3-way weight combos
  5. Report AUC overall + per-fold, plus precision (win-rate) at thresholds.
  6. Save merged parquet + metrics CSVs.

Output: ensemble_output/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
XGB_PATH = ROOT / "to_team_ensemble" / "predictions_all.parquet"
LSTM_BASE = ROOT / "lstm" / "models" / "lstm" / "general_v2" / "lookback_30"
CNN_DIR = ROOT / "outputs" / "cnn" / "baseline"
OUT_DIR = ROOT / "outputs" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = "3model_baseline"

# 3-way weight grid (xgb, lstm, cnn). Equal + each-model-heavy + a couple of mixes.
WEIGHT_GRID = [
    (1 / 3, 1 / 3, 1 / 3),
    (0.50, 0.25, 0.25),
    (0.25, 0.50, 0.25),
    (0.25, 0.25, 0.50),
    (0.40, 0.40, 0.20),
    (0.40, 0.20, 0.40),
    (0.20, 0.40, 0.40),
    (0.50, 0.30, 0.20),
    (0.50, 0.20, 0.30),
    (0.30, 0.50, 0.20),
    (0.20, 0.50, 0.30),
    (0.30, 0.20, 0.50),
    (0.20, 0.30, 0.50),
]

# stdout: ensure utf-8 so prob_익절 doesn't blow up on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def load_xgb(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={"p_pos": "p_xgb"})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    return df[["date", "ticker", "p_xgb", "label_binary", "test_year"]]


def load_lstm(base: Path) -> pd.DataFrame:
    folds = sorted(base.glob("fold_*"))
    parts = []
    for f in folds:
        p = f / "test_predictions.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) == 1, f"unexpected prob cols in {p}: {df.columns.tolist()}"
        df = df.rename(columns={prob_cols[0]: "p_lstm"})
        # fold_NN_testYYYY → test_year
        test_year = int(f.name.split("test")[-1])
        df["test_year"] = test_year
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out[["date", "ticker", "p_lstm", "label_binary", "test_year"]]


def load_cnn(cnn_dir: Path) -> pd.DataFrame:
    files = sorted(cnn_dir.glob("predictions_test_pattern_ty*.csv"))
    assert files, f"no CNN CSVs under {cnn_dir}"
    parts = []
    for f in files:
        df = pd.read_csv(f)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) == 1, f"unexpected prob cols in {f}: {df.columns.tolist()}"
        df = df.rename(columns={prob_cols[0]: "p_cnn"})
        # filename: predictions_test_pattern_ty2018.csv → test_year=2018
        test_year = int(f.stem.split("ty")[-1])
        df["test_year"] = test_year
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out[["date", "ticker", "p_cnn", "label_binary", "test_year"]]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def rank_norm(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def summarize(merged: pd.DataFrame, prob_col: str, label_col: str = "label_binary") -> dict:
    mask = merged[label_col].notna()
    y = merged.loc[mask, label_col].astype(int).values
    p = merged.loc[mask, prob_col].astype(float).values
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    base_rate = float(y.mean())
    out = {"AUC": auc, "AP": ap, "base_rate": base_rate, "n": int(mask.sum())}
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
        sel = p >= thr
        n = int(sel.sum())
        win = float(y[sel].mean()) if n > 0 else float("nan")
        out[f"n@{thr:.2f}"] = n
        out[f"win@{thr:.2f}"] = win
    return out


def per_fold_auc(merged: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    rows = []
    for ty, g in merged[merged["label_binary"].notna()].groupby("test_year"):
        y = g["label_binary"].astype(int).values
        p = g[prob_col].astype(float).values
        auc = np.nan if len(np.unique(y)) < 2 else roc_auc_score(y, p)
        rows.append({"test_year": ty, "n": len(g), prob_col: auc})
    return pd.DataFrame(rows).set_index("test_year")


def _w_col(wx: float, wl: float, wc: float) -> str:
    return f"p_w_{int(round(wx*100)):02d}xgb_{int(round(wl*100)):02d}lstm_{int(round(wc*100)):02d}cnn"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"XGB:  {XGB_PATH}")
    print(f"LSTM: {LSTM_BASE}")
    print(f"CNN:  {CNN_DIR}")
    print(f"OUT:  {OUT_DIR}\n")

    xgb = load_xgb(XGB_PATH)
    print(f"XGB  rows={len(xgb):,}  years={sorted(xgb['test_year'].unique())}")
    lstm = load_lstm(LSTM_BASE)
    print(f"LSTM rows={len(lstm):,}  years={sorted(lstm['test_year'].unique())}")
    cnn = load_cnn(CNN_DIR)
    print(f"CNN  rows={len(cnn):,}  years={sorted(cnn['test_year'].unique())}")

    # ---- 3-way inner join on (date, ticker) ----
    # carry each source's label/test_year through so we can validate
    xgb_r = xgb.rename(columns={"label_binary": "lbl_xgb", "test_year": "ty_xgb"})
    lstm_r = lstm.rename(columns={"label_binary": "lbl_lstm", "test_year": "ty_lstm"})
    cnn_r = cnn.rename(columns={"label_binary": "lbl_cnn", "test_year": "ty_cnn"})

    merged = xgb_r.merge(lstm_r, on=["date", "ticker"], how="inner")
    merged = merged.merge(cnn_r, on=["date", "ticker"], how="inner")
    print(
        f"\nMerged (XGB ∩ LSTM ∩ CNN): {len(merged):,} rows  "
        f"years={sorted(merged['ty_xgb'].unique())}"
    )

    # Label consistency check (XGB authoritative — same B_outcome label upstream)
    for col in ("lbl_lstm", "lbl_cnn"):
        both = merged["lbl_xgb"].notna() & merged[col].notna()
        if both.sum():
            eq = (merged.loc[both, "lbl_xgb"] == merged.loc[both, col]).mean()
            print(f"  label match XGB vs {col}: {eq * 100:.2f}%  ({both.sum():,} rows)")

    # test_year consistency check
    ty_match = (
        (merged["ty_xgb"] == merged["ty_lstm"]) & (merged["ty_xgb"] == merged["ty_cnn"])
    ).mean()
    print(f"  test_year match across 3 sources: {ty_match * 100:.2f}%")

    merged["label_binary"] = (
        merged["lbl_xgb"]
        .combine_first(merged["lbl_lstm"])
        .combine_first(merged["lbl_cnn"])
    )
    merged["test_year"] = merged["ty_xgb"]
    merged = merged.drop(columns=["lbl_xgb", "lbl_lstm", "lbl_cnn", "ty_xgb", "ty_lstm", "ty_cnn"])

    # ---- Ensembles ----
    merged["p_avg"] = (merged["p_xgb"] + merged["p_lstm"] + merged["p_cnn"]) / 3.0

    # rank-normalized per fold (scales differ between models)
    merged["p_xgb_rk"] = merged.groupby("test_year")["p_xgb"].transform(rank_norm)
    merged["p_lstm_rk"] = merged.groupby("test_year")["p_lstm"].transform(rank_norm)
    merged["p_cnn_rk"] = merged.groupby("test_year")["p_cnn"].transform(rank_norm)
    merged["p_avg_rank"] = (
        merged["p_xgb_rk"] + merged["p_lstm_rk"] + merged["p_cnn_rk"]
    ) / 3.0

    # pair averages — useful for sanity-checking what CNN adds vs XGB+LSTM only
    merged["p_xl"] = (merged["p_xgb"] + merged["p_lstm"]) / 2.0
    merged["p_xc"] = (merged["p_xgb"] + merged["p_cnn"]) / 2.0
    merged["p_lc"] = (merged["p_lstm"] + merged["p_cnn"]) / 2.0

    # weighted 3-way grid
    weight_cols: list[str] = []
    for wx, wl, wc in WEIGHT_GRID:
        col = _w_col(wx, wl, wc)
        merged[col] = wx * merged["p_xgb"] + wl * merged["p_lstm"] + wc * merged["p_cnn"]
        weight_cols.append(col)

    # ---- Reports ----
    print("\n[Overall metrics]")
    eval_cols = (
        ["p_xgb", "p_lstm", "p_cnn", "p_xl", "p_xc", "p_lc", "p_avg", "p_avg_rank"]
        + weight_cols
    )
    rows = []
    for col in eval_cols:
        s = summarize(merged, col)
        s["model"] = col
        rows.append(s)
    summary = pd.DataFrame(rows).set_index("model")
    show = summary[
        [
            "n", "base_rate", "AUC", "AP",
            "n@0.55", "win@0.55", "n@0.60", "win@0.60",
            "n@0.65", "win@0.65", "n@0.70", "win@0.70",
        ]
    ]
    print(show.round(4).to_string())

    print("\n[Per-fold AUC]")
    base_prob_cols = ["p_xgb", "p_lstm", "p_cnn", "p_avg", "p_avg_rank"]
    per_fold = pd.concat(
        [per_fold_auc(merged, c)[[c]] for c in base_prob_cols],
        axis=1,
    )
    n_per_fold = per_fold_auc(merged, "p_xgb")[["n"]]
    per_fold = pd.concat([n_per_fold, per_fold], axis=1)
    print(per_fold.round(4).to_string())
    print("\n  mean AUC: " + ", ".join(f"{c}={per_fold[c].mean():.4f}" for c in base_prob_cols))

    # ---- Save ----
    out_pred = OUT_DIR / f"ensemble_{VARIANT}.parquet"
    keep_cols = (
        ["date", "ticker", "test_year", "label_binary",
         "p_xgb", "p_lstm", "p_cnn",
         "p_xl", "p_xc", "p_lc",
         "p_avg", "p_avg_rank"]
        + weight_cols
    )
    merged[keep_cols].to_parquet(out_pred, index=False)
    print(f"\n→ saved: {out_pred}  ({len(merged):,} rows)")

    out_metrics = OUT_DIR / f"metrics_{VARIANT}.csv"
    show.round(6).to_csv(out_metrics)
    out_per_fold = OUT_DIR / f"per_fold_auc_{VARIANT}.csv"
    per_fold.round(6).to_csv(out_per_fold)
    print(f"→ saved: {out_metrics}")
    print(f"→ saved: {out_per_fold}")


if __name__ == "__main__":
    main()
