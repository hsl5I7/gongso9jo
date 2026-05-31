"""
3-model ensemble (XGB + LSTM + CNN) for `baseline` and `youtube` feature sets.

Inputs per variant:

  baseline:
    XGB:  xgboost/wf_v4_70feat_thr05/predictions_all_70feat_noYT.parquet
          cols: date, ticker, p_pos, label_binary, test_year
          (numerically identical to to_team_ensemble/xgb_baseline_output)
    LSTM: lstm/models/lstm/general_v2/lookback_30/fold_*/test_predictions.parquet
          cols: date, ticker, prob_익절, label_binary   (test_year from fold name)
    CNN:  outputs/cnn/baseline/predictions_test_pattern_ty{YYYY}.csv
          cols: date, ticker, prob_익절, label_binary

  youtube:
    XGB:  xgboost/wf_v4_77feat_thr05/predictions_all_77feat_Aonly.parquet
          cols: date, ticker, p_pos, label_binary, test_year
          (77feat_Aonly = 70 base + 7 A-group youtuber signals;
           same feature set as LSTM general_v2_yt → all 3 youtube models aligned)
    LSTM: lstm/models/lstm/general_v2_yt/lookback_30/fold_*/test_predictions.parquet
          cols: date, ticker, prob_익절, label_binary
    CNN:  outputs/cnn/youtube/predictions_test_pattern_yt12_ty{YYYY}.csv
          cols: date, ticker, prob_익절, label_binary

Pipeline (same for both variants):
  1. Load and normalize to (date, ticker, p_<model>, label_binary, test_year).
  2. Inner-join all three on (date, ticker).
  3. Verify label_binary agrees across sources; XGB label is authoritative.
  4. Compute ensembles: simple mean, rank-norm mean (per fold), pair means,
     weighted grid over (w_xgb, w_lstm, w_cnn).
  5. Report AUC overall + per-fold, plus n@/win@ at thresholds.
  6. Save merged parquet + metrics CSVs to ensemble_output/.

Usage:
  python ensemble_3model.py                  # runs both variants
  python ensemble_3model.py --variant baseline
  python ensemble_3model.py --variant youtube
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# stdout: utf-8 so prob_익절 doesn't blow up on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Variant config                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class VariantPaths:
    name: str
    xgb_dir: Path
    xgb_glob: str
    xgb_format: str        # "csv_prob_익절_by_filename" | "parquet_p_pos_with_year"
    lstm_base: Path
    cnn_dir: Path
    cnn_glob: str


VARIANTS: dict[str, VariantPaths] = {
    "baseline": VariantPaths(
        name="baseline",
        xgb_dir=ROOT / "xgboost" / "wf_v4_70feat_thr05",
        xgb_glob="predictions_all_70feat_noYT.parquet",
        xgb_format="parquet_p_pos_with_year",
        lstm_base=ROOT / "lstm" / "models" / "lstm" / "general_v2" / "lookback_30",
        cnn_dir=ROOT / "outputs" / "cnn" / "baseline",
        cnn_glob="predictions_test_pattern_ty*.csv",
    ),
    "youtube": VariantPaths(
        name="youtube",
        xgb_dir=ROOT / "xgboost" / "wf_v4_77feat_thr05",
        xgb_glob="predictions_all_77feat_Aonly.parquet",
        xgb_format="parquet_p_pos_with_year",
        lstm_base=ROOT / "lstm" / "models" / "lstm" / "general_v2_yt" / "lookback_30",
        cnn_dir=ROOT / "outputs" / "cnn" / "youtube",
        cnn_glob="predictions_test_pattern_yt12_ty*.csv",
    ),
}


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def _normalize(df: pd.DataFrame, prob_in: str, prob_out: str) -> pd.DataFrame:
    df = df.rename(columns={prob_in: prob_out})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    return df


def _year_from_filename(stem: str) -> int:
    # works for "predictions_test_pattern_ty2018",
    #          "predictions_test_pattern_yt12_ty2018",
    #          "predictions_fold_2018_99feat_full"
    if "ty" in stem:
        tail = stem.rsplit("ty", 1)[-1]          # "2018" or "2018_..."
    else:
        tail = stem.split("fold_", 1)[-1]        # "2018_99feat_full"
    return int(tail.split("_", 1)[0])


def load_xgb(v: VariantPaths) -> pd.DataFrame:
    files = sorted(v.xgb_dir.glob(v.xgb_glob))
    assert files, f"no XGB files under {v.xgb_dir} matching {v.xgb_glob}"
    parts = []
    for f in files:
        if v.xgb_format == "csv_prob_익절_by_filename":
            df = pd.read_csv(f)
            df = _normalize(df, "prob_익절", "p_xgb")
            df["test_year"] = _year_from_filename(f.stem)
        elif v.xgb_format == "parquet_p_pos_with_year":
            df = pd.read_parquet(f)
            df = _normalize(df, "p_pos", "p_xgb")
            # test_year already present
        else:
            raise ValueError(f"unknown xgb_format: {v.xgb_format}")
        parts.append(df[["date", "ticker", "p_xgb", "label_binary", "test_year"]])
    return pd.concat(parts, ignore_index=True)


def load_lstm(v: VariantPaths) -> pd.DataFrame:
    folds = sorted(v.lstm_base.glob("fold_*"))
    assert folds, f"no LSTM folds under {v.lstm_base}"
    parts = []
    for f in folds:
        p = f / "test_predictions.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) == 1, f"unexpected prob cols in {p}: {df.columns.tolist()}"
        df = _normalize(df, prob_cols[0], "p_lstm")
        # fold_NN_testYYYY → test_year
        df["test_year"] = int(f.name.split("test")[-1])
        parts.append(df[["date", "ticker", "p_lstm", "label_binary", "test_year"]])
    return pd.concat(parts, ignore_index=True)


def load_cnn(v: VariantPaths) -> pd.DataFrame:
    files = sorted(v.cnn_dir.glob(v.cnn_glob))
    assert files, f"no CNN files under {v.cnn_dir} matching {v.cnn_glob}"
    parts = []
    for f in files:
        df = pd.read_csv(f)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        assert len(prob_cols) == 1, f"unexpected prob cols in {f}: {df.columns.tolist()}"
        df = _normalize(df, prob_cols[0], "p_cnn")
        df["test_year"] = _year_from_filename(f.stem)
        parts.append(df[["date", "ticker", "p_cnn", "label_binary", "test_year"]])
    return pd.concat(parts, ignore_index=True)


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
    return (
        f"p_w_{int(round(wx*100)):02d}xgb"
        f"_{int(round(wl*100)):02d}lstm"
        f"_{int(round(wc*100)):02d}cnn"
    )


# --------------------------------------------------------------------------- #
# Run one variant                                                             #
# --------------------------------------------------------------------------- #
def run_variant(v: VariantPaths) -> None:
    tag = f"3model_{v.name}"
    print(f"\n{'=' * 76}\nVariant: {v.name}\n{'=' * 76}")
    print(f"XGB:  {v.xgb_dir}")
    print(f"LSTM: {v.lstm_base}")
    print(f"CNN:  {v.cnn_dir}")
    print(f"OUT:  {OUT_DIR}\n")

    xgb = load_xgb(v)
    print(f"XGB  rows={len(xgb):,}  years={sorted(xgb['test_year'].unique())}")
    lstm = load_lstm(v)
    print(f"LSTM rows={len(lstm):,}  years={sorted(lstm['test_year'].unique())}")
    cnn = load_cnn(v)
    print(f"CNN  rows={len(cnn):,}  years={sorted(cnn['test_year'].unique())}")

    # ---- 3-way inner join on (date, ticker) ----
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

    # pair averages
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
    out_pred = OUT_DIR / f"ensemble_{tag}.parquet"
    keep_cols = (
        ["date", "ticker", "test_year", "label_binary",
         "p_xgb", "p_lstm", "p_cnn",
         "p_xl", "p_xc", "p_lc",
         "p_avg", "p_avg_rank"]
        + weight_cols
    )
    merged[keep_cols].to_parquet(out_pred, index=False)
    print(f"\n→ saved: {out_pred}  ({len(merged):,} rows)")

    out_metrics = OUT_DIR / f"metrics_{tag}.csv"
    show.round(6).to_csv(out_metrics)
    out_per_fold = OUT_DIR / f"per_fold_auc_{tag}.csv"
    per_fold.round(6).to_csv(out_per_fold)
    print(f"→ saved: {out_metrics}")
    print(f"→ saved: {out_per_fold}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="3-model ensemble (XGB+LSTM+CNN)")
    ap.add_argument(
        "--variant",
        choices=["baseline", "youtube", "both"],
        default="both",
        help="which feature set to run (default: both)",
    )
    args = ap.parse_args()

    targets = ["baseline", "youtube"] if args.variant == "both" else [args.variant]
    for name in targets:
        run_variant(VARIANTS[name])


if __name__ == "__main__":
    main()
