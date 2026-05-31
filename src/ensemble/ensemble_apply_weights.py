"""
Apply a fixed (w_xgb, w_lstm, w_cnn) weight to the 3-model ensemble and save
per-test-year CSVs that mirror the CNN input format.

For each variant we write one CSV per year to:
    outputs/ensemble/{variant}/predictions_test_pattern_<stem>_ty{YYYY}.csv
        cols: date, ticker, prob_익절, label_binary

The `<stem>` matches the CNN file naming so the ensemble outputs slot into
the same directory layout as the per-model outputs:
    baseline → predictions_test_pattern_ty{YYYY}.csv
    youtube  → predictions_test_pattern_yt12_ty{YYYY}.csv

Loaders and join logic are imported from ensemble_3model.py so the merged
universe stays identical to the headline 3-model ensemble.

Usage:
    python ensemble_apply_weights.py                       # 0.78/0.07/0.15, both variants
    python ensemble_apply_weights.py --w 0.5 0.25 0.25     # custom weights
    python ensemble_apply_weights.py --variant baseline
"""
from __future__ import annotations

import argparse
import sys
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

from ensemble_3model import VARIANTS, VariantPaths, load_cnn, load_lstm, load_xgb


ROOT = Path(__file__).resolve().parents[2]
ENS_ROOT = ROOT / "outputs" / "ensemble"

# Per-variant filename templates — match the existing CNN naming so the
# ensemble files drop into the same directory layout.
FNAME_TEMPLATES = {
    "baseline": "predictions_test_pattern_ty{year}.csv",
    "youtube": "predictions_test_pattern_yt12_ty{year}.csv",
}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def merge_three(v: VariantPaths) -> pd.DataFrame:
    """Inner-join XGB ∩ LSTM ∩ CNN on (date, ticker); same recipe as
    `ensemble_3model.run_variant` so we share the same evaluation universe."""
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


def run(variant: str, w_xgb: float, w_lstm: float, w_cnn: float, thr: float = 0.50) -> None:
    v = VARIANTS[variant]
    out_dir = ENS_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    fname_tmpl = FNAME_TEMPLATES[variant]

    print(f"\n{'=' * 76}\nVariant: {variant}   weights=(xgb={w_xgb}, lstm={w_lstm}, cnn={w_cnn})\n{'=' * 76}")
    m = merge_three(v)
    print(f"  merged rows={len(m):,}  years={sorted(m['test_year'].unique())}")

    m["prob_익절"] = w_xgb * m["p_xgb"] + w_lstm * m["p_lstm"] + w_cnn * m["p_cnn"]

    # Write per-year CSVs that mirror the CNN input schema
    written = []
    for year, g in m.groupby("test_year"):
        out = out_dir / fname_tmpl.format(year=int(year))
        g_out = g[["date", "ticker", "prob_익절", "label_binary"]].copy()
        g_out["date"] = pd.to_datetime(g_out["date"]).dt.strftime("%Y-%m-%d")
        g_out.to_csv(out, index=False)
        written.append((int(year), out, len(g_out)))
    for year, path, n in written:
        print(f"  → {path}   rows={n:,}")

    # Sanity metrics on the union
    y = m["label_binary"].astype(int).values
    p = m["prob_익절"].astype(float).values
    pred = (p >= thr).astype(int)
    print(
        "\n  metrics (pooled, all years):"
        f"  AUC={roc_auc_score(y, p):.4f}"
        f"  Acc={accuracy_score(y, pred):.4f}"
        f"  Prec={precision_score(y, pred, zero_division=0):.4f}"
        f"  Rec={recall_score(y, pred, zero_division=0):.4f}"
        f"  F1={f1_score(y, pred, zero_division=0):.4f}"
        f"  (thr={thr:.2f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "youtube", "both"], default="both")
    ap.add_argument(
        "--w", nargs=3, type=float, metavar=("W_XGB", "W_LSTM", "W_CNN"),
        default=[0.78, 0.07, 0.15],
        help="ensemble weights (must roughly sum to 1)",
    )
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="threshold for the sanity Acc/Prec/Rec/F1 readout")
    args = ap.parse_args()

    if not np.isclose(sum(args.w), 1.0, atol=1e-6):
        print(f"warning: weights sum to {sum(args.w):.6f}, not 1.0", file=sys.stderr)

    targets = ["baseline", "youtube"] if args.variant == "both" else [args.variant]
    for name in targets:
        run(name, *args.w, thr=args.threshold)


if __name__ == "__main__":
    main()
