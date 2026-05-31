"""
Top-K-per-day ensemble weight search (linear, XGB + LSTM + CNN).

Objective is *trading selection quality*, not AUC: on every trading day pick the
K highest-scoring tickers (K=10 default) by the linear ensemble score
    s = w_xgb * p_xgb + w_lstm * p_lstm + w_cnn * p_cnn
and maximise the win rate (mean label_binary == "익절") of those picks.

Search space: the 3-simplex {w_xgb + w_lstm + w_cnn = 1, w ∈ {0, step, ..., 1}}.
For each weight we evaluate the top-K/day win rate per fold (test_year), then:
  - mean-fold-best  : argmax of the mean across folds (deployable single weight)
  - pooled-best     : argmax of the win rate over all selected trades pooled
  - walk-forward OOS : for fold Y, pick the weight that was best on folds < Y
                       (expanding window), apply to Y → honest out-of-sample win rate

Reference rows: random (= base rate), XGB-alone, equal 1/3, and the AUC-optimal
weights, all scored with the same top-K/day rule.

Inputs come from ensemble_3model loaders (already re-based to xgboost/), so the
merged universe is identical to the headline ensemble.

Outputs (outputs/ensemble/):
  topk{K}_sweep_{variant}.parquet     long: (test_year, w_xgb, w_lstm, w_cnn, winrate, n_sel)
  topk{K}_best_{variant}.csv          per-fold win rate at the mean-fold-best weight
  topk{K}_weights_summary.csv         one row per variant: best weights + win rates

Usage:
  python topk_weight_search.py                 # both variants, K=10, step=0.01
  python topk_weight_search.py --k 10 --step 0.01 --variant youtube
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_3model import VARIANTS, load_cnn, load_lstm, load_xgb

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# AUC-optimal weights from optimal_weight_search (for reference comparison)
AUC_OPT = {"baseline": (0.82, 0.16, 0.02), "youtube": (0.88, 0.11, 0.01)}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def build_merged(variant: str) -> pd.DataFrame:
    v = VARIANTS[variant]
    xgb = load_xgb(v).rename(columns={"label_binary": "lbl", "test_year": "ty"})
    lstm = load_lstm(v).drop(columns=["label_binary", "test_year"])
    cnn = load_cnn(v).drop(columns=["label_binary", "test_year"])
    m = xgb.merge(lstm, on=["date", "ticker"], how="inner")
    m = m.merge(cnn, on=["date", "ticker"], how="inner")
    m = m.rename(columns={"lbl": "label_binary", "ty": "test_year"})
    m = m[m["label_binary"].notna()].copy()
    m["label_binary"] = m["label_binary"].astype(int)
    return m[["date", "ticker", "test_year", "label_binary", "p_xgb", "p_lstm", "p_cnn"]]


def gen_weights(step: float) -> np.ndarray:
    n = int(round(1.0 / step))
    rows = [(i / n, j / n, (n - i - j) / n)
            for i in range(n + 1) for j in range(n + 1 - i)]
    return np.array(rows, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Top-K/day win rate                                                          #
# --------------------------------------------------------------------------- #
def topk_select_mask(score: np.ndarray, day_id: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the top-k rows by score within each day group."""
    order = np.lexsort((-score, day_id))            # day asc, score desc within day
    ds = day_id[order]
    first = np.searchsorted(ds, ds, side="left")    # first index of each day in sorted order
    within = np.arange(len(ds)) - first             # 0-based rank inside the day
    sel_sorted = within < k
    mask = np.zeros(len(score), dtype=bool)
    mask[order[sel_sorted]] = True
    return mask


def winrate(P: np.ndarray, y: np.ndarray, day_id: np.ndarray, w: np.ndarray, k: int) -> tuple[float, int]:
    mask = topk_select_mask(P @ w, day_id, k)
    n_sel = int(mask.sum())
    return (float(y[mask].mean()) if n_sel else float("nan")), n_sel


# --------------------------------------------------------------------------- #
# Run one variant                                                             #
# --------------------------------------------------------------------------- #
def run(variant: str, k: int, step: float) -> dict:
    t0 = time.time()
    m = build_merged(variant)
    folds = sorted(m["test_year"].unique())
    base_rate = float(m["label_binary"].mean())

    # per-fold numpy bundles (day_id factorized within fold)
    F = {}
    for ty in folds:
        g = m[m["test_year"] == ty]
        F[ty] = (
            g[["p_xgb", "p_lstm", "p_cnn"]].to_numpy(np.float64),
            g["label_binary"].to_numpy(),
            pd.factorize(g["date"].to_numpy())[0],
        )

    W = gen_weights(step)
    n_combos = len(W)
    print(f"\n{'=' * 78}\n{variant.upper()}  (n={len(m):,}, days={m['date'].nunique():,}, "
          f"folds {folds[0]}-{folds[-1]}, base_rate={base_rate:.4f})")
    print(f"  top-{k}/day win-rate sweep: {n_combos:,} weights × {len(folds)} folds")

    # win-rate matrix [fold, combo]  +  selected-count matrix
    wr = np.empty((len(folds), n_combos))
    nsel = np.empty((len(folds), n_combos), dtype=np.int64)
    for fi, ty in enumerate(folds):
        P, y, day_id = F[ty]
        for ci in range(n_combos):
            mask = topk_select_mask(P @ W[ci], day_id, k)
            nsel[fi, ci] = mask.sum()
            wr[fi, ci] = y[mask].mean()
        print(f"    fold {ty}: done ({time.time() - t0:.0f}s)")

    mean_fold = wr.mean(axis=0)
    # pooled win rate per combo: total wins / total selected across folds
    wins = (wr * nsel).sum(axis=0)
    pooled = wins / nsel.sum(axis=0)

    i_mf = int(np.argmax(mean_fold))
    i_pl = int(np.argmax(pooled))

    def ref(w):
        ws = np.array(w, dtype=np.float64)
        per = np.array([winrate(*F[ty], ws, k)[0] for ty in folds])
        return per.mean()

    ref_xgb = ref((1, 0, 0))
    ref_eq = ref((1 / 3, 1 / 3, 1 / 3))
    ref_auc = ref(AUC_OPT[variant])

    # walk-forward: fold Y uses best (mean win rate) weight over folds < Y
    wf = []
    for fi, ty in enumerate(folds):
        if fi == 0:
            w = np.array([1 / 3, 1 / 3, 1 / 3]); crit = "equal(no prior)"
        else:
            bi = int(np.argmax(wr[:fi].mean(axis=0))); w = W[bi]
            crit = f"({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})"
        oos, _ = winrate(*F[ty], w, k)
        wf.append((ty, crit, oos))
    wf_mean = float(np.mean([o for _, _, o in wf]))
    wf_mean_excl = float(np.mean([o for (ty, _, o) in wf if ty != folds[0]]))

    mf_w = W[i_mf]
    print(f"\n  [reference top-{k}/day win rate, mean over folds]")
    print(f"     random (base)      = {base_rate:.4f}")
    print(f"     XGB-alone          = {ref_xgb:.4f}")
    print(f"     equal 1/3          = {ref_eq:.4f}")
    print(f"     AUC-opt {AUC_OPT[variant]} = {ref_auc:.4f}")
    print(f"  [top-{k} mean-fold-best] w=({mf_w[0]:.2f}, {mf_w[1]:.2f}, {mf_w[2]:.2f})  "
          f"mean-fold winrate={mean_fold[i_mf]:.4f}")
    print(f"  [top-{k} pooled-best]    w=({W[i_pl][0]:.2f}, {W[i_pl][1]:.2f}, {W[i_pl][2]:.2f})  "
          f"pooled winrate={pooled[i_pl]:.4f}")
    print(f"  [walk-forward OOS]       mean winrate={wf_mean:.4f}  (excl first={wf_mean_excl:.4f})")

    print(f"\n  per-fold win rate @ mean-fold-best ({mf_w[0]:.2f}, {mf_w[1]:.2f}, {mf_w[2]:.2f}):")
    per_rows = []
    for ty in folds:
        w_, n_ = winrate(*F[ty], mf_w, k)
        print(f"     {ty}: winrate={w_:.4f}  (trades={n_:,}, base={F[ty][1].mean():.3f})")
        per_rows.append({"test_year": ty, "winrate": w_, "trades": n_, "base_rate": float(F[ty][1].mean())})
    print(f"  walk-forward selected weights per fold:")
    for ty, crit, oos in wf:
        print(f"     {ty}: sel={crit:18s} OOS winrate={oos:.4f}")

    # save
    sweep = pd.DataFrame({
        "test_year": np.repeat(folds, n_combos),
        "w_xgb": np.tile(W[:, 0], len(folds)),
        "w_lstm": np.tile(W[:, 1], len(folds)),
        "w_cnn": np.tile(W[:, 2], len(folds)),
        "winrate": wr.reshape(-1),
        "n_sel": nsel.reshape(-1),
    })
    sweep.to_parquet(OUT_DIR / f"topk{k}_sweep_{variant}.parquet", index=False)
    pd.DataFrame(per_rows).round(6).to_csv(OUT_DIR / f"topk{k}_best_{variant}.csv", index=False)

    return {
        "variant": variant, "k": k, "base_rate": base_rate,
        "mf_w_xgb": mf_w[0], "mf_w_lstm": mf_w[1], "mf_w_cnn": mf_w[2],
        "mf_winrate": mean_fold[i_mf],
        "pl_w_xgb": W[i_pl][0], "pl_w_lstm": W[i_pl][1], "pl_w_cnn": W[i_pl][2],
        "pl_winrate": pooled[i_pl],
        "walkforward_oos_winrate": wf_mean,
        "ref_random": base_rate, "ref_xgb": ref_xgb, "ref_equal": ref_eq, "ref_auc_opt": ref_auc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--variant", choices=["baseline", "youtube", "both"], default="both")
    args = ap.parse_args()
    targets = ["baseline", "youtube"] if args.variant == "both" else [args.variant]
    rows = [run(v, args.k, args.step) for v in targets]
    out = pd.DataFrame(rows)
    path = OUT_DIR / f"topk{args.k}_weights_summary.csv"
    out.round(6).to_csv(path, index=False)
    print(f"\n→ saved: {path}")


if __name__ == "__main__":
    main()
