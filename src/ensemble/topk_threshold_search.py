"""
Joint search over linear-ensemble weights × top-k × score-threshold a.

For each (w_xgb, w_lstm, w_cnn) on the 0.0x simplex, daily rule:
    score = w·(p_xgb, p_lstm, p_cnn);  buy the top-k tickers/day with score >= a.
Realised outcome uses B_return (익절/손절 turtle return), label = B_return>0.

"Reasonable" objective (default): maximise the fold-equal-weighted mean per-trade
return (so a single bull year like 2020 can't dominate), subject to coverage
(trades in >= MIN_FOLDS of 8 folds) and a trades/year floor. We compute every
metric per config so the frontier can be re-ranked by win rate / Sharpe / count.

Outputs (outputs/ensemble/):
  ktha_sweep_{variant}.parquet     every (w,k,a) config × metrics
  ktha_frontier_{variant}.csv      best config per trades/yr tier (coverage>=MIN_FOLDS)
  ktha_reco_{variant}.csv          single recommended config + per-fold breakdown

Usage:
  python topk_threshold_search.py --variant both --weight-step 0.02
  python topk_threshold_search.py --variant youtube --weight-step 0.01 \
      --kset 10 --aset 0.0 0.50 0.55      # refine weights at fixed (k,a) region
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KSET = [3, 5, 7, 10, 15, 20, 30]
ASET = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65]          # thresholds on RAW prob score
ASET_RANK = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95]     # thresholds on RANK-percentile score
MIN_FOLD_N = 30        # every fold must have >= this many trades (real per-fold sample)
TPY_BANDS = [(20, 50), (50, 100), (100, 250), (250, 500), (500, 1500), (1500, 10**9)]
SUFFIX = ""            # set to "_rank" when --rank-norm (keeps outputs separate)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_return_map() -> pd.DataFrame:
    parts = []
    for f in glob.glob(str(ROOT / "lstm" / "data" / "processed" / "labels" / "*.parquet")):
        tk = os.path.basename(f).split(".")[0]
        d = pd.read_parquet(f, columns=["dt", "B_return"])
        d["ticker"] = tk
        parts.append(d)
    lab = pd.concat(parts, ignore_index=True)
    lab["date"] = pd.to_datetime(lab["dt"])
    lab["ticker"] = lab["ticker"].astype(str).str.zfill(6)
    return lab[["date", "ticker", "B_return"]].drop_duplicates(["date", "ticker"])


def build(variant: str, ret: pd.DataFrame, rank_norm: bool = False):
    m = pd.read_parquet(OUT_DIR / f"ensemble_3model_{variant}.parquet")
    m["date"] = pd.to_datetime(m["date"])
    m["ticker"] = m["ticker"].astype(str).str.zfill(6)
    m = m.merge(ret, on=["date", "ticker"], how="left")
    folds = sorted(m["test_year"].unique())
    F = {}
    for ty in folds:
        g = m[m["test_year"] == ty]
        P = g[["p_xgb", "p_lstm", "p_cnn"]].astype(np.float64)
        if rank_norm:
            # per-fold percentile [0,1] per model → scale-invariant before weighting
            P = P.rank(pct=True)
        F[ty] = (
            P.to_numpy(np.float64),
            g["label_binary"].to_numpy(),
            g["B_return"].to_numpy(np.float64),
            pd.factorize(g["date"].to_numpy())[0],
        )
    return F, folds


def gen_weights(step: float) -> np.ndarray:
    n = int(round(1.0 / step))
    return np.array([(i / n, j / n, (n - i - j) / n)
                     for i in range(n + 1) for j in range(n + 1 - i)], dtype=np.float64)


def within_day_rank(score: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    order = np.lexsort((-score, day_id))
    ds = day_id[order]
    first = np.searchsorted(ds, ds, side="left")
    rank = np.empty(len(score), dtype=np.int64)
    rank[order] = np.arange(len(score)) - first
    return rank


def run(variant: str, F, folds, W, kset, aset) -> pd.DataFrame:
    t0 = time.time()
    n_w = len(W)
    print(f"\n[{variant}] joint sweep: {n_w:,} weights × {len(kset)} k × {len(aset)} a "
          f"= {n_w*len(kset)*len(aset):,} configs", flush=True)
    nfold = len(folds)
    i2020 = folds.index(2020) if 2020 in folds else -1
    rows = []
    for wi in range(n_w):
        w = W[wi]
        # precompute per fold: score, rank
        per = []
        for ty in folds:
            P, y, r, day = F[ty]
            score = P @ w
            per.append((score, within_day_rank(score, day), y, r))
        for k in kset:
            for a in aset:
                fold_n = np.zeros(nfold); fold_rs = np.zeros(nfold)
                wins = pos_ret_sum = pos_n = 0
                for fi in range(nfold):
                    score, rank, y, r = per[fi]
                    sel = (rank < k) & (score >= a)
                    nf = int(sel.sum())
                    if nf == 0:
                        continue
                    ys = y[sel]; rs = r[sel]
                    fold_n[fi] = nf; fold_rs[fi] = float(rs.sum())
                    wins += int(ys.sum()); pr = rs[rs > 0]
                    pos_ret_sum += float(pr.sum()); pos_n += int(pr.size)
                n_tot = int(fold_n.sum())
                if n_tot == 0:
                    continue
                active = fold_n > 0
                fold_ret = np.where(active, fold_rs / np.where(fold_n == 0, 1, fold_n), np.nan)
                n_act = int(active.sum())
                min_fn = int(fold_n[active].min())
                pos_folds = int(np.nansum(fold_ret[active] > 0))
                worst = float(np.nanmin(fold_ret[active]))
                mean_fold = float(np.nanmean(fold_ret[active]))
                n_ex = n_tot - (fold_n[i2020] if i2020 >= 0 else 0)
                ret_ex = ((fold_rs.sum() - (fold_rs[i2020] if i2020 >= 0 else 0)) / n_ex * 100
                          if n_ex > 0 else np.nan)
                pct2020 = (fold_n[i2020] / n_tot * 100) if i2020 >= 0 else 0.0
                rows.append((
                    w[0], w[1], w[2], k, a,
                    n_tot, n_tot / 8.0, n_act, min_fn,
                    wins / n_tot * 100,                       # pooled win %
                    fold_rs.sum() / n_tot * 100,              # pooled avg ret %
                    (pos_ret_sum / pos_n * 100) if pos_n else np.nan,  # avg ret | win %
                    mean_fold * 100, worst * 100, pos_folds,  # robustness
                    ret_ex, pct2020,                          # 2020 dependence
                ))
        if wi % 200 == 0:
            print(f"    {wi}/{n_w}  ({time.time()-t0:.0f}s)", flush=True)
    cols = ["w_xgb", "w_lstm", "w_cnn", "k", "a", "n_total", "trades_yr", "active_folds",
            "min_fold_n", "winrate", "avg_ret", "avg_ret_win", "mean_fold_ret",
            "worst_fold_ret", "pos_folds", "ret_ex2020", "pct_2020"]
    df = pd.DataFrame(rows, columns=cols)
    print(f"  done: {len(df):,} non-empty configs in {time.time()-t0:.0f}s", flush=True)
    return df


def report(df: pd.DataFrame, variant: str):
    df.to_parquet(OUT_DIR / f"ktha_sweep{SUFFIX}_{variant}.parquet", index=False)
    nfold = int(df["active_folds"].max())
    # "reasonable" = trades in every fold AND >= MIN_FOLD_N in the weakest fold
    ok = df[(df["active_folds"] == nfold) & (df["min_fold_n"] >= MIN_FOLD_N)].copy()
    print(f"\n[{variant}] reasonable configs (all {nfold} folds, min_fold_n>={MIN_FOLD_N}): "
          f"{len(ok):,} / {len(df):,}")
    show = ["k", "a", "w_xgb", "w_lstm", "w_cnn", "trades_yr", "min_fold_n", "winrate",
            "avg_ret", "ret_ex2020", "worst_fold_ret", "pos_folds", "pct_2020"]
    if not len(ok):
        print("  (none meet the constraint — loosen MIN_FOLD_N)")
        return

    # frontier: per trades/yr band, the best pooled-return AND best robust(ex-2020) config
    fr_rows = []
    for lo, hi in TPY_BANDS:
        sub = ok[(ok["trades_yr"] >= lo) & (ok["trades_yr"] < hi)]
        if not len(sub):
            continue
        band = f"{lo}-{hi if hi < 10**8 else '∞'}"
        for crit in ("avg_ret", "ret_ex2020"):
            b = sub.loc[sub[crit].idxmax()].copy()
            b["band"] = band; b["by"] = crit; fr_rows.append(b)
    if fr_rows:
        fr = pd.DataFrame(fr_rows)
        print(f"\n[{variant}] frontier — best config per trades/yr band (by pooled return & by ex-2020 return):")
        print(fr[["band", "by"] + show].round(3).to_string(index=False))
        fr[["band", "by"] + show].round(5).to_csv(OUT_DIR / f"ktha_frontier{SUFFIX}_{variant}.csv", index=False)

    # primary: max pooled avg_ret among deployable (trades_yr>=50); robust: max return ex-2020
    cand = ok[ok["trades_yr"] >= 50]; cand = cand if len(cand) else ok
    reco = cand.loc[cand["avg_ret"].idxmax()]
    robust = ok.loc[ok["ret_ex2020"].idxmax()]
    for tag, r in [("★ MAX pooled return (trades/yr>=50)", reco),
                   ("◆ ROBUST: MAX return excluding 2020", robust)]:
        print(f"\n[{variant}] {tag}:")
        print(f"    weights=({r.w_xgb:.2f},{r.w_lstm:.2f},{r.w_cnn:.2f})  k={int(r.k)}  a={r.a:.2f}  "
              f"trades/yr={r.trades_yr:.0f}  min_fold_n={int(r.min_fold_n)}")
        print(f"    winrate={r.winrate:.1f}%  avg_ret={r.avg_ret:.2f}%  ret_ex2020={r.ret_ex2020:.2f}%  "
              f"worst_fold={r.worst_fold_ret:.2f}%  pos_folds={int(r.pos_folds)}/{nfold}  pct_2020={r.pct_2020:.0f}%")
    pd.DataFrame([reco, robust]).round(5).to_csv(OUT_DIR / f"ktha_reco{SUFFIX}_{variant}.csv", index=False)

    trap = df.loc[df["avg_ret"].idxmax()]
    print(f"\n[{variant}] (contrast) max avg_ret, NO constraints → trap: "
          f"k={int(trap.k)} a={trap.a:.2f} w=({trap.w_xgb:.2f},{trap.w_lstm:.2f},{trap.w_cnn:.2f})  "
          f"avg_ret={trap.avg_ret:.1f}%  trades/yr={trap.trades_yr:.1f}  min_fold_n={int(trap.min_fold_n)}  "
          f"pct_2020={trap.pct_2020:.0f}%")


def main():
    global SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "youtube", "both"], default="both")
    ap.add_argument("--weight-step", type=float, default=0.02)
    ap.add_argument("--kset", type=int, nargs="+", default=KSET)
    ap.add_argument("--aset", type=float, nargs="+", default=None)
    ap.add_argument("--rank-norm", action="store_true",
                    help="per-fold percentile-normalize each model before weighting (scale-invariant)")
    args = ap.parse_args()

    # threshold set default depends on mode (percentile thresholds for rank-norm)
    aset = args.aset if args.aset is not None else (ASET_RANK if args.rank_norm else ASET)
    if args.rank_norm:
        SUFFIX = "_rank"
    print(f"mode={'RANK-NORM (per-fold percentile)' if args.rank_norm else 'RAW prob'}  "
          f"k={args.kset}  a={aset}", flush=True)

    ret = load_return_map()
    W = gen_weights(args.weight_step)
    targets = ["baseline", "youtube"] if args.variant == "both" else [args.variant]
    for v in targets:
        F, folds = build(v, ret, rank_norm=args.rank_norm)
        df = run(v, F, folds, W, args.kset, aset)
        report(df, v)


if __name__ == "__main__":
    main()
