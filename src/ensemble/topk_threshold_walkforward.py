"""
Walk-forward (expanding-window) OOS validation of the top-k + threshold linear
ensemble policy.

For test fold Y (folds are ordered years): select (w_xgb,w_lstm,w_cnn, k, a) using
ONLY folds < Y (expanding window), then apply that frozen policy to fold Y and
record out-of-sample trades / win rate / B_return. This removes the in-sample
optimism of fitting weights+k+a on the same folds we report.

Selection rule (on past folds): maximise pooled per-trade B_return, subject to
"active in every past fold with >= MIN_FOLD_N trades" (so we never pick a policy
that only trades in one lucky past year).

We precompute a per-(weight,k,a,fold) metric tensor once, so each walk-forward
step is just array aggregation.

Outputs (outputs/ensemble/):
  wf_oos_perfold_{variant}.csv   per test-fold OOS result + the policy chosen
  wf_oos_summary.csv             pooled OOS across folds, both variants

Usage:
  python topk_threshold_walkforward.py --weight-step 0.02
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

KSET = [10, 20, 30]
ASET = [0.0, 0.50, 0.55, 0.60, 0.65]
MIN_FOLD_N = 30        # each PAST fold must have >= this many trades for a policy to be eligible

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


def build(variant, ret):
    m = pd.read_parquet(OUT_DIR / f"ensemble_3model_{variant}.parquet")
    m["date"] = pd.to_datetime(m["date"]); m["ticker"] = m["ticker"].astype(str).str.zfill(6)
    m = m.merge(ret, on=["date", "ticker"], how="left")
    folds = sorted(m["test_year"].unique())
    F = {}
    for ty in folds:
        g = m[m["test_year"] == ty]
        F[ty] = (g[["p_xgb", "p_lstm", "p_cnn"]].to_numpy(np.float64),
                 g["label_binary"].to_numpy(),
                 g["B_return"].to_numpy(np.float64),
                 pd.factorize(g["date"].to_numpy())[0])
    return F, folds


def gen_weights(step):
    n = int(round(1.0 / step))
    return np.array([(i / n, j / n, (n - i - j) / n)
                     for i in range(n + 1) for j in range(n + 1 - i)], dtype=np.float64)


def within_day_rank(score, day_id):
    order = np.lexsort((-score, day_id)); ds = day_id[order]
    first = np.searchsorted(ds, ds, side="left")
    rank = np.empty(len(score), dtype=np.int64); rank[order] = np.arange(len(score)) - first
    return rank


def build_tensor(F, folds, W, kset, aset):
    """TEN[metric][wi,ki,ai,fi]; metrics: n, wins, retsum."""
    nW, nK, nA, nF = len(W), len(kset), len(aset), len(folds)
    N = np.zeros((nW, nK, nA, nF)); WIN = np.zeros_like(N); RS = np.zeros_like(N)
    t0 = time.time()
    for wi in range(nW):
        w = W[wi]
        for fi, ty in enumerate(folds):
            P, y, r, day = F[ty]
            score = P @ w
            rank = within_day_rank(score, day)
            for ki, k in enumerate(kset):
                topk = rank < k
                for ai, a in enumerate(aset):
                    sel = topk & (score >= a)
                    nf = int(sel.sum())
                    if nf == 0:
                        continue
                    N[wi, ki, ai, fi] = nf
                    WIN[wi, ki, ai, fi] = int(y[sel].sum())
                    RS[wi, ki, ai, fi] = float(r[sel].sum())
        if wi % 200 == 0:
            print(f"    tensor {wi}/{nW} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  tensor built in {time.time()-t0:.0f}s", flush=True)
    return N, WIN, RS


def walk_forward(variant, F, folds, W, kset, aset):
    N, WIN, RS = build_tensor(F, folds, W, kset, aset)
    nF = len(folds)
    rows = []
    for t in range(1, nF):                       # test fold index (need >=1 past fold)
        past = slice(0, t)
        n_past = N[:, :, :, past]                 # (W,K,A,t)
        active_all = (n_past > 0).all(axis=3)
        min_fold = np.where(n_past.min(axis=3) >= MIN_FOLD_N, True, False)
        feasible = active_all & min_fold
        tot_n = n_past.sum(axis=3)
        tot_rs = RS[:, :, :, past].sum(axis=3)
        with np.errstate(invalid="ignore", divide="ignore"):
            past_avg = np.where(tot_n > 0, tot_rs / tot_n, -np.inf)
        past_avg = np.where(feasible, past_avg, -np.inf)
        if not np.isfinite(past_avg).any():
            rows.append({"test_year": folds[t], "n": 0, "note": "no feasible policy on past"})
            continue
        wi, ki, ai = np.unravel_index(np.argmax(past_avg), past_avg.shape)
        # apply to test fold t (OOS)
        n = int(N[wi, ki, ai, t]); win = WIN[wi, ki, ai, t]; rs = RS[wi, ki, ai, t]
        P, y, r, _ = F[folds[t]]
        base = float(y.mean()); fold_mean_ret = float(r.mean())
        rows.append({
            "test_year": folds[t],
            "w_xgb": round(W[wi][0], 2), "w_lstm": round(W[wi][1], 2), "w_cnn": round(W[wi][2], 2),
            "k": kset[ki], "a": aset[ai],
            "n": n, "win%": round(win / n * 100, 1) if n else np.nan,
            "avg_ret%": round(rs / n * 100, 2) if n else np.nan,
            "base%": round(base * 100, 1), "rand_ret%": round(fold_mean_ret * 100, 2),
            "past_avg%": round(float(past_avg[wi, ki, ai]) * 100, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"wf_oos_perfold_{variant}.csv", index=False)
    print(f"\n[{variant}] WALK-FORWARD OOS (policy chosen on folds < Y, applied to Y):")
    print(df.to_string(index=False))
    # pooled OOS aggregate (trade-weighted over the OOS folds that traded)
    tr = df[df["n"] > 0]
    tot_n = int(tr["n"].sum())
    pooled_win = float((tr["win%"] / 100 * tr["n"]).sum() / tot_n)
    pooled_ret = float((tr["avg_ret%"] / 100 * tr["n"]).sum() / tot_n)
    ex20 = tr[tr["test_year"] != 2020]
    ex20_ret = float((ex20["avg_ret%"] / 100 * ex20["n"]).sum() / ex20["n"].sum()) if len(ex20) else np.nan
    pos = int((tr["avg_ret%"] > 0).sum())
    print(f"\n  OOS pooled: trades={tot_n:,} ({tot_n/len(tr):.0f}/fold)  win={pooled_win*100:.1f}%  "
          f"avg_ret={pooled_ret*100:.2f}%  ex2020={ex20_ret*100:.2f}%  pos_folds={pos}/{len(tr)}")
    return {"variant": variant, "oos_trades": tot_n, "oos_win%": round(pooled_win*100,1),
            "oos_avg_ret%": round(pooled_ret*100,2), "oos_ex2020%": round(ex20_ret*100,2),
            "pos_folds": pos, "n_oos_folds": len(tr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "youtube", "both"], default="both")
    ap.add_argument("--weight-step", type=float, default=0.02)
    args = ap.parse_args()
    ret = load_return_map()
    W = gen_weights(args.weight_step)
    print(f"weights={len(W):,} (step {args.weight_step})  k={KSET}  a={ASET}  MIN_FOLD_N={MIN_FOLD_N}")
    targets = ["baseline", "youtube"] if args.variant == "both" else [args.variant]
    summ = []
    for v in targets:
        F, folds = build(v, ret)
        summ.append(walk_forward(v, F, folds, W, KSET, ASET))
    s = pd.DataFrame(summ)
    s.to_csv(OUT_DIR / "wf_oos_summary.csv", index=False)
    print("\n" + "=" * 70 + "\n[WALK-FORWARD OOS SUMMARY]\n" + "=" * 70)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
