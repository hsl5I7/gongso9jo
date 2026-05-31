"""
Leak-free RANK-NORM walk-forward OOS validation.

The in-sample rank-norm search normalised each model per *fold* (uses the whole
year's distribution → mild look-ahead). Here we make it honest:

For test fold Y:
  1. Fit each model's empirical CDF on the pooled RAW predictions of folds < Y only.
  2. Map every prediction (past folds + fold Y) to its percentile under that
     past CDF  → scale-invariant score, NO future info.
  3. Select (w_xgb,w_lstm,w_cnn, k, a) on the past folds (max pooled B_return,
     subject to "active in every past fold with >= MIN_FOLD_N trades").
  4. Freeze that policy, apply to fold Y, record OOS trades / win / B_return.

So both the normalisation AND the policy are fit only on the past. This is the
true deployable performance of the rank-norm ensemble.

Output: outputs/ensemble/wf_oos_rank_perfold_{variant}.csv, wf_oos_rank_summary.csv

Usage:  python topk_threshold_walkforward_rank.py --weight-step 0.02
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
ASET = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95]   # percentile thresholds
MIN_FOLD_N = 30

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_return_map():
    parts = []
    for f in glob.glob(str(ROOT / "lstm" / "data" / "processed" / "labels" / "*.parquet")):
        tk = os.path.basename(f).split(".")[0]
        d = pd.read_parquet(f, columns=["dt", "B_return"]); d["ticker"] = tk
        parts.append(d)
    lab = pd.concat(parts, ignore_index=True)
    lab["date"] = pd.to_datetime(lab["dt"]); lab["ticker"] = lab["ticker"].astype(str).str.zfill(6)
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


def fold_metrics(Pn, y, r, day, W, kset, aset):
    """Per-(weight,k,a) metrics for ONE fold. Returns N, WIN, RS arrays (nW,nK,nA)."""
    nW, nK, nA = len(W), len(kset), len(aset)
    N = np.zeros((nW, nK, nA)); WIN = np.zeros_like(N); RS = np.zeros_like(N)
    for wi in range(nW):
        score = Pn @ W[wi]
        rank = within_day_rank(score, day)
        for ki, k in enumerate(kset):
            topk = rank < k
            for ai, a in enumerate(aset):
                sel = topk & (score >= a)
                nf = int(sel.sum())
                if nf:
                    N[wi, ki, ai] = nf; WIN[wi, ki, ai] = y[sel].sum(); RS[wi, ki, ai] = r[sel].sum()
    return N, WIN, RS


def run(variant, F, folds, W, kset, aset):
    t0 = time.time()
    nF = len(folds)
    rows = []
    for t in range(1, nF):
        past = folds[:t]; testy = folds[t]
        # past CDF per model (pooled past raw predictions)
        sorted_past = [np.sort(np.concatenate([F[ty][0][:, c] for ty in past])) for c in range(3)]
        def norm(ty):
            P = F[ty][0]
            return np.column_stack([np.searchsorted(sorted_past[c], P[:, c], side="right") / len(sorted_past[c])
                                    for c in range(3)])
        # accumulate past metrics
        Npast = np.zeros((len(W), len(kset), len(aset))); RSpast = np.zeros_like(Npast)
        active = np.ones_like(Npast, dtype=bool); minfold = np.full_like(Npast, np.inf)
        for ty in past:
            N, _, RS = fold_metrics(norm(ty), F[ty][1], F[ty][2], F[ty][3], W, kset, aset)
            Npast += N; RSpast += RS
            active &= (N > 0); minfold = np.minimum(minfold, N)
        feasible = active & (minfold >= MIN_FOLD_N)
        with np.errstate(invalid="ignore", divide="ignore"):
            past_avg = np.where((Npast > 0) & feasible, RSpast / np.where(Npast > 0, Npast, 1), -np.inf)
        if not np.isfinite(past_avg).any():
            rows.append({"test_year": testy, "n": 0, "note": "no feasible policy"}); continue
        wi, ki, ai = np.unravel_index(np.argmax(past_avg), past_avg.shape)
        # apply frozen policy to test fold (normalised by PAST cdf)
        Nt, WINt, RSt = fold_metrics(norm(testy), F[testy][1], F[testy][2], F[testy][3],
                                     W[wi:wi+1], [kset[ki]], [aset[ai]])
        n = int(Nt[0, 0, 0]); win = WINt[0, 0, 0]; rs = RSt[0, 0, 0]
        y, r = F[testy][1], F[testy][2]
        rows.append({
            "test_year": testy, "w_xgb": round(W[wi][0], 2), "w_lstm": round(W[wi][1], 2),
            "w_cnn": round(W[wi][2], 2), "k": kset[ki], "a": aset[ai],
            "n": n, "win%": round(win / n * 100, 1) if n else np.nan,
            "avg_ret%": round(rs / n * 100, 2) if n else np.nan,
            "base%": round(float(y.mean()) * 100, 1), "rand_ret%": round(float(r.mean()) * 100, 2),
            "past_avg%": round(float(past_avg[wi, ki, ai]) * 100, 2),
        })
        print(f"  {testy}: chose w=({W[wi][0]:.2f},{W[wi][1]:.2f},{W[wi][2]:.2f}) k={kset[ki]} a={aset[ai]} "
              f"→ n={n} win={rows[-1]['win%']}% ret={rows[-1]['avg_ret%']}%  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"wf_oos_rank_perfold_{variant}.csv", index=False)
    print(f"\n[{variant}] RANK-NORM WALK-FORWARD OOS (leak-free):")
    print(df.to_string(index=False))
    tr = df[df["n"] > 0]
    tot = int(tr["n"].sum())
    pw = float((tr["win%"] / 100 * tr["n"]).sum() / tot)
    pr = float((tr["avg_ret%"] / 100 * tr["n"]).sum() / tot)
    ex = tr[tr["test_year"] != 2020]
    exr = float((ex["avg_ret%"] / 100 * ex["n"]).sum() / ex["n"].sum()) if len(ex) else np.nan
    pos = int((tr["avg_ret%"] > 0).sum())
    print(f"\n  OOS pooled: trades={tot:,} ({tot/len(tr):.0f}/fold)  win={pw*100:.1f}%  "
          f"avg_ret={pr*100:.2f}%  ex2020={exr*100:.2f}%  pos_folds={pos}/{len(tr)}")
    return {"variant": variant, "oos_trades": tot, "oos_win%": round(pw*100, 1),
            "oos_avg_ret%": round(pr*100, 2), "oos_ex2020%": round(exr*100, 2),
            "pos_folds": pos, "n_oos_folds": len(tr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "youtube", "both"], default="both")
    ap.add_argument("--weight-step", type=float, default=0.02)
    args = ap.parse_args()
    ret = load_return_map(); W = gen_weights(args.weight_step)
    print(f"RANK-NORM leak-free WF | weights={len(W):,} k={KSET} a={ASET} MIN_FOLD_N={MIN_FOLD_N}")
    targets = ["baseline", "youtube"] if args.variant == "both" else [args.variant]
    summ = []
    for v in targets:
        F, folds = build(v, ret)
        summ.append(run(v, F, folds, W, KSET, ASET))
    s = pd.DataFrame(summ); s.to_csv(OUT_DIR / "wf_oos_rank_summary.csv", index=False)
    print("\n" + "=" * 70 + "\n[RANK-NORM WALK-FORWARD OOS SUMMARY]\n" + "=" * 70)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
