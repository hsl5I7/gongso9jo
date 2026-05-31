"""
permutation_selection.py
=========================

Permutation importance로 피처 importance 정밀 측정 → 상위 N개 선택 → 재학습.

전략:
- 학습된 모델의 예측 성능을 기준으로
- 각 피처를 무작위로 섞었을 때 성능 하락 측정
- 하락이 큰 피처 = 중요한 피처

작업 흐름:
1. 지정된 fold들에 대해 학습 + permutation 측정
2. Fold별 importance를 평균 (또는 median)
3. 상위 N개 선택
4. Selected 피처로 전체 9 fold 재학습
5. baseline vs selected 비교

사용법:
    # 전략 1: 최근 4 fold만 사용 (빠름, 약 20분)
    python permutation_selection.py --data_dir ..\output_v3 --ohlcv_dir ..\data\processed\ohlcv ^
        --output_dir ..\results\perm_sel --top_n 70 ^
        --perm_fold_years 2023,2024,2025,2026

    # 전략 2: 전체 fold (정통, 약 60분)
    python permutation_selection.py ... --perm_fold_years 2018,2019,2020,2021,2022,2023,2024,2025,2026
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_walkforward_v2 import (
    simulate_topk_with_slots, trade_stats, load_raw_ohlcv,
    run_one_fold, LABEL_UP,
)


# ============================================================
# 1. Permutation importance for 1 fold
# ============================================================

def permutation_importance_one_fold(
    df_X: pd.DataFrame,
    y_3class: pd.Series,
    meta: pd.DataFrame,
    raw_data: dict,
    test_year: int,
    cfg: dict,
    n_repeats: int = 3,
    rng_seed: int = 42,
) -> pd.Series:
    """
    한 fold에 대해 permutation importance 계산.

    측정 기준: Sharpe (단리). 피처 섞으면 Sharpe 얼마나 떨어지나.
    n_repeats: 같은 피처를 여러 번 섞어서 평균 (노이즈 줄이기)
    """
    train_start = pd.Timestamp(cfg["train_start"])
    valid_start = pd.Timestamp(f"{test_year - 1}-01-01")
    valid_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    dates = meta["date"]
    train_mask = (dates >= train_start) & (dates < valid_start)
    valid_mask = (dates >= valid_start) & (dates <= valid_end)
    test_mask = (dates >= test_start) & (dates <= test_end)

    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        return pd.Series(dtype=float)

    X_tr = df_X.loc[train_mask].values
    y_tr = y_3class.loc[train_mask].values.astype(int)
    X_va = df_X.loc[valid_mask].values
    y_va = y_3class.loc[valid_mask].values.astype(int)
    X_te = df_X.loc[test_mask].values

    n_classes = int(max(y_tr.max(), y_va.max() if len(y_va) > 0 else 0)) + 1
    params = dict(cfg["xgb_params"])
    params["num_class"] = n_classes

    print(f"  [fold {test_year}] 학습 중... (train={len(y_tr):,}, test={len(X_te):,})")
    t0 = time.time()
    model = xgb.XGBClassifier(**params)
    if len(y_va) > 0:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        params_no_es = dict(params)
        params_no_es.pop("early_stopping_rounds", None)
        model = xgb.XGBClassifier(**params_no_es)
        model.fit(X_tr, y_tr, verbose=False)
    print(f"  [fold {test_year}] 학습 완료 ({time.time()-t0:.0f}s)")

    # baseline Sharpe (피처 안 섞은 상태)
    def get_sharpe(X_input):
        proba = model.predict_proba(X_input)
        p_up_col = LABEL_UP if proba.shape[1] > LABEL_UP else proba.shape[1] - 1
        test_meta = meta.loc[test_mask].reset_index(drop=True)
        pred_df = pd.DataFrame({
            "date": test_meta["date"].values,
            "ticker": test_meta["ticker"].values,
            "p_up": proba[:, p_up_col],
        })
        trades, eq_s, _ = simulate_topk_with_slots(
            pred_df, raw_data,
            k=cfg["topk_k"], horizon=cfg["topk_horizon"],
            tp_pct=cfg["topk_tp_pct"], sl_pct=cfg["topk_sl_pct"],
            selection_mode="top",
            initial_capital=cfg["initial_capital"],
        )
        stats = trade_stats(trades, eq_s, cfg["initial_capital"])
        return stats["sharpe"] if not np.isnan(stats["sharpe"]) else 0.0

    baseline_sharpe = get_sharpe(X_te)
    print(f"  [fold {test_year}] baseline Sharpe: {baseline_sharpe:+.3f}")

    # 각 피처마다 permutation importance
    rng = np.random.default_rng(rng_seed + test_year)
    feature_names = df_X.columns.tolist()
    n_features = len(feature_names)
    importances = np.zeros(n_features)

    t_perm_start = time.time()
    for i, fname in enumerate(feature_names):
        repeats = []
        for r in range(n_repeats):
            X_te_shuffled = X_te.copy()
            shuffled_col = X_te_shuffled[:, i].copy()
            rng.shuffle(shuffled_col)
            X_te_shuffled[:, i] = shuffled_col

            shuffled_sharpe = get_sharpe(X_te_shuffled)
            # importance = baseline 대비 떨어진 정도
            repeats.append(baseline_sharpe - shuffled_sharpe)

        importances[i] = np.mean(repeats)

        # 진행 상황 (10개마다)
        if (i + 1) % 20 == 0 or i == n_features - 1:
            elapsed = time.time() - t_perm_start
            eta = elapsed / (i + 1) * (n_features - i - 1)
            print(f"    피처 {i+1}/{n_features} 완료 "
                  f"({elapsed:.0f}s, 남은 ETA: {eta:.0f}s)")

    return pd.Series(importances, index=feature_names, name=f"perm_imp_{test_year}")


# ============================================================
# 2. 메인
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ohlcv_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./perm_sel_results")
    p.add_argument("--top_n", type=int, default=70)
    p.add_argument("--perm_fold_years", type=str, default="2023,2024,2025,2026",
                   help="permutation importance 측정할 fold (콤마 구분)")
    p.add_argument("--retrain_test_years", type=str,
                   default="2018,2019,2020,2021,2022,2023,2024,2025,2026",
                   help="재학습 시 전체 fold")
    p.add_argument("--n_repeats", type=int, default=3,
                   help="피처당 permutation 반복 횟수")
    p.add_argument("--train_start", type=str, default="2008-01-01")
    p.add_argument("--topk_k", type=int, default=5)
    p.add_argument("--topk_horizon", type=int, default=30)
    p.add_argument("--tp_pct", type=float, default=0.15)
    p.add_argument("--sl_pct", type=float, default=0.035)
    p.add_argument("--initial_capital", type=float, default=10_000_000)
    p.add_argument("--skip_retrain", action="store_true",
                   help="재학습 스킵 (importance만 측정)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perm_fold_years = [int(y) for y in args.perm_fold_years.split(",")]
    retrain_test_years = [int(y) for y in args.retrain_test_years.split(",")]

    cfg = {
        "train_start": args.train_start,
        "topk_k": args.topk_k,
        "topk_horizon": args.topk_horizon,
        "topk_tp_pct": args.tp_pct,
        "topk_sl_pct": args.sl_pct,
        "initial_capital": args.initial_capital,
        "xgb_params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 30,
        },
    }

    t_start = time.time()

    # --- 1. 데이터 로드 ---
    print("=" * 70)
    print("데이터 로드")
    print("=" * 70)
    data_dir = Path(args.data_dir)
    df_full = pd.read_parquet(data_dir / "df_full.parquet")
    df_full["date"] = pd.to_datetime(df_full["date"])
    df_full = df_full.dropna(subset=["label_3class"]).copy()
    df_full["label_3class"] = df_full["label_3class"].astype(int)

    exclude = {
        "date", "ticker", "label_3class", "label_binary",
        "days_to_event", "realized_ret",
        "open", "high", "low", "close", "volume", "trade_value",
        "turnover_rt", "chg_sig", "chg", "tradeable",
        "shares_est", "mktcap_est",
        "trade_value_ma20", "turnover_rt_ma20",
        "stk_nm", "market", "sector",
        "year", "month", "_year",
    }
    feature_cols_all = [
        c for c in df_full.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_full[c])
    ]

    df_X = df_full[feature_cols_all].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y_3class = df_full["label_3class"].reset_index(drop=True)
    meta = df_full[["date", "ticker"]].reset_index(drop=True)
    print(f"  전체 피처: {len(feature_cols_all)}개")
    print(f"  Permutation 측정 fold: {perm_fold_years}")
    print(f"  반복 횟수: {args.n_repeats}")

    # 원본 OHLCV
    print("\n원본 OHLCV 로드...")
    tickers = df_full["ticker"].unique().tolist()
    raw_data = load_raw_ohlcv(Path(args.ohlcv_dir), tickers)
    print(f"  {len(raw_data)}/{len(tickers)}개 종목")

    # --- 2. Fold별 permutation importance ---
    print("\n" + "=" * 70)
    print(f"Permutation Importance 측정 ({len(perm_fold_years)} fold)")
    print("=" * 70)
    print(f"⚠ 예상 시간: ~{len(perm_fold_years) * 8}분 "
          f"(fold당 약 8분, 162 피처 × {args.n_repeats}회)")

    fold_imps = []
    for test_year in perm_fold_years:
        print(f"\n--- Fold {test_year} ---")
        imp = permutation_importance_one_fold(
            df_X, y_3class, meta, raw_data, test_year, cfg,
            n_repeats=args.n_repeats,
        )
        if len(imp) > 0:
            fold_imps.append(imp)

    # Fold 평균
    if not fold_imps:
        print("[ERROR] permutation importance 측정 실패")
        return

    perm_df = pd.concat(fold_imps, axis=1)
    perm_df["mean_importance"] = perm_df.mean(axis=1)
    perm_df["median_importance"] = perm_df.median(axis=1)
    perm_df["std_importance"] = perm_df.std(axis=1) if perm_df.shape[1] > 1 else 0
    perm_df = perm_df.sort_values("mean_importance", ascending=False)
    perm_df.to_csv(output_dir / "permutation_importance.csv")

    # --- 3. 상위 N개 선택 ---
    print("\n" + "=" * 70)
    print(f"상위 {args.top_n}개 피처 선택")
    print("=" * 70)

    selected = perm_df.head(args.top_n).index.tolist()
    pd.DataFrame({"feature": selected}).to_csv(
        output_dir / "selected_features.csv", index=False)

    # 카테고리
    cs_cnt = sum(1 for c in selected if c.endswith("_cs"))
    inter_cnt = sum(1 for c in selected if c.startswith("inter_"))
    csrank_cnt = sum(1 for c in selected if c.startswith("cs_rank_"))
    sector_cnt = sum(1 for c in selected if c.startswith("sector_"))
    print(f"\n선택된 피처 카테고리:")
    print(f"  cs (정규화)        : {cs_cnt}")
    print(f"  inter (상호작용)   : {inter_cnt}")
    print(f"  cs_rank            : {csrank_cnt}")
    print(f"  sector_            : {sector_cnt}")
    print(f"  나머지 (원본)      : {len(selected) - cs_cnt - inter_cnt - csrank_cnt - sector_cnt}")

    print(f"\nTop 20 (permutation importance 기준):")
    for i, c in enumerate(selected[:20], 1):
        imp = perm_df.loc[c, "mean_importance"]
        imp_std = perm_df.loc[c, "std_importance"]
        print(f"  {i:>2}. {c:<35} {imp:+.4f} ± {imp_std:.4f}")

    print(f"\n잘려나간 하위 10개 (importance 낮은 거):")
    bottom = perm_df.tail(10).index.tolist()
    for c in bottom:
        imp = perm_df.loc[c, "mean_importance"]
        print(f"  {c:<35} {imp:+.4f}")

    # 음수 importance (피처 빼는 게 더 나음)
    negative = perm_df[perm_df["mean_importance"] < 0]
    if len(negative) > 0:
        print(f"\n⚠ Importance가 음수인 피처 ({len(negative)}개):")
        print(f"  (이 피처들은 노이즈 추가 — 빼는 게 모델에 도움)")
        for c in negative.index[:10]:
            print(f"  {c:<35} {negative.loc[c, 'mean_importance']:+.4f}")

    # --- 4. Selected로 재학습 ---
    if args.skip_retrain:
        print("\n[skip_retrain] 재학습 생략")
        print(f"\n총 시간: {time.time()-t_start:.1f}s")
        return

    print("\n" + "=" * 70)
    print(f"Selected {args.top_n}개 피처로 전체 {len(retrain_test_years)} fold 재학습")
    print("=" * 70)

    cfg["test_years"] = retrain_test_years
    cfg["n_folds"] = len(retrain_test_years)
    df_X_sel = df_X[selected]

    fold_records = []
    running_cap_s = cfg["initial_capital"]
    combined_eq_s = []

    for fold_idx, test_year in enumerate(retrain_test_years):
        t_fold = time.time()
        result = run_one_fold(
            df_X_sel, y_3class, meta, raw_data, test_year, fold_idx, cfg
        )
        if result is None:
            continue

        top_s = result["sim_results"]["top"]["stats_simple"]
        top_c = result["sim_results"]["top"]["stats_compound"]
        rand_s = result["sim_results"]["random"]["stats_simple"]

        fold_records.append({
            "test_year": test_year,
            "n_trades": top_s["n_trades"],
            "win_rate": top_s["win_rate"],
            "cum_ret_simple": top_s["cum_return"],
            "sharpe_simple": top_s["sharpe"],
            "mdd_simple": top_s["mdd"],
            "cum_ret_compound": top_c["cum_return"],
            "sharpe_compound": top_c["sharpe"],
            "random_cum_ret": rand_s["cum_return"],
        })

        print(f"  [fold {fold_idx+1}/{cfg['n_folds']}] test={test_year}  "
              f"n={top_s['n_trades']:>4}  "
              f"cum_s={top_s['cum_return']*100:+.2f}%  "
              f"sharpe_s={top_s['sharpe']:+.3f}  "
              f"mdd_s={top_s['mdd']*100:+.2f}%  "
              f"({time.time()-t_fold:.0f}s)")

        eq_s = result["sim_results"]["top"]["equity_simple"]
        if len(eq_s) > 0:
            relative = eq_s / cfg["initial_capital"]
            scaled = running_cap_s * relative
            combined_eq_s.append(scaled)
            running_cap_s = scaled.iloc[-1]

    # 결과 요약
    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(output_dir / "fold_results_selected.csv", index=False)

    combined_s = pd.concat(combined_eq_s).sort_index() if combined_eq_s else pd.Series()
    if len(combined_s) > 0:
        combined_s = combined_s[~combined_s.index.duplicated(keep="last")]
        combined_s.to_csv(output_dir / "equity_simple_selected.csv")

    print("\n" + "=" * 70)
    print(f"결과 (Selected {len(selected)}개 피처)")
    print("=" * 70)
    print(f"  Mean Sharpe (단리): {fold_df['sharpe_simple'].mean():+.3f}")
    print(f"  Mean cum_ret/yr   : {fold_df['cum_ret_simple'].mean()*100:+.2f}%")
    print(f"  Mean Sharpe (복리): {fold_df['sharpe_compound'].mean():+.3f}")
    print(f"  Mean cum_ret/yr (복리): {fold_df['cum_ret_compound'].mean()*100:+.2f}%")
    print(f"  Positive years    : {(fold_df['cum_ret_simple'] > 0).sum()}/{len(fold_df)}")

    if len(combined_s) > 0:
        cum_total = combined_s.iloc[-1] / cfg["initial_capital"] - 1
        ret = combined_s.pct_change().dropna()
        sh_total = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        mdd = ((combined_s - combined_s.cummax()) / combined_s.cummax()).min()
        print(f"\n  전체 단리 누적: {cum_total*100:+.2f}%")
        print(f"  전체 Sharpe   : {sh_total:+.3f}")
        print(f"  전체 MDD      : {mdd*100:+.2f}%")

    # 비교 (기존 baseline은 이전 결과 사용)
    print(f"\n기존 baseline (162개)과 비교:")
    print(f"  ──────────────────────────────────────────────")
    print(f"  Baseline (162개) : Sharpe +1.477, cum +16.40%, MDD -28.51% (이전 결과)")
    print(f"  Selected ({len(selected)}개) : "
          f"Sharpe {fold_df['sharpe_simple'].mean():+.3f}, "
          f"cum {fold_df['cum_ret_simple'].mean()*100:+.2f}%")

    # 저장: selected만으로 df_full
    keep_cols = ["date", "ticker", "label_3class", "label_binary",
                 "days_to_event", "realized_ret",
                 "open", "high", "low", "close", "volume"] + selected
    keep_cols = [c for c in keep_cols if c in df_full.columns]
    df_full[keep_cols].to_parquet(output_dir / "df_full.parquet", index=False)
    df_X_sel.to_parquet(output_dir / "X.parquet", index=False)
    pd.DataFrame({"label_binary": (y_3class == 2).astype(int)}).to_parquet(
        output_dir / "y.parquet", index=False)
    meta.to_parquet(output_dir / "meta.parquet", index=False)

    summary = {
        "method": "permutation_importance",
        "n_selected": len(selected),
        "n_total": len(feature_cols_all),
        "perm_fold_years": perm_fold_years,
        "n_repeats": args.n_repeats,
        "mean_sharpe_simple": float(fold_df["sharpe_simple"].mean()),
        "mean_cum_ret_simple": float(fold_df["cum_ret_simple"].mean()),
        "mean_sharpe_compound": float(fold_df["sharpe_compound"].mean()),
        "positive_years": int((fold_df["cum_ret_simple"] > 0).sum()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n저장: {output_dir}")
    print(f"총 시간: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}분)")


if __name__ == "__main__":
    main()
