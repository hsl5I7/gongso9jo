"""
train_walkforward_v4.py
========================

Walk-forward 학습 + binary classification (B_outcome 기반).
v2 대비 변경:
  1. 3-class → binary (label_binary)
  2. XGBoost objective: multi:softprob → binary:logistic
  3. 평가 메트릭: AUC, AP, Precision@k 추가
  4. 슬롯 시뮬: 라벨의 B_return 직접 사용 (도달 여부가 아닌 실제 수익률)

사용법:
    python train_walkforward_v4.py \
        --data_dir ..\\output_v4_features \
        --ohlcv_dir ..\\data\\processed\\ohlcv \
        --output_dir ..\\results\\wf_v4_baseline
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, precision_score,
)


# ============================================================
# 슬롯 시뮬 (v4: 라벨 기반 직접 사용)
# ============================================================

def simulate_threshold_v4(pred_df, label_df, threshold=0.5,
                          mode='top', initial_capital=10_000_000):
    """
    Threshold 방식: 확률 > threshold 인 모든 신호 매수.
    각 거래를 등가중으로 보고 거래 단위의 평균 수익률을 측정.
    (자본 동학 시뮬 안 함 — 순수 alpha 측정)

    Args:
        pred_df: columns=[date, ticker, p_pos]
        label_df: columns=[date, ticker, B_return, B_holding_days, B_outcome]
        threshold: 매수 임계값
        mode: 'top'/'bottom'/'random'

    Returns:
        trades_df: 모든 거래 기록
        equity_simple: (legacy 호환) 거래별 누적 단리
        equity_compound: (legacy 호환) 거래별 누적 복리
    """
    label_lookup = label_df.set_index(['date', 'ticker'])

    trades = []
    pred_df = pred_df.sort_values('date').reset_index(drop=True)
    daily_groups = pred_df.groupby('date')
    rng = np.random.RandomState(42)

    for date, day_df in daily_groups:
        if mode == 'top':
            selected = day_df[day_df['p_pos'] > threshold]
        elif mode == 'bottom':
            selected = day_df[day_df['p_pos'] < (1 - threshold)]
        elif mode == 'random':
            # TOP과 같은 개수만큼 무작위 선택
            n_top = (day_df['p_pos'] > threshold).sum()
            if n_top > 0:
                selected = day_df.sample(n=min(n_top, len(day_df)),
                                         random_state=rng.randint(0, 99999))
            else:
                selected = day_df.iloc[0:0]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for _, row in selected.iterrows():
            try:
                lbl = label_lookup.loc[(row['date'], row['ticker'])]
                ret = lbl['B_return']
                if pd.notna(ret):
                    trades.append({
                        'date': date,
                        'ticker': row['ticker'],
                        'p_pos': row['p_pos'],
                        'return': float(ret),
                        'outcome': lbl.get('B_outcome', pd.NA),
                        'holding_days': lbl.get('B_holding_days', pd.NA),
                    })
            except KeyError:
                pass

    trades_df = pd.DataFrame(trades)

    # legacy 호환: 거래별 등가중 누적
    # 단리: 각 거래의 수익률을 단순 합산
    # 복리: 각 거래 (1+r)을 곱해서 누적
    if len(trades_df) > 0:
        # 등가중: 자본을 trades 개수로 나눈 후 각 거래 수익률 적용
        n = len(trades_df)
        equity_simple = np.cumsum(np.concatenate([[initial_capital],
                                                   initial_capital / n * trades_df['return'].values]))
        # 복리: 같은 비중으로 등가중 포트폴리오 (전체 평균 수익률을 자본 1회 적용)
        avg_ret = trades_df['return'].mean()
        equity_compound = np.array([initial_capital,
                                     initial_capital * (1 + avg_ret)])
    else:
        equity_simple = np.array([initial_capital])
        equity_compound = np.array([initial_capital])

    return trades_df, equity_simple, equity_compound


def trade_stats(trades_df, equity, initial_capital, periods_per_year=252):
    """거래 단위 통계 (자본 시뮬 X — 순수 alpha 측정)"""
    if len(trades_df) == 0:
        return {'n_trades': 0, 'win_rate': 0, 'avg_return': 0,
                'total_return_pct': 0, 'sharpe': 0, 'mdd_pct': 0,
                'median_return': 0, 'std_return': 0}

    rets = trades_df['return'].values
    n_trades = len(rets)
    win_rate = (rets > 0).mean() * 100
    avg_return = rets.mean() * 100
    median_return = np.median(rets) * 100
    std_return = rets.std() * 100

    # 등가중 포트폴리오: 평균 수익률이 곧 total return
    total_return_pct = avg_return

    # 거래 단위 Sharpe (per trade × sqrt(n_trades))
    if std_return > 0:
        sharpe = (avg_return / std_return) * np.sqrt(n_trades)
    else:
        sharpe = 0

    # MDD: 시간 순 누적 수익률에서 계산
    if 'date' in trades_df.columns:
        sorted_rets = trades_df.sort_values('date')['return'].values
    else:
        sorted_rets = rets
    cum = np.cumprod(1 + sorted_rets)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    mdd_pct = drawdown.min() * 100 if len(drawdown) > 0 else 0

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'median_return': median_return,
        'std_return': std_return,
        'total_return_pct': total_return_pct,
        'sharpe': sharpe,
        'mdd_pct': mdd_pct,
    }


def precision_at_k(y_true, y_score, k_frac=0.1):
    """상위 k% 예측의 정확도"""
    n = len(y_true)
    k = max(1, int(n * k_frac))
    top_idx = np.argsort(y_score)[-k:]
    return y_true[top_idx].mean()


def find_best_threshold(pred_df, label_df, candidate_thresholds=None,
                        initial_capital=10_000_000):
    """
    Validation 데이터에서 threshold별 Sharpe 평가 → 최적값 반환.
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.arange(0.30, 0.91, 0.05)

    results = []
    for t in candidate_thresholds:
        trades, _, eq_c = simulate_threshold_v4(
            pred_df, label_df,
            threshold=t, mode='top',
            initial_capital=initial_capital,
        )
        stats = trade_stats(trades, eq_c, initial_capital)
        results.append({
            'threshold': t,
            'n_trades': stats['n_trades'],
            'sharpe': stats['sharpe'],
            'total_ret': stats['total_return_pct'],
            'win_rate': stats['win_rate'],
        })

    # 거래 수 너무 적은 거 제외 (최소 30개 거래)
    valid_results = [r for r in results if r['n_trades'] >= 30]
    if len(valid_results) == 0:
        valid_results = results  # 거래 부족하면 그냥 전체에서 선택

    # Sharpe 최대
    best = max(valid_results, key=lambda x: x['sharpe'])
    return best['threshold'], results


# ============================================================
# Walk-forward
# ============================================================

def make_fold_masks(meta, test_year, val_months=12, purge_days=60):
    """v3와 동일: train은 test 시작 - purge 이전, val은 train 마지막 1년"""
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    val_start = test_start - pd.Timedelta(days=purge_days + val_months * 30)
    val_end = test_start - pd.Timedelta(days=purge_days)
    train_end = val_start - pd.Timedelta(days=1)

    test_mask = (meta['date'] >= test_start) & (meta['date'] <= test_end)
    valid_mask = (meta['date'] >= val_start) & (meta['date'] <= val_end)
    train_mask = (meta['date'] <= train_end)

    return train_mask, valid_mask, test_mask


def train_one_fold(X, y, meta, test_year, cfg, label_df):
    """단일 fold 학습 + 평가"""
    train_mask, valid_mask, test_mask = make_fold_masks(
        meta, test_year,
        val_months=12,
        purge_days=cfg['purge_days'],
    )

    # 라벨 NaN 제외
    label_valid = ~y['label_binary'].isna()
    train_mask = train_mask & label_valid
    valid_mask = valid_mask & label_valid
    test_mask = test_mask & label_valid

    X_tr = X.loc[train_mask].values
    y_tr = y.loc[train_mask, 'label_binary'].astype(int).values
    X_va = X.loc[valid_mask].values
    y_va = y.loc[valid_mask, 'label_binary'].astype(int).values
    X_te = X.loc[test_mask].values
    y_te = y.loc[test_mask, 'label_binary'].astype(int).values

    print(f"  test_year={test_year}  train({len(y_tr):,}) "
          f"valid({len(y_va):,}) test({len(y_te):,})")
    print(f"    train pos_rate: {y_tr.mean()*100:.2f}% / "
          f"test pos_rate: {y_te.mean()*100:.2f}%")

    # 모델
    params = dict(cfg['xgb_params'])
    model = xgb.XGBClassifier(**params)

    if len(y_va) > 0:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        params_no_es = dict(params)
        params_no_es.pop("early_stopping_rounds", None)
        model = xgb.XGBClassifier(**params_no_es)
        model.fit(X_tr, y_tr, verbose=False)

    # 예측 (test)
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    # Validation 예측 → 최적 threshold 찾기
    if cfg.get('auto_threshold', True) and len(y_va) > 0:
        y_proba_va = model.predict_proba(X_va)[:, 1]
        valid_meta = meta.loc[valid_mask].reset_index(drop=True)
        pred_va = pd.DataFrame({
            'date': valid_meta['date'].values,
            'ticker': valid_meta['ticker'].values,
            'p_pos': y_proba_va,
        })
        best_threshold, thr_results = find_best_threshold(
            pred_va, label_df,
            initial_capital=cfg['initial_capital'],
        )
        print(f"    [auto threshold] best={best_threshold:.2f}  "
              f"(validation Sharpe 최대)")
    else:
        best_threshold = cfg.get('threshold', 0.5)
        thr_results = None

    # 메트릭
    try:
        auc = roc_auc_score(y_te, y_proba)
    except Exception:
        auc = np.nan
    try:
        ap = average_precision_score(y_te, y_proba)
    except Exception:
        ap = np.nan
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    prec10 = precision_at_k(y_te, y_proba, k_frac=0.1)
    prec5 = precision_at_k(y_te, y_proba, k_frac=0.05)

    # 슬롯 시뮬
    test_meta = meta.loc[test_mask].reset_index(drop=True)
    pred_df = pd.DataFrame({
        'date': test_meta['date'].values,
        'ticker': test_meta['ticker'].values,
        'p_pos': y_proba,
    })

    sim_results = {}
    for sel_mode in ['top', 'bottom', 'random']:
        trades, eq_s, eq_c = simulate_threshold_v4(
            pred_df, label_df,
            threshold=best_threshold, mode=sel_mode,
            initial_capital=cfg['initial_capital'],
        )
        stats_s = trade_stats(trades, eq_s, cfg['initial_capital'])
        stats_c = trade_stats(trades, eq_c, cfg['initial_capital'])
        sim_results[sel_mode] = {
            'trades': trades, 'equity_simple': eq_s, 'equity_compound': eq_c,
            'stats_simple': stats_s, 'stats_compound': stats_c,
        }

    # 모든 test 행 예측 저장 (앙상블용)
    test_meta_full = meta.loc[test_mask].reset_index(drop=True)
    test_predictions = pd.DataFrame({
        'date': test_meta_full['date'].values,
        'ticker': test_meta_full['ticker'].values,
        'p_pos': y_proba,
        'label_binary': y_te,
    })

    return {
        'test_year': test_year,
        'n_train': int(train_mask.sum()),
        'n_valid': int(valid_mask.sum()),
        'n_test': int(test_mask.sum()),
        'auc': float(auc),
        'ap': float(ap),
        'accuracy': float(acc),
        'f1': float(f1),
        'precision_top10pct': float(prec10),
        'precision_top5pct': float(prec5),
        'pos_rate_test': float(y_te.mean()),
        'best_threshold': float(best_threshold),
        'threshold_search': thr_results,
        'sim_results': sim_results,
        'test_predictions': test_predictions,
        'model': model,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="combine_v4.py 결과 폴더")
    p.add_argument("--ohlcv_dir", type=str, required=False,
                   help="(v4에선 사용 안 함, 호환성만 유지)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--test_years", type=str,
                   default="2018,2019,2020,2021,2022,2023,2024,2025,2026")
    p.add_argument("--purge_days", type=int, default=60,
                   help="train 마지막과 valid 시작 사이 gap")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="고정 매수 임계값 (--auto_threshold 끄면 사용)")
    p.add_argument("--auto_threshold", action="store_true", default=True,
                   help="Validation에서 Sharpe 최대 threshold 자동 결정 (기본)")
    p.add_argument("--fixed_threshold", action="store_true",
                   help="--threshold 값 고정 사용 (auto 비활성화)")
    p.add_argument("--topk_k", type=int, default=5,
                   help="(deprecated)")
    p.add_argument("--initial_capital", type=float, default=10_000_000)
    p.add_argument("--best_params_file", type=str, default=None)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 설정
    cfg = {
        'test_years': [int(y) for y in args.test_years.split(',')],
        'n_folds': len(args.test_years.split(',')),
        'purge_days': args.purge_days,
        'threshold': args.threshold,
        'auto_threshold': not args.fixed_threshold,
        'topk_k': args.topk_k,
        'initial_capital': args.initial_capital,
        'xgb_params': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'auc',
            'early_stopping_rounds': 30,
        },
    }

    # best_params 적용 (옵션)
    if args.best_params_file is not None:
        bp_path = Path(args.best_params_file)
        if not bp_path.exists():
            raise FileNotFoundError(f"{bp_path} 없음")
        with open(bp_path, 'r') as f:
            best_data = json.load(f)
        best_params = best_data.get('best_params', best_data)
        print(f"\n[best_params 적용] {bp_path}")
        for k, v in best_params.items():
            cfg['xgb_params'][k] = v
            print(f"  {k}: {v}")

    # 데이터 로드
    print("=" * 75)
    print("데이터 로드")
    print("=" * 75)
    X = pd.read_parquet(data_dir / 'X.parquet')
    y = pd.read_parquet(data_dir / 'y.parquet')
    meta = pd.read_parquet(data_dir / 'meta.parquet')
    meta['date'] = pd.to_datetime(meta['date'])
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  meta: {meta.shape}")

    # 라벨 분포
    print(f"\n  label_binary 분포:")
    print(y['label_binary'].value_counts(dropna=False))

    # 라벨 데이터 (슬롯 시뮬용)
    # B_return, B_outcome, B_holding_days 가 필요
    label_df = meta.copy()
    label_df['B_return'] = meta['B_return'] if 'B_return' in meta.columns else np.nan
    label_df['B_outcome'] = meta['B_outcome'] if 'B_outcome' in meta.columns else np.nan
    if 'B_holding_days' in meta.columns:
        label_df['B_holding_days'] = meta['B_holding_days']

    # Walk-forward 실행
    print(f"\n" + "=" * 75)
    print(f"Walk-forward 학습 ({cfg['n_folds']} folds)")
    print("=" * 75)

    fold_results = []
    t0 = time.time()
    for i, ty in enumerate(cfg['test_years']):
        print(f"\n[fold {i+1}/{cfg['n_folds']}]")
        result = train_one_fold(X, y, meta, ty, cfg, label_df)
        fold_results.append(result)

        # 출력
        r = result
        print(f"    AUC: {r['auc']:.4f}  AP: {r['ap']:.4f}  "
              f"Prec@10%: {r['precision_top10pct']:.4f}")

        for sm in ['top', 'random', 'bottom']:
            s = r['sim_results'][sm]['stats_compound']
            print(f"    [{sm:>6}] trades={s['n_trades']:>5} "
                  f"win={s['win_rate']:>5.1f}% "
                  f"avg_ret={s['avg_return']:>+6.2f}% "
                  f"total={s['total_return_pct']:>+8.2f}% "
                  f"sharpe={s['sharpe']:+.3f} mdd={s['mdd_pct']:+.2f}%")

    print(f"\n  학습 완료 ({time.time()-t0:.1f}s)")

    # 요약
    print(f"\n" + "=" * 75)
    print("Walk-forward 요약")
    print("=" * 75)

    summary_rows = []
    for r in fold_results:
        row = {
            'test_year': r['test_year'],
            'n_test': r['n_test'],
            'auc': r['auc'],
            'ap': r['ap'],
            'prec_top10': r['precision_top10pct'],
            'pos_rate': r['pos_rate_test'],
            'best_threshold': r['best_threshold'],
        }
        for sm in ['top', 'random', 'bottom']:
            s = r['sim_results'][sm]['stats_compound']
            row[f'{sm}_sharpe'] = s['sharpe']
            row[f'{sm}_total_ret'] = s['total_return_pct']
            row[f'{sm}_mdd'] = s['mdd_pct']
            row[f'{sm}_winrate'] = s['win_rate']
            row[f'{sm}_n_trades'] = s['n_trades']
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    # 평균
    print(f"\n[평균 메트릭]")
    print(f"  Mean AUC: {summary_df['auc'].mean():.4f}")
    print(f"  Mean AP: {summary_df['ap'].mean():.4f}")
    print(f"  Mean Prec@10%: {summary_df['prec_top10'].mean():.4f}")
    print(f"\n[Avg Return per Trade (TOP vs RANDOM vs BOTTOM)]")
    print(f"  TOP    avg: {summary_df['top_total_ret'].mean():+.2f}%   "
          f"median fold: {summary_df['top_total_ret'].median():+.2f}%")
    print(f"  RANDOM avg: {summary_df['random_total_ret'].mean():+.2f}%   "
          f"median fold: {summary_df['random_total_ret'].median():+.2f}%")
    print(f"  BOTTOM avg: {summary_df['bottom_total_ret'].mean():+.2f}%   "
          f"median fold: {summary_df['bottom_total_ret'].median():+.2f}%")
    print(f"\n[모델 Alpha (TOP - RANDOM)]")
    alpha = summary_df['top_total_ret'] - summary_df['random_total_ret']
    print(f"  Mean alpha: {alpha.mean():+.2f}% per trade")
    print(f"  Positive alpha folds: {(alpha > 0).sum()}/{len(summary_df)}")
    print(f"  Per-fold alpha: {alpha.tolist()}")

    # 저장
    summary_df.to_csv(output_dir / 'walk_forward_summary.csv', index=False)
    print(f"\n저장: {output_dir / 'walk_forward_summary.csv'}")

    # 마지막 fold 모델 저장
    last = fold_results[-1]
    last['model'].save_model(output_dir / 'last_model.json')

    # 거래 로그 (top mode)
    all_trades = pd.concat([r['sim_results']['top']['trades']
                            for r in fold_results], ignore_index=True)
    all_trades.to_csv(output_dir / 'all_trades_top.csv', index=False)
    print(f"저장: {output_dir / 'all_trades_top.csv'} ({len(all_trades)} trades)")

    # ============================================================
    # 앙상블용: 모든 test 행 예측 저장 (date, ticker, p_pos, label)
    # ============================================================
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    all_preds = []
    for r in fold_results:
        pred = r['test_predictions'].copy()
        pred['test_year'] = r['test_year']
        # fold별 저장
        pred.to_parquet(
            pred_dir / f"predictions_fold_{r['test_year']}.parquet",
            index=False
        )
        all_preds.append(pred)
    # 전체 통합
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    all_preds_df.to_parquet(output_dir / 'predictions_all.parquet', index=False)
    print(f"저장: {output_dir / 'predictions_all.parquet'} "
          f"({len(all_preds_df)} rows)")
    print(f"     fold별 파일: {pred_dir}/")


if __name__ == "__main__":
    main()
