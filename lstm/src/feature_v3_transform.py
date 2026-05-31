"""
피처 v3: Cross-sectional 정규화 + 상호작용 피처
================================================

기존 v2 파이프라인의 피처에 추가 변환을 적용:
1. Cross-sectional z-score (같은 날짜 내 종목간 표준화)
2. 상호작용 피처 (논문/실전 검증된 조합)
3. 절대 시장 변수 → cross-sectional 시장 변수로 약화

사용법:
    from feature_v3_transform import transform_features_v3
    df_v3 = transform_features_v3(df_with_v2_features)

또는 CLI:
    python feature_v3_transform.py --input ../output/df_full.parquet --output ../output/df_v3.parquet
"""

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Step 1: Cross-sectional z-score 변환
# ============================================================

# CS 정규화할 피처 목록 (도메인 지식 기반 선별)
CS_NORMALIZE_FEATURES = [
    # 모멘텀
    'rsi_14', 'rsi_21',
    'stoch_fastK', 'stoch_fastD', 'stoch_slowD',
    'macd', 'macd_signal', 'macd_hist',
    'roc_5', 'roc_10', 'roc_20', 'roc_60',
    'williams_r_14',
    'cci_20',
    'mom_1', 'mom_5', 'mom_10', 'mom_15',

    # 변동성
    'vol_5', 'vol_10', 'vol_20', 'vol_60',
    'bb_pctB', 'bb_bandwidth',
    'atr_pct_14',
    'parkinson_vol_20', 'gk_vol_20',
    'vol_of_vol_20',
    'dist_to_15pct',

    # 거래량
    'vol_ratio_20', 'trade_value_ratio_20',
    'obv_chg_5',
    'vwap_dev',
    'mfi_14',
    'cmf_20',
    'vol_cv_20',
    'turnover_rt_chg',

    # 추세
    'adx_14',
    'pos_in_52w', 'dist_to_52w_high', 'dist_to_52w_low',
    'price_percentile_60d',

    # 수익률
    'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
    'close_to_ma_5', 'close_to_ma_10', 'close_to_ma_20',
    'close_to_ma_60', 'close_to_ma_120',
    'ma_ratio_5_20', 'ma_ratio_20_60',

    # 상대수익률 (이미 CS 성격이지만 z-score 추가)
    'rel_ret_5d', 'rel_ret_20d',
    'beta_60',

    # 타깃 전용
    'past_15pct_hits_1y', 'past_drawdown_neg35_1y',
    'mfe_30d_mean', 'mae_30d_mean',
]


def add_cs_zscore(df: pd.DataFrame, features: list, verbose: bool = True) -> pd.DataFrame:
    """
    같은 date 내 종목간 z-score 변환.
    원본 컬럼은 유지, '_cs' 접미사로 새 컬럼 추가.

    Z-score = (x - mean_t) / std_t
    where mean/std는 같은 date의 모든 종목 기준.

    NaN-safe: groupby transform이 NaN을 무시함.
    """
    df = df.copy()
    available = [c for c in features if c in df.columns]
    skipped = [c for c in features if c not in df.columns]
    if verbose and skipped:
        print(f"  [INFO] 누락된 컬럼 스킵 ({len(skipped)}): {skipped[:5]}...")

    t0 = time.time()
    # 벡터화: 한 번에 groupby로 처리
    grouped = df.groupby('date')[available]
    means = grouped.transform('mean')
    stds = grouped.transform('std')

    for c in available:
        # 분모 0 보호
        z = (df[c] - means[c]) / (stds[c] + 1e-10)
        # 극단값 클리핑 (-5 ~ +5)
        z = z.clip(-5, 5)
        df[f'{c}_cs'] = z

    if verbose:
        print(f"  [cs_zscore] {len(available)}개 피처 CS-정규화 완료 "
              f"({time.time()-t0:.1f}s)")
    return df


# ============================================================
# Step 2: 상호작용 피처
# ============================================================

def add_interaction_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    도메인 지식 기반 상호작용 피처 추가.
    원본이 있어야 동작하므로 안전하게 체크.
    """
    df = df.copy()
    added = []

    def safe_mult(a, b, name):
        """둘 다 있을 때만 추가"""
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]
            added.append(name)

    # 1. 모멘텀 × 거래량 (브레이크아웃 신호)
    safe_mult('rsi_14_cs', 'vol_ratio_20_cs', 'inter_rsi_volsurge')
    safe_mult('roc_20_cs', 'vol_ratio_20_cs', 'inter_roc_volsurge')
    safe_mult('macd_hist_cs', 'vol_ratio_20_cs', 'inter_macd_volsurge')

    # 2. 신고가 + 추세 강도 (강한 모멘텀 신호)
    safe_mult('dist_to_52w_high_cs', 'adx_14_cs', 'inter_52high_trend')
    safe_mult('pos_in_52w_cs', 'adx_14_cs', 'inter_52pos_trend')

    # 3. 과매도 + 거래량 (반등 신호)
    if 'rsi_14_cs' in df.columns and 'vol_ratio_20_cs' in df.columns:
        oversold = (-df['rsi_14_cs']).clip(lower=0)  # RSI 낮을수록 큰 값
        df['inter_oversold_volume'] = oversold * df['vol_ratio_20_cs']
        added.append('inter_oversold_volume')

    # 4. 변동성 × 모멘텀 (큰 움직임 가능성)
    safe_mult('vol_20_cs', 'roc_20_cs', 'inter_vol_momentum')
    safe_mult('atr_pct_14_cs', 'rsi_14_cs', 'inter_atr_rsi')

    # 5. 업종 강세 × 종목 강세 (sector 정보 있을 때만)
    if 'sector_rank_ret_20d' in df.columns and 'cs_rank_ret_20d' in df.columns:
        df['inter_sector_stock_strong'] = (
            df['sector_rank_ret_20d'] * df['cs_rank_ret_20d']
        )
        added.append('inter_sector_stock_strong')

    # 6. 변동성 대비 15% 거리 × 모멘텀 (target-aware)
    safe_mult('dist_to_15pct_cs', 'roc_20_cs', 'inter_dist15_momentum')

    # 7. Bollinger %B × 거래량 (밴드 돌파 + 확인)
    safe_mult('bb_pctB_cs', 'vol_ratio_20_cs', 'inter_bb_volume')

    # 8. 시총 × 변동성 (소형주 변동성 효과)
    if 'log_mktcap' in df.columns and 'vol_20_cs' in df.columns:
        # 시총 작을수록 변동성 영향 큼: -log_mktcap × vol
        df['inter_smallcap_vol'] = -df['log_mktcap'] * df['vol_20_cs']
        added.append('inter_smallcap_vol')

    # 9. 과거 +15% 도달 빈도 × 현재 모멘텀 (반복성)
    safe_mult('past_15pct_hits_1y_cs', 'roc_20_cs', 'inter_history_momentum')

    if verbose:
        print(f"  [interactions] {len(added)}개 상호작용 피처 추가")
    return df


# ============================================================
# Step 3: 시장 변수 약화
# ============================================================

ABSOLUTE_MARKET_FEATURES = [
    'market_ret_1d', 'market_ret_5d', 'market_ret_20d',
    'market_vol_20', 'market_vol_chg', 'market_breadth',
    'sector_ret_1d',  # sector_rel_*은 유지
]


def weaken_market_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    절대 시장 변수 제거. Cross-sectional 시장 변수(rank, relative)는 유지.
    이러면 모델이 시장 타이밍 대신 종목 선별에 집중.
    """
    df = df.copy()
    to_drop = [c for c in ABSOLUTE_MARKET_FEATURES if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    if verbose:
        print(f"  [weaken_market] {len(to_drop)}개 절대 시장 변수 제거: {to_drop}")
    return df


# ============================================================
# 통합 변환
# ============================================================

def transform_features_v3(
    df: pd.DataFrame,
    add_cs: bool = True,
    add_interactions: bool = True,
    weaken_market: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    v2 피처에 v3 변환 적용.

    Args:
        df: v2 파이프라인 결과 (build_*_features 후)
        add_cs: cross-sectional z-score 추가
        add_interactions: 상호작용 피처 추가
        weaken_market: 절대 시장 변수 제거
    """
    if verbose:
        print(f"\n[transform_v3] 입력 shape: {df.shape}")

    if add_cs:
        df = add_cs_zscore(df, CS_NORMALIZE_FEATURES, verbose=verbose)

    if add_interactions:
        df = add_interaction_features(df, verbose=verbose)

    if weaken_market:
        df = weaken_market_features(df, verbose=verbose)

    if verbose:
        print(f"[transform_v3] 출력 shape: {df.shape}")

    return df


# ============================================================
# 학습 데이터 재준비 (v3 컬럼 반영)
# ============================================================

def prepare_train_data_v3(df: pd.DataFrame, label_col: str = 'label_binary'):
    """v3용 X, y, meta 분리"""
    default_drop = [
        # 식별자·날짜
        'date', 'ticker',
        # 시간 누수
        'year', 'month', 'day', 'quarter', 'dayofweek', '_year',
        # 원본 절대값
        'open', 'high', 'low', 'close', 'volume', 'trade_value',
        'chg_sig', 'chg', 'tradeable',
        # 추정 절대값
        'shares_est', 'mktcap_est',
        'trade_value_ma20', 'turnover_rt_ma20',
        # 텍스트
        'stk_nm', 'market', 'sector',
        # 라벨
        'label_3class', 'label_binary', 'days_to_event', 'realized_ret',
    ]
    drop_cols = list(set(default_drop))
    df = df.dropna(subset=[label_col]).copy()
    y = df[label_col].astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan)
    meta = df[['date', 'ticker']].reset_index(drop=True)
    return X, y, meta


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True,
                   help='v2 파이프라인 결과 df_full.parquet')
    p.add_argument('--output_dir', type=str, default='../output_v3')
    p.add_argument('--no_cs', action='store_true', help='CS 정규화 안 함')
    p.add_argument('--no_interactions', action='store_true', help='상호작용 안 함')
    p.add_argument('--keep_market', action='store_true', help='절대 시장 변수 유지')
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 60)
    print(f"입력 로드: {args.input}")
    print("=" * 60)
    df = pd.read_parquet(args.input)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  shape: {df.shape}")
    print(f"  date 범위: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  컬럼 수: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("v3 변환 적용")
    print("=" * 60)
    df = transform_features_v3(
        df,
        add_cs=not args.no_cs,
        add_interactions=not args.no_interactions,
        weaken_market=not args.keep_market,
    )

    print("\n" + "=" * 60)
    print("X, y 분리 + 저장")
    print("=" * 60)
    X, y, meta = prepare_train_data_v3(df)
    print(f"  X: {X.shape}  ({X.shape[1]}개 피처)")
    print(f"  y: {y.shape}  (양성 비율 {y.mean():.4f})")
    print(f"  결측치: {X.isna().sum().sum():,} / {X.size:,} "
          f"({X.isna().sum().sum()/X.size*100:.2f}%)")

    X.to_parquet(output_dir / 'X.parquet', index=False)
    y.to_frame('label_binary').to_parquet(output_dir / 'y.parquet', index=False)
    meta.to_parquet(output_dir / 'meta.parquet', index=False)
    df.to_parquet(output_dir / 'df_full.parquet', index=False)

    # 피처 카테고리별 카운트
    cats = {
        'cs (정규화)': sum(1 for c in X.columns if c.endswith('_cs')),
        'inter (상호작용)': sum(1 for c in X.columns if c.startswith('inter_')),
        'cs_rank (cross-section rank)': sum(1 for c in X.columns if c.startswith('cs_rank_')),
        'sector_': sum(1 for c in X.columns if c.startswith('sector_')),
        'market_': sum(1 for c in X.columns if c.startswith('market_')),
        '나머지 (원본)': 0,
    }
    cats['나머지 (원본)'] = X.shape[1] - sum(v for k, v in cats.items() if k != '나머지 (원본)')
    print("\n피처 카테고리:")
    for k, v in cats.items():
        print(f"  {k}: {v}개")

    print(f"\n저장 위치: {output_dir}")
    print(f"총 시간: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
