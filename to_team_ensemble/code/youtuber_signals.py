"""
youtuber_signals.py
====================

15개 유튜버 자막에서 추출한 5개 매매기법 그룹의 시그널을 모두 생성.

5개 그룹:
A. 역배열 반등 (영매공파) - 영상 1,3,4,6,15
B. 정배열 눌림목 (N자형) - 영상 5,8,14
C. 수렴 후 돌파 - 영상 11,9 (256검색기)
D. 추세선 변곡 - 영상 12
E. 종가+거래량+이평 돌파 - 영상 10,13

각 그룹: 개별 시그널 + AND + Strength

이동평균: MA(5, 20, 60, 112, 224, 256, 448) - 영상 그대로 + 256 추가

사용법:
    python youtuber_signals.py --input ..\output_v3\df_full.parquet --output ..\output_v3_yt\df_full.parquet --analyze
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 공통 유틸: 이동평균선 + 볼린저
# ============================================================

def compute_indicators(g: pd.DataFrame) -> pd.DataFrame:
    """필요한 보조 지표 계산 (MA, BB)"""
    c = g['close']

    # 이동평균선
    g['_ma5'] = c.rolling(5, min_periods=3).mean()
    g['_ma20'] = c.rolling(20, min_periods=10).mean()
    g['_ma60'] = c.rolling(60, min_periods=30).mean()
    g['_ma112'] = c.rolling(112, min_periods=60).mean()
    g['_ma224'] = c.rolling(224, min_periods=120).mean()
    g['_ma256'] = c.rolling(256, min_periods=130).mean()
    g['_ma448'] = c.rolling(448, min_periods=200).mean()

    # 볼린저 밴드 (20일, 2 std)
    bb_mid = c.rolling(20, min_periods=10).mean()
    bb_std = c.rolling(20, min_periods=10).std()
    g['_bb_upper'] = bb_mid + bb_std * 2
    g['_bb_lower'] = bb_mid - bb_std * 2
    g['_bb_mid'] = bb_mid
    g['_bb_bandwidth'] = (g['_bb_upper'] - g['_bb_lower']) / (bb_mid + 1e-10)

    return g


# ============================================================
# Group A: 역배열 반등 (영매공파) - 이미 만든 함수 재사용
# ============================================================

def add_group_a_yokbae_rebound(
    g: pd.DataFrame,
    maejip_lookback: int = 20,
    maejip_vol_ratio: float = 2.0,
    gonguri_lookback: int = 20,
    gonguri_max_std_pct: float = 0.05,
    paran_proximity: float = 0.05,
    ma112_proximity: float = 0.05,
) -> pd.DataFrame:
    """역배열 반등 (영매공파)"""
    c = g['close']
    l = g['low']
    v = g['volume']

    # 1. 역배열
    g['sig_yokbae'] = ((g['_ma448'] > g['_ma224']) & (g['_ma224'] > g['_ma112'])).astype(int)

    # 2. 매집봉 (최근 N일 내 거래량 폭증)
    vol_avg = v.rolling(60, min_periods=20).mean()
    is_maejip_day = (v >= vol_avg * maejip_vol_ratio).astype(int)
    g['sig_maejip'] = is_maejip_day.rolling(maejip_lookback, min_periods=1).max().fillna(0).astype(int)

    # 3. 공구리 (박스권 지지)
    low_std = l.rolling(gonguri_lookback, min_periods=10).std()
    low_mean = l.rolling(gonguri_lookback, min_periods=10).mean()
    box_tight = low_std / (low_mean + 1e-10)
    g['sig_gonguri'] = ((box_tight < gonguri_max_std_pct) & (c > low_mean * 1.01)).astype(int)

    # 4. 파란 점선 (BB 하단 근접)
    dist_bb_lower = (c - g['_bb_lower']) / (c + 1e-10)
    g['sig_paran'] = ((dist_bb_lower >= 0) & (dist_bb_lower < paran_proximity)).astype(int)

    # 5. MA112 근접 + MA60 위
    dist_ma112 = (c - g['_ma112']).abs() / (c + 1e-10)
    g['sig_ma112'] = ((dist_ma112 < ma112_proximity) & (c > g['_ma60'])).astype(int)

    # 종합
    cond_cols = ['sig_yokbae', 'sig_maejip', 'sig_gonguri', 'sig_paran', 'sig_ma112']
    g['sig_yg_all'] = (g[cond_cols].sum(axis=1) == 5).astype(int)
    g['sig_yg_strength'] = g[cond_cols].sum(axis=1).astype(int)

    return g


# ============================================================
# Group B: 정배열 눌림목 (N자형, 이평때리기)
# ============================================================

def add_group_b_pullback_uptrend(
    g: pd.DataFrame,
    upbar_lookback: int = 10,
    upbar_min_ret: float = 0.05,
    pullback_proximity: float = 0.05,
    volume_ratio: float = 1.3,
) -> pd.DataFrame:
    """
    정배열 눌림목 매수 (영상 5, 8, 14 - N자형, 이평때리기)

    조건:
    1. 정배열: MA112 > MA224 > MA448 (역배열의 반대)
    2. MA60 또는 MA112 위 안착 (정배열 지지선)
    3. 최근 N일 내 장대양봉 (큰 상승)
    4. 현재 눌림 (이평선 근처로 되돌림)
    5. 거래량 동반 (당일 거래량 > 평균)
    """
    c = g['close']
    o = g['open']
    v = g['volume']

    # 1. 정배열
    g['sig_pullback_jeongbae'] = (
        (g['_ma112'] > g['_ma224']) & (g['_ma224'] > g['_ma448'])
    ).astype(int)

    # 2. MA60 위
    g['sig_pullback_ma60'] = (c > g['_ma60']).astype(int)

    # 3. 최근 N일 내 장대양봉 (전일 대비 5% 이상 상승하는 양봉)
    bar_ret = (c - c.shift(1)) / (c.shift(1) + 1e-10)
    is_jangde_yang = ((bar_ret >= upbar_min_ret) & (c > o)).astype(int)
    g['sig_pullback_recent_up'] = is_jangde_yang.rolling(upbar_lookback, min_periods=1).max().fillna(0).astype(int)

    # 4. 거래량 동반 (당일 거래량 > 20일 평균 × ratio)
    vol_avg = v.rolling(20, min_periods=10).mean()
    g['sig_pullback_volume'] = (v >= vol_avg * volume_ratio).astype(int)

    # 종합
    cond_cols = ['sig_pullback_jeongbae', 'sig_pullback_ma60',
                 'sig_pullback_recent_up', 'sig_pullback_volume']
    g['sig_pullback_all'] = (g[cond_cols].sum(axis=1) == 4).astype(int)
    g['sig_pullback_strength'] = g[cond_cols].sum(axis=1).astype(int)

    return g


# ============================================================
# Group C: 수렴 후 돌파 (256 검색기)
# ============================================================

def add_group_c_squeeze_breakout(
    g: pd.DataFrame,
    bb_squeeze_threshold: float = 0.10,
    ma_squeeze_threshold: float = 0.03,
    breakout_min_ret: float = 0.02,
) -> pd.DataFrame:
    """
    수렴 후 돌파 (영상 1, 9, 11 - 256검색기, 엘리어트 2파)

    조건:
    1. 볼린저 수렴 (bandwidth < 임계값, 변동성 압축)
    2. 이평 수렴 (5,20,60일선 간격이 좁음)
    3. 양봉 돌파 (당일 수익률 양수, 이평 위로 돌파)
    """
    c = g['close']
    o = g['open']

    # 1. 볼린저 수렴 (bandwidth가 과거 60일 평균보다 작음)
    bb_band_avg = g['_bb_bandwidth'].rolling(60, min_periods=20).mean()
    g['sig_squeeze_bb'] = (g['_bb_bandwidth'] < bb_band_avg * 0.8).astype(int)

    # 2. 이평 수렴 (MA5/MA20/MA60 간격이 좁음)
    ma_max = g[['_ma5', '_ma20', '_ma60']].max(axis=1)
    ma_min = g[['_ma5', '_ma20', '_ma60']].min(axis=1)
    ma_spread = (ma_max - ma_min) / (ma_min + 1e-10)
    g['sig_squeeze_ma'] = (ma_spread < ma_squeeze_threshold).astype(int)

    # 3. 양봉 돌파 (당일 수익률 양수 + 종가가 MA20 위)
    bar_ret = (c - o) / (o + 1e-10)
    g['sig_squeeze_breakout'] = (
        (bar_ret >= breakout_min_ret) & (c > g['_ma20'])
    ).astype(int)

    # 종합
    cond_cols = ['sig_squeeze_bb', 'sig_squeeze_ma', 'sig_squeeze_breakout']
    g['sig_squeeze_all'] = (g[cond_cols].sum(axis=1) == 3).astype(int)
    g['sig_squeeze_strength'] = g[cond_cols].sum(axis=1).astype(int)

    return g


# ============================================================
# Group D: 추세선 변곡 (higher highs/lows)
# ============================================================

def add_group_d_trend_change(
    g: pd.DataFrame,
    lookback_short: int = 20,
    lookback_long: int = 40,
) -> pd.DataFrame:
    """
    추세선 변곡 (영상 12)

    조건:
    1. higher lows: 최근 N일 저점이 그 이전 N일 저점보다 높음
    2. higher highs: 최근 N일 고점이 그 이전 N일 고점보다 높음
    3. 추세선 돌파: MA20을 양봉으로 돌파
    """
    h = g['high']
    l = g['low']
    c = g['close']
    o = g['open']

    # 최근 N일 저점/고점 vs 그 이전 N일 저점/고점
    recent_low = l.rolling(lookback_short, min_periods=10).min()
    prev_low = l.shift(lookback_short).rolling(lookback_short, min_periods=10).min()
    g['sig_trend_higher_lows'] = (recent_low > prev_low).astype(int)

    recent_high = h.rolling(lookback_short, min_periods=10).max()
    prev_high = h.shift(lookback_short).rolling(lookback_short, min_periods=10).max()
    g['sig_trend_higher_highs'] = (recent_high > prev_high).astype(int)

    # 추세선 돌파: 양봉으로 MA20 통과 (어제는 MA20 아래, 오늘은 위 + 양봉)
    crossed_ma20 = (c > g['_ma20']) & (c.shift(1) <= g['_ma20'].shift(1)) & (c > o)
    g['sig_trend_breakout'] = crossed_ma20.astype(int)

    # 종합
    cond_cols = ['sig_trend_higher_lows', 'sig_trend_higher_highs', 'sig_trend_breakout']
    g['sig_trend_all'] = (g[cond_cols].sum(axis=1) == 3).astype(int)
    g['sig_trend_strength'] = g[cond_cols].sum(axis=1).astype(int)

    return g


# ============================================================
# Group E: 종가+거래량+이평 돌파 (단테 대박 검색기)
# ============================================================

def add_group_e_close_breakout(
    g: pd.DataFrame,
    volume_explosion_ratio: float = 3.0,
    jangde_min_ret: float = 0.07,
) -> pd.DataFrame:
    """
    종가+거래량+이평 돌파 (영상 10, 13 - 단테 대박 검색기)

    조건:
    1. 거래량 폭증: 당일 거래량 > 20일 평균 × 3배
    2. 장대양봉: 당일 수익률 +7% 이상
    3. 장기이평 돌파: MA224 또는 MA448 돌파 (어제 아래 → 오늘 위)
    4. 볼린저 상단 돌파: 종가 > BB upper (5% 우수생)
    """
    c = g['close']
    o = g['open']
    v = g['volume']

    # 1. 거래량 폭증
    vol_avg = v.rolling(20, min_periods=10).mean()
    g['sig_close_volume'] = (v >= vol_avg * volume_explosion_ratio).astype(int)

    # 2. 장대양봉
    bar_ret = (c - c.shift(1)) / (c.shift(1) + 1e-10)
    g['sig_close_jangde'] = ((bar_ret >= jangde_min_ret) & (c > o)).astype(int)

    # 3. 장기이평 돌파 (MA224 또는 MA448)
    ma224_break = (c > g['_ma224']) & (c.shift(1) <= g['_ma224'].shift(1))
    ma448_break = (c > g['_ma448']) & (c.shift(1) <= g['_ma448'].shift(1))
    g['sig_close_ma_break'] = (ma224_break | ma448_break).astype(int)

    # 4. 볼린저 상단 돌파
    g['sig_close_bb_upper'] = (c > g['_bb_upper']).astype(int)

    # 종합
    cond_cols = ['sig_close_volume', 'sig_close_jangde',
                 'sig_close_ma_break', 'sig_close_bb_upper']
    g['sig_close_all'] = (g[cond_cols].sum(axis=1) == 4).astype(int)
    g['sig_close_strength'] = g[cond_cols].sum(axis=1).astype(int)

    return g


# ============================================================
# 종목별 통합 처리
# ============================================================

def add_all_signals_per_ticker(g: pd.DataFrame, **params) -> pd.DataFrame:
    """한 종목에 대해 5개 그룹 시그널 모두 추가"""
    g = compute_indicators(g)
    g = add_group_a_yokbae_rebound(g)
    g = add_group_b_pullback_uptrend(g)
    g = add_group_c_squeeze_breakout(g)
    g = add_group_d_trend_change(g)
    g = add_group_e_close_breakout(g)

    # 임시 지표 컬럼 제거
    temp_cols = [c for c in g.columns if c.startswith('_')]
    g = g.drop(columns=temp_cols)

    return g


def add_all_signals(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """전체 데이터에 종목별로 시그널 추가"""
    if verbose:
        print(f"[signals] 종목별 시그널 계산 ({df['ticker'].nunique()}개)...")

    t0 = time.time()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    result = df.groupby('ticker', group_keys=True).apply(
        add_all_signals_per_ticker, include_groups=False
    )
    result = result.reset_index()
    if 'level_1' in result.columns:
        result = result.drop(columns=['level_1'])
    result = result.reset_index(drop=True)

    if verbose:
        print(f"[signals] 완료 ({time.time()-t0:.1f}s)")
    return result


# ============================================================
# 분석
# ============================================================

GROUP_SIGNALS = {
    'A. 역배열 반등 (영매공파)': {
        'individuals': ['sig_yokbae', 'sig_maejip', 'sig_gonguri', 'sig_paran', 'sig_ma112'],
        'all': 'sig_yg_all',
        'strength': 'sig_yg_strength',
        'max_strength': 5,
    },
    'B. 정배열 눌림목 (N자형)': {
        'individuals': ['sig_pullback_jeongbae', 'sig_pullback_ma60',
                        'sig_pullback_recent_up', 'sig_pullback_volume'],
        'all': 'sig_pullback_all',
        'strength': 'sig_pullback_strength',
        'max_strength': 4,
    },
    'C. 수렴 후 돌파 (256 검색기)': {
        'individuals': ['sig_squeeze_bb', 'sig_squeeze_ma', 'sig_squeeze_breakout'],
        'all': 'sig_squeeze_all',
        'strength': 'sig_squeeze_strength',
        'max_strength': 3,
    },
    'D. 추세선 변곡': {
        'individuals': ['sig_trend_higher_lows', 'sig_trend_higher_highs', 'sig_trend_breakout'],
        'all': 'sig_trend_all',
        'strength': 'sig_trend_strength',
        'max_strength': 3,
    },
    'E. 종가+거래량+이평 돌파 (대박 검색기)': {
        'individuals': ['sig_close_volume', 'sig_close_jangde',
                        'sig_close_ma_break', 'sig_close_bb_upper'],
        'all': 'sig_close_all',
        'strength': 'sig_close_strength',
        'max_strength': 4,
    },
}


def analyze_signals(df: pd.DataFrame):
    """그룹별 시그널 분석"""
    print("\n" + "=" * 75)
    print("유튜버 시그널 분석 (5개 그룹)")
    print("=" * 75)

    has_label = 'label_3class' in df.columns
    if has_label:
        df_labeled = df.dropna(subset=['label_3class']).copy()
        df_labeled['_pos'] = (df_labeled['label_3class'] == 2).astype(int)
        overall_pos = df_labeled['_pos'].mean()
        print(f"\n전체 평균 양성 비율 (label=up): {overall_pos*100:.2f}%")

    for group_name, info in GROUP_SIGNALS.items():
        print(f"\n{'─' * 75}")
        print(f"## {group_name}")
        print(f"{'─' * 75}")

        # 개별 시그널 발생률
        print(f"\n[개별 조건 발생률]")
        for col in info['individuals']:
            rate = df[col].mean()
            cnt = int(df[col].sum())
            print(f"  {col:<28} {rate*100:>6.2f}% ({cnt:>9,}회)")

        # AND 시그널
        and_col = info['all']
        and_rate = df[and_col].mean()
        print(f"  {and_col:<28} {and_rate*100:>6.2f}% ({int(df[and_col].sum()):>9,}회)  ★ AND")

        # 시그널 + 라벨 분석
        if has_label:
            print(f"\n[시그널 켜졌을 때 양성 비율 vs 안 켜졌을 때]")
            for col in info['individuals'] + [info['all']]:
                on = df_labeled[df_labeled[col] == 1]['_pos']
                off = df_labeled[df_labeled[col] == 0]['_pos']
                if len(on) > 0:
                    pos_on = on.mean()
                    lift = pos_on / overall_pos if overall_pos > 0 else np.nan
                    star = ' ★' if col == info['all'] else ''
                    star += ' ✓' if lift >= 1.3 else (' ✗' if lift < 0.9 else '')
                    print(f"  {col:<28} {pos_on*100:>6.2f}% (lift {lift:>5.2f}x, n={len(on):>8,}){star}")

            # Strength별
            strength_col = info['strength']
            max_s = info['max_strength']
            print(f"\n[Strength별 양성 비율 (0~{max_s})]")
            for s in range(max_s + 1):
                sub = df_labeled[df_labeled[strength_col] == s]
                if len(sub) > 0:
                    pos = sub['_pos'].mean()
                    lift = pos / overall_pos if overall_pos > 0 else np.nan
                    star = ' ✓' if lift >= 1.3 else ''
                    print(f"  Strength={s:<2}  {pos*100:>6.2f}%  lift={lift:>5.2f}x  n={len(sub):>9,}{star}")

    # 전체 요약
    print(f"\n{'=' * 75}")
    print("AND 시그널 요약 (5개 그룹 비교)")
    print(f"{'=' * 75}")
    if has_label:
        print(f"\n{'그룹':<35} {'발생률':>10} {'양성률':>10} {'Lift':>8}")
        print('─' * 75)
        for group_name, info in GROUP_SIGNALS.items():
            and_col = info['all']
            rate = df[and_col].mean()
            on = df_labeled[df_labeled[and_col] == 1]['_pos']
            if len(on) > 0:
                pos_on = on.mean()
                lift = pos_on / overall_pos if overall_pos > 0 else np.nan
                print(f"{group_name:<35} {rate*100:>9.3f}% {pos_on*100:>9.2f}% {lift:>7.2f}x")


# ============================================================
# 메인
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="v3 df_full.parquet")
    p.add_argument("--output", type=str, required=True, help="시그널 추가된 출력")
    p.add_argument("--analyze", action="store_true", help="시그널 통계 분석")
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print("=" * 75)
    print("데이터 로드")
    print("=" * 75)
    df = pd.read_parquet(input_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  shape: {df.shape}")
    print(f"  종목: {df['ticker'].nunique()}개")
    print(f"  기간: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 필수 컬럼 확인
    required = ['date', 'ticker', 'close', 'high', 'low', 'open', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] 필수 컬럼 누락: {missing}")
        return

    # 시그널 생성
    print("\n" + "=" * 75)
    print("유튜버 시그널 생성 (5개 그룹)")
    print("=" * 75)
    df = add_all_signals(df)

    # 새 시그널 컬럼 카운트
    sig_cols = [c for c in df.columns if c.startswith('sig_')]
    print(f"\n  생성된 시그널: {len(sig_cols)}개")
    print(f"  최종 shape: {df.shape}")

    # 분석
    if args.analyze:
        analyze_signals(df)

    # 저장
    print(f"\n저장: {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"\n총 시간: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
