"""
ohlcv_helpers.py
================

26개 유튜버 룰을 계산하기 위한 공통 헬퍼 컬럼들.
이평선, 볼린저밴드, 매물대, body 비율, 추세 기울기, 이평 spread 등
여러 룰에서 재사용되는 기본 시계열을 만든다.

모든 함수는 한 종목 단위 DataFrame을 받아 컬럼을 추가해 반환한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# 이동평균선
# ─────────────────────────────────────────────────────────────────────
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    유튜버가 자주 언급하는 이평선 모두 추가.
    - 단기: 5, 20, 60
    - 장기: 112, 224, 448
    """
    close = df["Close"]
    for w in [5, 20, 60, 112, 224, 448]:
        df[f"ma_{w}"] = close.rolling(w, min_periods=w).mean()
    return df


# ─────────────────────────────────────────────────────────────────────
# 볼린저밴드 (영매공파의 "파")
# ─────────────────────────────────────────────────────────────────────
def add_bollinger(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    """
    표준 볼린저밴드 (window=20, k=2).
    R10.1, R10.2, R1.4, R9.6 등에 사용.
    """
    close = df["Close"]
    ma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    df["bb_mid"] = ma
    df["bb_upper"] = ma + k * std
    df["bb_lower"] = ma - k * std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_width_ratio"] = df["bb_width"] / ma  # 정규화된 폭 (수렴 측정)
    return df


# ─────────────────────────────────────────────────────────────────────
# 캔들 body / shadow 비율 (R2.3, R13.4, R13.5, R13.7 사용)
# ─────────────────────────────────────────────────────────────────────
def add_candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    rng = (h - l).replace(0, np.nan)  # zero-range 방지
    body = (c - o)

    df["body_size"] = body
    df["body_abs"] = body.abs()
    df["body_ratio_signed"] = body / rng                     # 음수=음봉
    df["body_ratio_abs"] = df["body_abs"] / rng
    df["body_pct_of_close"] = df["body_abs"] / c             # 종가 대비 몸통 크기
    df["upper_shadow"] = (h - df[["Open", "Close"]].max(axis=1)) / rng
    df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - l) / rng
    df["change_pct"] = c / c.shift(1) - 1                    # 전일 종가 대비 등락률
    df["is_bullish"] = (c > o).astype(int)
    df["is_bearish"] = (c < o).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────
# 거래량 헬퍼
# ─────────────────────────────────────────────────────────────────────
def add_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    v = df["Volume"]
    df["vol_ma_5"] = v.rolling(5, min_periods=5).mean()
    df["vol_ma_20"] = v.rolling(20, min_periods=20).mean()
    df["vol_ma_60"] = v.rolling(60, min_periods=60).mean()
    df["vol_ratio_20"] = v / df["vol_ma_20"]
    df["vol_ratio_60"] = v / df["vol_ma_60"]
    return df


# ─────────────────────────────────────────────────────────────────────
# 매물대 (R13.1 사용)
# ─────────────────────────────────────────────────────────────────────
def add_volume_profile_resistance(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    매물대 = 직전 lookback일에서 거래량이 가장 많이 터졌던 자리의 high.
    실제 volume profile(가격대별 거래량 분포)을 단순화한 근사.
    """
    high = df["High"]
    vol = df["Volume"]

    # rolling argmax: 직전 lookback일 중 거래량이 가장 큰 인덱스
    def _peak_high(i):
        if i < lookback:
            return np.nan
        window_vol = vol.iloc[i - lookback : i].values
        window_high = high.iloc[i - lookback : i].values
        return float(window_high[np.argmax(window_vol)])

    peaks = np.full(len(df), np.nan)
    for i in range(len(df)):
        peaks[i] = _peak_high(i)
    df["resistance_high"] = peaks
    return df


# ─────────────────────────────────────────────────────────────────────
# 추세선 기울기 (R12.1, R12.4, R12.5 사용)
# ─────────────────────────────────────────────────────────────────────
def add_trend_slopes(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    직전 window일의 high 시계열에 선형회귀 기울기 계산.
    음수면 하락 추세, 양수면 상승 추세.
    """
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    high_slope = np.full(n, np.nan)
    low_slope = np.full(n, np.nan)
    x = np.arange(window)

    for i in range(window - 1, n):
        h_win = high[i - window + 1 : i + 1]
        l_win = low[i - window + 1 : i + 1]
        if np.isnan(h_win).any() or np.isnan(l_win).any():
            continue
        # 단순 polyfit 1차 (numpy 스칼라 두 개 반환)
        h_slope_val, _ = np.polyfit(x, h_win, 1)
        l_slope_val, _ = np.polyfit(x, l_win, 1)
        # 기울기를 가격 단위가 아닌 정규화 단위로
        high_slope[i] = h_slope_val / np.mean(h_win)
        low_slope[i] = l_slope_val / np.mean(l_win)

    df[f"high_trend_slope_{window}"] = high_slope
    df[f"low_trend_slope_{window}"] = low_slope
    return df


# ─────────────────────────────────────────────────────────────────────
# 이평 정렬 상태 (R4.1, R4.2, R4.3 사용)
# ─────────────────────────────────────────────────────────────────────
def add_ma_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    이평선 정렬 상태 분석 헬퍼.
    """
    ma5 = df["ma_5"]
    ma20 = df["ma_20"]
    ma60 = df["ma_60"]
    ma112 = df["ma_112"]
    ma224 = df["ma_224"]
    ma448 = df["ma_448"]

    # 단기 정배열: ma5 > ma20 > ma60
    df["short_term_aligned_up"] = (
        (ma5 > ma20) & (ma20 > ma60)
    ).astype(int)

    # 장기 역배열: ma112 < ma224 < ma448
    df["long_term_inverted"] = (
        (ma112 < ma224) & (ma224 < ma448)
    ).astype(int)

    # 완전 정배열: ma5 > ma20 > ma60 > ma112 > ma224 > ma448
    df["full_uptrend_aligned"] = (
        (ma5 > ma20) & (ma20 > ma60) &
        (ma60 > ma112) & (ma112 > ma224) & (ma224 > ma448)
    ).astype(int)

    # 이평 spread (수렴 측정 — R9.6 활용)
    ma_stack = pd.concat([ma20, ma60, ma112, ma224], axis=1)
    df["ma_spread_std"] = ma_stack.std(axis=1) / ma_stack.mean(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────
# Cross-over 헬퍼
# ─────────────────────────────────────────────────────────────────────
def cross_over(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """A가 B를 위로 막 돌파한 시점 → 1, 아니면 0."""
    return (
        (series_a >= series_b) &
        (series_a.shift(1) < series_b.shift(1))
    ).astype(int)


def cross_under(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """A가 B를 아래로 막 이탈한 시점 → 1, 아니면 0."""
    return (
        (series_a <= series_b) &
        (series_a.shift(1) > series_b.shift(1))
    ).astype(int)


# ─────────────────────────────────────────────────────────────────────
# 통합 함수
# ─────────────────────────────────────────────────────────────────────
def add_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """모든 헬퍼 컬럼 한 번에 추가."""
    df = add_moving_averages(df)
    df = add_bollinger(df)
    df = add_candle_metrics(df)
    df = add_volume_metrics(df)
    df = add_volume_profile_resistance(df, lookback=60)
    df = add_trend_slopes(df, window=20)
    df = add_ma_alignment(df)
    return df
