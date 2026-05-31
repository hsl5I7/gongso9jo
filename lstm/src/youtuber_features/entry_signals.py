"""
entry_signals.py
================

진입 신호 12개 (유튜버 영상 1~15에서 추출).

룰 ID와 영상 출처:
- R3.1   : MA224 cross-over (영상 3)
- R4.1   : 장기 역배열 + 단기 골든크로스 (영상 4)
- R5.1   : 돌파 후 눌림 양봉 (영상 5)
- R5.2   : 음봉 후 양봉 변곡 (영상 5)
- R9.6   : 이평 수렴 후 재돌파 (영상 9)
- R10.2  : 볼린저 + MA224 종가 동시 돌파 (영상 10) ★ 핵심 단순 룰
- R12.1  : 하락 추세선 돌파 (영상 12)
- R13.1  : 매물대 돌파 + 거래량 동반 (영상 13)
- R14.1  : 이격 벌어진 후 이평 회귀 (영상 14)
- R15.1  : 역헤드앤숄더 (영상 15)
- R15.2  : Higher Low (영상 15)
- R2.1   : 영매공파 5조건 합성 (영상 2) ★ 메인 알파

모든 신호 컬럼은 0/1 binary.
prefix 'sig_' 로 통일.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ohlcv_helpers import cross_over


# ─────────────────────────────────────────────────────────────────────
# R3.1 : MA224 cross-over
# ─────────────────────────────────────────────────────────────────────
def sig_ma224_crossover(df: pd.DataFrame) -> pd.Series:
    """오늘 종가가 MA224를 위로 막 돌파."""
    return cross_over(df["Close"], df["ma_224"])


# ─────────────────────────────────────────────────────────────────────
# R4.1 : 장기 역배열 + 단기 골든크로스
# ─────────────────────────────────────────────────────────────────────
def sig_long_inverted_short_golden(df: pd.DataFrame) -> pd.Series:
    """
    장기 이평 역배열 상태에서 단기 이평 골든크로스 발생.
    영매공파의 "역" + "정렬 단계" 부분을 구체화한 신호.
    """
    long_inverted = df["long_term_inverted"] == 1
    ma20_cross_ma60 = cross_over(df["ma_20"], df["ma_60"])
    return (long_inverted & (ma20_cross_ma60 == 1)).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R5.1 : 돌파 후 눌림 양봉
# ─────────────────────────────────────────────────────────────────────
def sig_breakout_pullback_bullish(
    df: pd.DataFrame, breakout_lookback: int = 20, pullback_window: int = 3
) -> pd.Series:
    """
    조건:
      1) 직전 N일 안에 'breakout' = 직전 high를 갱신한 큰 양봉이 있음
      2) 그 이후 1~3일 내 close가 그 breakout high보다 살짝 눌림
      3) 오늘 양봉 발생
    """
    high = df["High"]
    close = df["Close"]
    open_ = df["Open"]

    # 직전 N일 (lookback) 동안 가장 큰 양봉 + 신고가 갱신했던 자리
    prev_high = high.shift(1).rolling(breakout_lookback, min_periods=1).max()
    breakout = (high > prev_high)  # 직전 신고가 돌파
    breakout_close = close.where(breakout)
    # pullback_window일 안에 발생한 가장 최근 breakout high
    recent_breakout_high = high.where(breakout).rolling(
        breakout_lookback, min_periods=1
    ).max().shift(1)

    # 오늘이 그 breakout high보다 살짝 아래(눌림) + 양봉
    is_pullback = (close < recent_breakout_high) & (
        close > recent_breakout_high * 0.95
    )
    is_bullish = close > open_
    return (is_pullback & is_bullish).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R5.2 : 음봉 후 양봉 변곡
# ─────────────────────────────────────────────────────────────────────
def sig_reversal_bullish_candle(
    df: pd.DataFrame, prev_window: int = 5, body_ratio_th: float = 0.5
) -> pd.Series:
    """
    직전 N일 음봉 우세 → 오늘 큰 양봉 (body_ratio > 0.5).
    """
    is_bearish = df["is_bearish"]
    is_bullish = df["is_bullish"]

    # 직전 N일 음봉 비율 60% 이상
    prev_bear_ratio = is_bearish.shift(1).rolling(prev_window).mean()
    prev_dominant_bear = prev_bear_ratio >= 0.6

    today_strong_bull = (is_bullish == 1) & (df["body_ratio_abs"] > body_ratio_th)
    return (prev_dominant_bear & today_strong_bull).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R9.6 : 이평 수렴 후 재돌파 (대시세 패턴)
# ─────────────────────────────────────────────────────────────────────
def sig_ma_squeeze_breakout(
    df: pd.DataFrame, squeeze_th: float = 0.02
) -> pd.Series:
    """
    조건:
      1) 이평 spread (ma_spread_std) 가 임계 이하 (수렴 상태)
      2) 오늘 close가 ma112 또는 ma224 위로 cross-over
    """
    is_squeezed = df["ma_spread_std"] < squeeze_th
    breakout = (
        (cross_over(df["Close"], df["ma_112"]) == 1) |
        (cross_over(df["Close"], df["ma_224"]) == 1)
    )
    return (is_squeezed & breakout).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R10.2 : 볼린저 + MA224 종가 동시 돌파 ★ 핵심 단순 룰
# ─────────────────────────────────────────────────────────────────────
def sig_bb_ma224_dual_breakout(df: pd.DataFrame) -> pd.Series:
    """
    종가가 볼린저 상단과 MA224를 동시에 막 돌파한 시점.
    "표준편차 우수생이 1년 저항을 뚫는 자리".
    """
    bb_breakout = cross_over(df["Close"], df["bb_upper"])
    ma224_breakout = cross_over(df["Close"], df["ma_224"])
    return ((bb_breakout == 1) & (ma224_breakout == 1)).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R12.1 : 하락 추세선 돌파
# ─────────────────────────────────────────────────────────────────────
def sig_downtrend_breakout(
    df: pd.DataFrame, slope_window: int = 20, breakout_lookback: int = 20
) -> pd.Series:
    """
    조건:
      1) 직전 slope_window일의 high 추세 기울기가 음수 (하락 추세)
      2) 오늘 close가 직전 breakout_lookback일 high의 max를 돌파
    """
    high_slope = df[f"high_trend_slope_{slope_window}"]
    prev_high_max = df["High"].shift(1).rolling(breakout_lookback, min_periods=1).max()
    is_downtrend = high_slope < 0
    breakout_today = df["Close"] > prev_high_max
    return (is_downtrend & breakout_today).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R13.1 : 매물대 돌파 + 거래량 동반
# ─────────────────────────────────────────────────────────────────────
def sig_volume_breakout_at_resistance(
    df: pd.DataFrame, vol_multiple: float = 2.0
) -> pd.Series:
    """
    조건:
      1) 종가가 매물대(직전 60일 거래량 peak 자리의 high)를 돌파
      2) 오늘 거래량이 20일 평균 거래량의 N배 이상
    """
    resistance = df["resistance_high"]
    breakout_resistance = (df["Close"] > resistance) & (
        df["Close"].shift(1) <= resistance.shift(1)
    )
    high_volume = df["vol_ratio_20"] > vol_multiple
    return (breakout_resistance & high_volume).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R14.1 : 이격 벌어진 후 이평 회귀 (이평 때리기)
# ─────────────────────────────────────────────────────────────────────
def sig_ma_pullback_signal(
    df: pd.DataFrame,
    gap_threshold: float = 0.20,
    near_threshold: float = 0.03,
    lookback: int = 20,
) -> pd.Series:
    """
    조건:
      1) 직전 lookback일 동안 close가 MA60 대비 +20% 이상 벌어진 적이 있음
      2) 오늘 close가 MA60 ±3% 이내
    """
    gap_from_ma60 = (df["Close"] - df["ma_60"]) / df["ma_60"]
    gap_was_high = gap_from_ma60.shift(1).rolling(lookback, min_periods=1).max() > gap_threshold
    near_ma60 = gap_from_ma60.abs() < near_threshold
    came_from_above = gap_from_ma60.shift(5) > gap_from_ma60  # 위에서 내려온 것
    return (gap_was_high & near_ma60 & came_from_above).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R15.1 : 역헤드앤숄더 (단순 근사)
# ─────────────────────────────────────────────────────────────────────
def sig_inverse_head_shoulders(df: pd.DataFrame) -> pd.Series:
    """
    단순 근사:
      - 가장 최근 15일 최저점이 직전 15~30일 최저점보다 높음 (오른어깨 > 머리)
      - 직전 15~30일 최저점이 30~60일 최저점보다 낮음 (머리 < 왼어깨)
      - 양 어깨 (최근, 30~60일 전) 가 비슷한 수준 (5% 이내)
    """
    low = df["Low"]
    recent = low.rolling(15, min_periods=15).min()                    # 오른어깨
    middle = low.shift(15).rolling(15, min_periods=15).min()           # 머리
    older = low.shift(30).rolling(30, min_periods=30).min()            # 왼어깨

    head_lower = middle < older                                        # 머리가 왼어깨보다 낮음
    right_shoulder_higher = recent > middle                            # 오른어깨가 머리보다 높음
    shoulders_balanced = (recent - older).abs() / older < 0.05         # 양 어깨 비슷

    return (head_lower & right_shoulder_higher & shoulders_balanced).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R15.2 : Higher Low (저점이 점차 높아짐)
# ─────────────────────────────────────────────────────────────────────
def sig_higher_low(df: pd.DataFrame, window: int = 15) -> pd.Series:
    """
    최근 window일 최저점이 직전 window일 최저점보다 높음.
    가장 단순한 추세 전환 시그널.
    """
    recent_low = df["Low"].rolling(window, min_periods=window).min()
    prev_low = df["Low"].shift(window).rolling(window, min_periods=window).min()
    return (recent_low > prev_low).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R2.1 : 영매공파 5조건 합성 ★ 메인 알파
# ─────────────────────────────────────────────────────────────────────
def sig_youngmaegongpa(
    df: pd.DataFrame,
    accumulation_lookback: int = 20,
    accumulation_vol_th: float = 2.0,
    accumulation_body_th: float = 0.3,
    box_window: int = 20,
    box_volatility_th: float = 0.10,
    bb_squeeze_th: float = 0.15,
    ma112_proximity_th: float = 0.05,
) -> pd.Series:
    """
    영매공파 = 5조건 AND
      영(역)  : ma112 < ma224 < ma448  (장기 역배열)
      매      : 직전 N일 내 매집봉 (대량거래 + 큰 양봉) 발생
      공(공구리): 좁은 박스권 (직전 박스 변동폭 < 5%)
      파      : 볼린저 수렴 (bb_width_ratio < 0.10)
      112    : close가 ma112 ±3% 이내 또는 위로 막 돌파

    실제 트레이더의 "영매공파 자리" 검색기를 모방한 신호.
    """
    # 영: 장기 역배열
    cond_inverted = df["long_term_inverted"] == 1

    # 매: 직전 N일 내 매집봉 (vol_ratio_20 > 2 + body_ratio > 0.4 + 양봉)
    is_acc_candle = (
        (df["vol_ratio_20"] > accumulation_vol_th) &
        (df["body_ratio_abs"] > accumulation_body_th) &
        (df["is_bullish"] == 1)
    ).astype(int)
    cond_acc = is_acc_candle.rolling(accumulation_lookback, min_periods=1).max() > 0

    # 공: 박스권 (직전 N일 high-low 변동폭 / 평균가 < 임계값)
    box_high = df["High"].shift(1).rolling(box_window, min_periods=box_window).max()
    box_low = df["Low"].shift(1).rolling(box_window, min_periods=box_window).min()
    box_mid = (box_high + box_low) / 2
    box_volatility = (box_high - box_low) / box_mid
    cond_box = box_volatility < box_volatility_th

    # 파: 볼린저 수렴
    cond_squeeze = df["bb_width_ratio"] < bb_squeeze_th

    # 112: ma112 근접 또는 막 돌파
    proximity = (df["Close"] - df["ma_112"]).abs() / df["ma_112"]
    near_or_cross = (proximity < ma112_proximity_th) | (
        cross_over(df["Close"], df["ma_112"]) == 1
    )

    signal = (
        cond_inverted & cond_acc & cond_box & cond_squeeze & near_or_cross
    ).astype(int)
    return signal


# ─────────────────────────────────────────────────────────────────────
# 통합 함수
# ─────────────────────────────────────────────────────────────────────
def add_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["sig_R3_1_ma224_crossover"] = sig_ma224_crossover(df)
    df["sig_R4_1_long_inv_short_golden"] = sig_long_inverted_short_golden(df)
    df["sig_R5_1_breakout_pullback"] = sig_breakout_pullback_bullish(df)
    df["sig_R5_2_reversal_bullish"] = sig_reversal_bullish_candle(df)
    df["sig_R9_6_ma_squeeze_breakout"] = sig_ma_squeeze_breakout(df)
    df["sig_R10_2_bb_ma224_dual"] = sig_bb_ma224_dual_breakout(df)
    df["sig_R12_1_downtrend_breakout"] = sig_downtrend_breakout(df)
    df["sig_R13_1_volume_breakout"] = sig_volume_breakout_at_resistance(df)
    df["sig_R14_1_ma_pullback"] = sig_ma_pullback_signal(df)
    df["sig_R15_1_inverse_hs"] = sig_inverse_head_shoulders(df)
    df["sig_R15_2_higher_low"] = sig_higher_low(df)
    df["sig_R2_1_youngmaegongpa"] = sig_youngmaegongpa(df)
    return df
