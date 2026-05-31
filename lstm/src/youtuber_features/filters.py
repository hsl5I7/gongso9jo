"""
filters.py
==========

회피 필터 4개 (진입 금지 시그널).

- R3.4   : MA224 위 과열 자리 회피 (영상 3)
- R4.2   : 정배열 확장 자리 회피 (영상 4)
- R12.5  : 고점 단계적 하락 (영상 12)
- R14.5  : 데드 기법 — 역배열에서 일시 반등 (영상 14)

모든 필터 컬럼은 0/1 binary.
'1 = 진입 금지/위험' 의미.
prefix 'filter_' 로 통일.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# R3.4 : MA224 위 과열 자리 회피
# ─────────────────────────────────────────────────────────────────────
def filter_ma224_overheat(df: pd.DataFrame, threshold: float = 0.20) -> pd.Series:
    """
    close가 MA224 대비 +20% 이상 위에 있을 때 = 과열.
    """
    gap = (df["Close"] - df["ma_224"]) / df["ma_224"]
    return (gap > threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R4.2 : 정배열 확장 자리 회피
# ─────────────────────────────────────────────────────────────────────
def filter_full_uptrend_expanding(df: pd.DataFrame) -> pd.Series:
    """
    완전 정배열 (ma5 > ma20 > ma60 > ma112 > ma224 > ma448) AND
    이평 spread가 점차 커지고 있는 상태 (확장).

    "누구나 좋아 보이는 자리" — 진입 금지.
    """
    full_aligned = df["full_uptrend_aligned"] == 1

    # spread가 5일 전보다 커졌으면 확장 중
    spread_now = df["ma_spread_std"]
    spread_5d_ago = df["ma_spread_std"].shift(5)
    expanding = spread_now > spread_5d_ago

    return (full_aligned & expanding).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R12.5 : 고점 단계적 하락 (Lower High 추세)
# ─────────────────────────────────────────────────────────────────────
def filter_lower_high_trend(df: pd.DataFrame, window: int = 15) -> pd.Series:
    """
    최근 window일 최고점이 직전 window일 최고점보다 낮음.
    추세선이 우하향 → 진입 금지.
    """
    recent_high = df["High"].rolling(window, min_periods=window).max()
    prev_high = df["High"].shift(window).rolling(window, min_periods=window).max()
    return (recent_high < prev_high).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R14.5 : 데드 기법 — 역배열에서 일시 반등
# ─────────────────────────────────────────────────────────────────────
def filter_deadcat_bounce(
    df: pd.DataFrame, vol_weak_ratio: float = 0.8, near_ma_threshold: float = 0.03
) -> pd.Series:
    """
    조건:
      1) 장기 이평 역배열 상태
      2) close가 ma60 또는 ma112 근접 (반등이 이평선까지 닿음)
      3) 거래량이 약함 (20일 평균의 80% 미만)

    → 진짜 추세 전환이 아니라 일시적 반등 (속임수). 진입 금지.
    """
    inverted = df["long_term_inverted"] == 1

    near_ma60 = (df["Close"] - df["ma_60"]).abs() / df["ma_60"] < near_ma_threshold
    near_ma112 = (df["Close"] - df["ma_112"]).abs() / df["ma_112"] < near_ma_threshold
    near_ma_resistance = near_ma60 | near_ma112

    weak_volume = df["vol_ratio_20"] < vol_weak_ratio

    return (inverted & near_ma_resistance & weak_volume).astype(int)


# ─────────────────────────────────────────────────────────────────────
# 통합 함수
# ─────────────────────────────────────────────────────────────────────
def add_filters(df: pd.DataFrame) -> pd.DataFrame:
    df["filter_R3_4_ma224_overheat"] = filter_ma224_overheat(df)
    df["filter_R4_2_full_uptrend_expanding"] = filter_full_uptrend_expanding(df)
    df["filter_R12_5_lower_high_trend"] = filter_lower_high_trend(df)
    df["filter_R14_5_deadcat_bounce"] = filter_deadcat_bounce(df)

    # 통합 필터: 위 4개 중 하나라도 발화하면 1
    df["filter_any_avoid"] = (
        df[
            [
                "filter_R3_4_ma224_overheat",
                "filter_R4_2_full_uptrend_expanding",
                "filter_R12_5_lower_high_trend",
                "filter_R14_5_deadcat_bounce",
            ]
        ].max(axis=1)
    ).astype(int)
    return df
