"""
candle_patterns.py
==================

캔들 형태 분류 5개.

- R2.3   : 매집봉 (영상 2)
- R13.4  : 장대양봉 = 분석 리셋 (영상 13)
- R13.5  : 장대음봉 = 추세 경고 (영상 13)
- R13.6  : 갭상승 (영상 13)
- R13.7  : 도지/흑삼병 등 특수 캔들 (영상 13)

prefix 'pat_' 로 통일.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# R2.3 : 매집봉
# ─────────────────────────────────────────────────────────────────────
def pat_accumulation_candle(
    df: pd.DataFrame, vol_th: float = 2.0, body_th: float = 0.4
) -> pd.Series:
    """
    매집봉:
      - 거래량 spike (20일 평균의 2배 이상)
      - 큰 양봉 (body_ratio_abs > 0.4)
      - 양봉
    """
    return (
        (df["vol_ratio_20"] > vol_th) &
        (df["body_ratio_abs"] > body_th) &
        (df["is_bullish"] == 1)
    ).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R13.4 : 장대양봉 (Huge bullish candle)
# ─────────────────────────────────────────────────────────────────────
def pat_huge_bullish(df: pd.DataFrame, change_th: float = 0.05) -> pd.Series:
    """
    장대양봉:
      - 양봉
      - 종가 기준 전일 대비 +5% 이상 상승
      - body가 종가 대비 5% 이상
    """
    return (
        (df["is_bullish"] == 1) &
        (df["change_pct"] > change_th) &
        (df["body_pct_of_close"] > change_th)
    ).astype(int)


def pat_days_since_huge_bullish(df: pd.DataFrame) -> pd.Series:
    """
    가장 최근 장대양봉 이후 며칠 지났는지.
    한 번 발생하면 0, 다음 날 1, 2, ... 식으로 증가.
    "분석 리셋 트리거" 시점부터의 경과일.
    """
    huge = df["pat_R13_4_huge_bullish"] if "pat_R13_4_huge_bullish" in df.columns else pat_huge_bullish(df)
    # group counter: 1일 때 reset
    days = []
    counter = np.nan
    for v in huge.values:
        if v == 1:
            counter = 0
        elif not np.isnan(counter):
            counter = counter + 1
        days.append(counter)
    return pd.Series(days, index=df.index, name="pat_days_since_huge_bullish")


# ─────────────────────────────────────────────────────────────────────
# R13.5 : 장대음봉
# ─────────────────────────────────────────────────────────────────────
def pat_huge_bearish(df: pd.DataFrame, change_th: float = 0.05) -> pd.Series:
    """
    장대음봉:
      - 음봉
      - 종가 기준 전일 대비 -5% 이상 하락
      - body가 종가 대비 5% 이상
    """
    return (
        (df["is_bearish"] == 1) &
        (df["change_pct"] < -change_th) &
        (df["body_pct_of_close"] > change_th)
    ).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R13.6 : 갭상승
# ─────────────────────────────────────────────────────────────────────
def pat_gap_up(df: pd.DataFrame, gap_th: float = 0.03) -> pd.Series:
    """
    오늘 시초가가 전일 종가 대비 +3% 이상 갭상승.
    """
    gap = (df["Open"] / df["Close"].shift(1) - 1)
    return (gap > gap_th).astype(int)


def pat_gap_down(df: pd.DataFrame, gap_th: float = 0.03) -> pd.Series:
    """전일 종가 대비 -3% 이상 갭하락."""
    gap = (df["Open"] / df["Close"].shift(1) - 1)
    return (gap < -gap_th).astype(int)


# ─────────────────────────────────────────────────────────────────────
# R13.7 : 도지 / 흑삼병
# ─────────────────────────────────────────────────────────────────────
def pat_doji(df: pd.DataFrame, body_th: float = 0.1) -> pd.Series:
    """도지: body_ratio_abs가 매우 작음 (시가 ≈ 종가)."""
    return (df["body_ratio_abs"] < body_th).astype(int)


def pat_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """흑삼병: 3일 연속 의미있는 음봉 (body_ratio_abs > 0.5)."""
    strong_bear = (df["is_bearish"] == 1) & (df["body_ratio_abs"] > 0.5)
    return (
        strong_bear &
        strong_bear.shift(1).fillna(False) &
        strong_bear.shift(2).fillna(False)
    ).astype(int)


def pat_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """홍삼병: 3일 연속 의미있는 양봉."""
    strong_bull = (df["is_bullish"] == 1) & (df["body_ratio_abs"] > 0.5)
    return (
        strong_bull &
        strong_bull.shift(1).fillna(False) &
        strong_bull.shift(2).fillna(False)
    ).astype(int)


# ─────────────────────────────────────────────────────────────────────
# 통합 함수
# ─────────────────────────────────────────────────────────────────────
def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df["pat_R2_3_accumulation"] = pat_accumulation_candle(df)
    df["pat_R13_4_huge_bullish"] = pat_huge_bullish(df)
    df["pat_days_since_huge_bullish"] = pat_days_since_huge_bullish(df)
    df["pat_R13_5_huge_bearish"] = pat_huge_bearish(df)
    df["pat_R13_6_gap_up"] = pat_gap_up(df)
    df["pat_gap_down"] = pat_gap_down(df)
    df["pat_R13_7_doji"] = pat_doji(df)
    df["pat_three_black_crows"] = pat_three_black_crows(df)
    df["pat_three_white_soldiers"] = pat_three_white_soldiers(df)
    return df
