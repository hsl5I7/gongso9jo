"""
youtuber_features
=================

유튜버 매매 시스템(영매공파 외 26개 룰)을 정량 피처로 변환한 모듈.

서브 모듈
---------
- ohlcv_helpers : 공통 유틸 (이평선, 볼린저밴드, 매물대 등)
- entry_signals : 진입 신호 12개 (R2.1, R3.1, R4.1, R5.1, R5.2,
                  R9.6, R10.2, R12.1, R13.1, R14.1, R15.1, R15.2)
- filters       : 회피 필터 4개 (R3.4, R4.2, R12.5, R14.5)
- candle_patterns : 캔들 분류 5개 (R2.3, R13.4, R13.5, R13.6, R13.7)
"""

from .ohlcv_helpers import add_helpers
from .entry_signals import add_entry_signals
from .filters import add_filters
from .candle_patterns import add_candle_patterns


def add_all_youtuber_features(df):
    """
    한 종목의 OHLCV DataFrame을 받아서 26개 유튜버 룰 피처를 모두 추가한다.

    df는 Date 오름차순 정렬된 상태여야 하며 컬럼: Open, High, Low, Close, Volume

    Returns
    -------
    df_out : 원본 + 헬퍼 + 26개 피처
    """
    df = add_helpers(df)
    df = add_entry_signals(df)
    df = add_filters(df)
    df = add_candle_patterns(df)
    return df
