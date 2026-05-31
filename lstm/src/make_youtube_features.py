"""
make_youtube_features.py
========================

OHLCV 데이터에서 유튜버 시그널(29개 sig_*)만 분리해 생성.

기본 피처 파이프라인(feature_pipeline_v2 + feature_v3_transform)과 별도로
유튜버 도메인 시그널만 추출하기 위한 wrapper.

사용법:
    python make_youtube_features.py \
        --data_dir ../data/processed/ohlcv \
        --ticker_list ../actually_used_230_tickers.txt \
        --output_dir ../output/youtube
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from feature_pipeline_v2 import load_universe, load_ticker_list, clean_ohlcv
from youtuber_signals import add_all_signals, analyze_signals


def main():
    p = argparse.ArgumentParser(
        description="OHLCV → 유튜버 시그널(29개) 분리 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="OHLCV parquet 디렉토리")
    p.add_argument("--ticker_list", type=str, default=None,
                   help="종목 리스트 파일 (없으면 디렉토리 내 전체)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--start_date", type=str, default=None)
    p.add_argument("--end_date", type=str, default=None)
    p.add_argument("--max_daily_change", type=float, default=0.30,
                   help="가격제한폭 (30%)")
    p.add_argument("--analyze", action="store_true",
                   help="시그널 통계 분석 출력")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 70)
    print("OHLCV → 유튜버 시그널 (29개 sig_*) 분리 생성")
    print("=" * 70)
    print(f"입력  : {data_dir}")
    print(f"출력  : {output_dir}")
    print(f"기간  : {args.start_date or '전체'} ~ {args.end_date or '최신'}")

    # 1. 로딩
    t0 = time.time()
    ticker_list = load_ticker_list(args.ticker_list) if args.ticker_list else None
    df = load_universe(data_dir, ticker_list=ticker_list)

    if args.start_date:
        df = df[df['date'] >= args.start_date]
    if args.end_date:
        df = df[df['date'] <= args.end_date]
    print(f"  기간 필터 후: {len(df):,} 행")
    print(f"  ⏱ 로딩: {time.time()-t0:.1f}s\n")

    # 2. 정제
    t0 = time.time()
    df = clean_ohlcv(
        df,
        max_daily_change=args.max_daily_change,
        drop_extreme_rows=True,
        drop_extreme_tickers_threshold=None,
    )
    print(f"  ⏱ 정제: {time.time()-t0:.1f}s\n")

    # 3. 유튜버 시그널 생성
    t0 = time.time()
    df = add_all_signals(df, verbose=True)
    print(f"  ⏱ 시그널 생성: {time.time()-t0:.1f}s\n")

    # 4. sig 컬럼만 추출
    sig_cols = sorted([c for c in df.columns if c.startswith('sig_')])
    meta_cols = ['date', 'ticker']
    ohlcv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    df_out = df[meta_cols + ohlcv_cols + sig_cols].copy()

    print("=" * 70)
    print(f"유튜버 시그널: {len(sig_cols)}개")
    print(f"출력 shape   : {df_out.shape}")
    print("=" * 70)

    # 5. 분석 (옵션)
    if args.analyze:
        analyze_signals(df_out)

    # 6. 저장
    t0 = time.time()
    df_out.to_parquet(output_dir / "df_full.parquet", index=False)

    X = df_out[sig_cols].replace([np.inf, -np.inf], np.nan)
    meta = df_out[meta_cols].copy()
    X.to_parquet(output_dir / "X.parquet", index=False)
    meta.to_parquet(output_dir / "meta.parquet", index=False)

    pd.DataFrame({'feature': sig_cols}).to_csv(
        output_dir / "feature_list.csv", index=False
    )

    print(f"\n[저장 완료] {output_dir}")
    for fname in ['df_full.parquet', 'X.parquet', 'meta.parquet', 'feature_list.csv']:
        fp = output_dir / fname
        if fp.exists():
            size_mb = fp.stat().st_size / 1024 / 1024
            print(f"  - {fname} ({size_mb:.2f} MB)")
    print(f"  ⏱ 저장: {time.time()-t0:.1f}s")

    print(f"\n총 실행 시간: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
