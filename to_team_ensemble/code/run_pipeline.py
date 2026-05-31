"""
CLI 실행 스크립트: feature_pipeline_v2를 cmd에서 실행

사용법:
    # 기본 실행 (현재 폴더의 모든 .parquet)
    python run_pipeline.py --data_dir ./data

    # 옵션 지정
    python run_pipeline.py --data_dir ./data --start_date 2010-01-01 ^
        --min_turnover_mn 1000 --min_mktcap 5e10 ^
        --upper 0.15 --lower -0.035 --horizon 30 ^
        --output_dir ./output

    # 작은 종목까지 포함 (시총 필터 없이)
    python run_pipeline.py --data_dir ./data --min_mktcap 0

    # 라벨 분포만 빠르게 확인
    python run_pipeline.py --data_dir ./data --label_only

출력 파일 (output_dir 폴더에):
    X.parquet          - 피처 행렬
    y.parquet          - 라벨
    meta.parquet       - (date, ticker) 메타정보
    df_full.parquet    - 피처+라벨 통합 (분석용)
    summary.txt        - 실행 요약
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# feature_pipeline_v2.py가 같은 폴더에 있다고 가정
sys.path.insert(0, str(Path(__file__).parent))
from feature_pipeline_v2 import (
    load_universe, clean_ohlcv, estimate_mktcap, apply_universe_filter,
    build_per_ticker_features, build_market_proxy_features,
    build_cross_sectional_features, make_triple_barrier_labels_fast,
    prepare_train_data,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="주식예측 XGBoost 피처 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_dir', type=str, required=True,
                   help='Parquet 파일들이 있는 폴더 (예: ./data)')
    p.add_argument('--ticker_list', type=str, default=None,
                   help='사용할 종목코드 리스트 파일 (예: ./meta/selected_tickers.txt). '
                        '지정하면 폴더 내 파일 중 이 리스트만 사용.')
    p.add_argument('--output_dir', type=str, default='./output',
                   help='결과 저장 폴더')
    p.add_argument('--pattern', type=str, default='*.parquet',
                   help='파일 패턴')

    # 필터
    p.add_argument('--start_date', type=str, default=None,
                   help='시작 날짜 (예: 2010-01-01). 빠를수록 데이터 많지만 IMF/금융위기 포함')
    p.add_argument('--end_date', type=str, default=None,
                   help='끝 날짜 (예: 2025-12-31)')
    p.add_argument('--min_turnover_mn', type=float, default=1000,
                   help='20일 평균 거래대금 최소값 (백만원). 1000=10억원')
    p.add_argument('--min_mktcap', type=float, default=5e10,
                   help='최소 시가총액 (원). 0이면 시총 필터 미적용')
    p.add_argument('--min_history', type=int, default=250,
                   help='종목별 최소 거래일수')

    # 데이터 정제
    p.add_argument('--max_daily_change', type=float, default=0.30,
                   help='정상 일간 변동 한계 (기본 30% = 한국 가격제한폭). '
                        '초과 행은 제거됨 (액면분할 미조정 의심)')
    p.add_argument('--drop_bad_tickers', type=int, default=3,
                   help='극단변동이 N회 이상인 종목은 통째 제거 (0이면 비활성화)')

    # 타깃
    p.add_argument('--upper', type=float, default=0.15, help='익절 배리어')
    p.add_argument('--lower', type=float, default=-0.035, help='손절 배리어')
    p.add_argument('--horizon', type=int, default=30, help='시간 배리어 (일)')

    # 옵션
    p.add_argument('--label_only', action='store_true',
                   help='라벨 분포만 확인 (피처 저장 안함)')
    p.add_argument('--skip_market_proxy', action='store_true',
                   help='시장 proxy 피처 생성 스킵 (단일/소수 종목 시)')
    p.add_argument('--skip_cs_features', action='store_true',
                   help='Cross-sectional 피처 스킵')
    p.add_argument('--no_save', action='store_true',
                   help='결과 저장 안함 (테스트용)')

    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        print(f"[ERROR] data_dir 없음: {data_dir}")
        sys.exit(1)

    files = list(data_dir.glob(args.pattern))
    if not files:
        print(f"[ERROR] {data_dir}에 '{args.pattern}' 파일 없음")
        sys.exit(1)

    print("=" * 60)
    print(f"입력 폴더 : {data_dir}")
    print(f"파일 개수 : {len(files)}")
    if args.ticker_list:
        print(f"종목 리스트: {args.ticker_list}")
    print(f"기간      : {args.start_date or '전체'} ~ {args.end_date or '최신'}")
    print(f"타깃      : +{args.upper:.1%} / {args.lower:.1%} / {args.horizon}일")
    print(f"필터      : 거래대금≥{args.min_turnover_mn:,.0f}백만원, "
          f"시총≥{args.min_mktcap:.0e}원")
    print("=" * 60)

    # --- 1. 로딩 ---
    t0 = time.time()
    ticker_list = None
    if args.ticker_list:
        from feature_pipeline_v2 import load_ticker_list
        ticker_list = load_ticker_list(args.ticker_list)

    df = load_universe(data_dir, pattern=args.pattern, ticker_list=ticker_list)
    df = df.rename(columns={'dt': 'date'}) if 'dt' in df.columns else df
    df['date'] = pd.to_datetime(df['date'])

    if args.start_date:
        df = df[df['date'] >= args.start_date]
    if args.end_date:
        df = df[df['date'] <= args.end_date]
    print(f"  → 기간 필터 후: {len(df):,} 행")
    print(f"  ⏱ 로딩: {time.time()-t0:.1f}s\n")

    # --- 2. 정제 ---
    t0 = time.time()
    df = clean_ohlcv(
        df,
        max_daily_change=args.max_daily_change,
        drop_extreme_rows=True,
        drop_extreme_tickers_threshold=(args.drop_bad_tickers if args.drop_bad_tickers > 0 else None),
    )
    df = estimate_mktcap(df)
    print(f"  ⏱ 정제: {time.time()-t0:.1f}s\n")

    # --- 3. 필터 ---
    t0 = time.time()
    df = apply_universe_filter(
        df,
        min_turnover_20d_mn=args.min_turnover_mn,
        min_mktcap=args.min_mktcap if args.min_mktcap > 0 else None,
        min_history_days=args.min_history,
    )
    print(f"  ⏱ 필터: {time.time()-t0:.1f}s\n")

    if len(df) == 0:
        print("[ERROR] 필터 후 데이터 없음. 필터 기준을 낮춰보세요.")
        print("  - --min_turnover_mn 0 --min_mktcap 0")
        sys.exit(1)

    # --- 4. 피처 ---
    t0 = time.time()
    df = build_per_ticker_features(df)
    print(f"  ⏱ 종목별 피처: {time.time()-t0:.1f}s\n")

    if not args.skip_market_proxy and df['ticker'].nunique() > 5:
        t0 = time.time()
        df = build_market_proxy_features(df)
        print(f"  ⏱ 시장 proxy: {time.time()-t0:.1f}s\n")

    if not args.skip_cs_features and df['ticker'].nunique() > 5:
        t0 = time.time()
        df = build_cross_sectional_features(df)
        print(f"  ⏱ Cross-sectional: {time.time()-t0:.1f}s\n")

    # sector/market 메타 정보 있으면 sector 피처 추가
    if 'sector' in df.columns or 'market' in df.columns:
        from feature_pipeline_v2 import build_sector_features
        t0 = time.time()
        df = build_sector_features(df)
        print(f"  ⏱ Sector 피처: {time.time()-t0:.1f}s\n")

    # --- 5. 라벨 ---
    t0 = time.time()
    df = make_triple_barrier_labels_fast(
        df, upper=args.upper, lower=args.lower, horizon=args.horizon
    )
    print(f"  ⏱ 라벨링: {time.time()-t0:.1f}s\n")

    # --- 6. 라벨 분포 분석 ---
    print("=" * 60)
    print("라벨 분포")
    print("=" * 60)
    dist = df['label_3class'].value_counts(dropna=False).sort_index()
    total = dist.sum()
    label_names = {0.0: '손절 (-3.5% 먼저)', 1.0: '익절 (+15% 먼저)', 2.0: '만기 청산'}
    for k, v in dist.items():
        name = label_names.get(k, 'NaN (미래 부족)')
        print(f"  {name:25s}: {v:>10,} ({v/total*100:>5.2f}%)")
    print(f"\n  양성 비율 (익절): {df['label_binary'].mean():.4f}")

    # 시점별 양성 비율 (분석용, df에 추가하지 않음)
    _year = df['date'].dt.year
    yearly = df.assign(_year=_year).groupby('_year')['label_binary'].agg(['mean', 'count']).round(4)
    yearly.columns = ['양성비율', '샘플수']
    print(f"\n[연도별 양성 비율]")
    print(yearly.to_string())

    # 평균 days_to_event
    if 'days_to_event' in df.columns:
        pos = df[df['label_3class'] == 1]
        neg = df[df['label_3class'] == 0]
        print(f"\n[평균 이벤트 도달일]")
        print(f"  익절 (+15%): {pos['days_to_event'].mean():.1f}일")
        print(f"  손절 (-3.5%): {neg['days_to_event'].mean():.1f}일")

    if args.label_only:
        print(f"\n[label_only 모드] 피처 저장 스킵")
        print(f"총 실행 시간: {time.time()-t_start:.1f}s")
        return

    # --- 7. X, y 분리 ---
    t0 = time.time()
    X, y, meta = prepare_train_data(df)
    print(f"\n  ⏱ X/y 분리: {time.time()-t0:.1f}s")

    print("=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"  X: {X.shape}  ({X.shape[1]}개 피처)")
    print(f"  y: {y.shape}  (양성 비율 {y.mean():.4f})")
    print(f"  결측치(X): {X.isna().sum().sum():,} / {X.size:,} "
          f"({X.isna().sum().sum()/X.size*100:.2f}%)")

    # --- 8. 저장 ---
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        X.to_parquet(output_dir / 'X.parquet', index=False)
        y.to_frame('label_binary').to_parquet(output_dir / 'y.parquet', index=False)
        meta.to_parquet(output_dir / 'meta.parquet', index=False)

        # 통합 파일 (분석용)
        df_save = df.drop(columns=['year'], errors='ignore')
        df_save.to_parquet(output_dir / 'df_full.parquet', index=False)

        # 요약 텍스트
        with open(output_dir / 'summary.txt', 'w', encoding='utf-8') as f:
            f.write(f"실행 시간: {pd.Timestamp.now()}\n")
            f.write(f"입력 폴더: {data_dir}\n")
            f.write(f"파일 수: {len(files)}\n")
            f.write(f"기간: {df['date'].min()} ~ {df['date'].max()}\n")
            f.write(f"종목 수: {df['ticker'].nunique()}\n")
            f.write(f"총 행수: {len(df):,}\n")
            f.write(f"\n타깃: +{args.upper:.1%} / {args.lower:.1%} / {args.horizon}일\n")
            f.write(f"양성 비율: {df['label_binary'].mean():.4f}\n")
            f.write(f"\nX shape: {X.shape}\n")
            f.write(f"y shape: {y.shape}\n")
            f.write(f"\n피처 목록:\n")
            for c in X.columns:
                f.write(f"  - {c}\n")

        print(f"\n  ⏱ 저장: {time.time()-t0:.1f}s")
        print(f"\n[저장 완료] {output_dir}")
        for fname in ['X.parquet', 'y.parquet', 'meta.parquet',
                      'df_full.parquet', 'summary.txt']:
            fp = output_dir / fname
            size_mb = fp.stat().st_size / 1024 / 1024
            print(f"  - {fname} ({size_mb:.1f} MB)")

    print(f"\n총 실행 시간: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
