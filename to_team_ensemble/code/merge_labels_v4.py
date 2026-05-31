"""
merge_labels_v4.py
====================

C:\\Users\\usus0\\Downloads\\data\\processed\\labels\\*.parquet (230 종목)을
하나의 통합 라벨 데이터로 합치고 분석.

사용법:
    python merge_labels_v4.py --labels_dir <경로> --output ..\\output_v4\\labels_merged.parquet --analyze
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", type=str, required=True,
                   help="종목별 라벨 parquet들이 있는 폴더")
    p.add_argument("--output", type=str, required=True,
                   help="통합 결과 저장 경로")
    p.add_argument("--analyze", action="store_true",
                   help="라벨 분포·수익률 통계 출력")
    args = p.parse_args()

    labels_dir = Path(args.labels_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # 1. 파일 목록
    parquet_files = sorted(labels_dir.glob("*.parquet"))
    print("=" * 75)
    print(f"라벨 파일 통합")
    print("=" * 75)
    print(f"  경로: {labels_dir}")
    print(f"  파일 수: {len(parquet_files)}개")

    if len(parquet_files) == 0:
        print(f"[ERROR] {labels_dir}에 parquet 파일이 없습니다.")
        return

    # 2. 통합
    print(f"\n로딩...")
    dfs = []
    failed = []
    for i, f in enumerate(parquet_files):
        try:
            df = pd.read_parquet(f)
            df['ticker'] = f.stem  # 파일명에서 ticker 추출
            dfs.append(df)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(parquet_files)}...")
        except Exception as e:
            failed.append((f.name, str(e)))

    if failed:
        print(f"\n[WARN] 로드 실패 {len(failed)}개:")
        for name, err in failed[:5]:
            print(f"  {name}: {err}")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n  통합 shape: {merged.shape}")
    print(f"  종목 수: {merged['ticker'].nunique()}개")
    print(f"  기간: {merged['dt'].min().date()} ~ {merged['dt'].max().date()}")

    # 3. 컬럼 정리: 'date'로 통일 (우리 시스템 표준)
    merged = merged.rename(columns={'dt': 'date'})

    # 4. 라벨 분포 분석
    if args.analyze:
        print("\n" + "=" * 75)
        print("라벨 분포 분석")
        print("=" * 75)

        for label_col in ['A_outcome', 'B_outcome', 'tp1_only_label', 'tp2_only_label']:
            if label_col in merged.columns:
                print(f"\n[{label_col}]")
                vc = merged[label_col].value_counts(dropna=False).sort_index()
                total = len(merged)
                for k, v in vc.items():
                    print(f"  {k}: {v:>8,} ({v/total*100:5.2f}%)")

        # B 시나리오 자세히 (우리 목표)
        print("\n" + "=" * 75)
        print("B 시나리오 상세 (터틀 -2N + Chandelier)")
        print("=" * 75)
        for o in [-1, 0, 1]:
            sub = merged[merged['B_outcome'] == o]
            if len(sub) > 0:
                days = sub['B_holding_days'].dropna()
                ret = sub['B_return'].dropna()
                print(f"\nB_outcome = {o} (n={len(sub):,}, {len(sub)/len(merged)*100:.2f}%):")
                if len(days) > 0:
                    print(f"  보유일 — median: {days.median():.0f}d  p90: {days.quantile(0.9):.0f}d  max: {days.max():.0f}d")
                if len(ret) > 0:
                    print(f"  수익률 — mean: {ret.mean()*100:+.2f}%  median: {ret.median()*100:+.2f}%  std: {ret.std()*100:.2f}%")

        # A 시나리오 비교
        print("\n" + "=" * 75)
        print("A 시나리오 상세 (빡빡한 손절)")
        print("=" * 75)
        for o in [-1, 0, 1]:
            sub = merged[merged['A_outcome'] == o]
            if len(sub) > 0:
                days = sub['A_holding_days'].dropna()
                ret = sub['A_return'].dropna()
                print(f"\nA_outcome = {o} (n={len(sub):,}):")
                if len(days) > 0:
                    print(f"  보유일 — median: {days.median():.0f}d  p90: {days.quantile(0.9):.0f}d  max: {days.max():.0f}d")
                if len(ret) > 0:
                    print(f"  수익률 — mean: {ret.mean()*100:+.2f}%  median: {ret.median()*100:+.2f}%")

        # Expected Value 비교
        print("\n" + "=" * 75)
        print("Expected Value 비교 (모델 alpha 없이 무작위 진입 시)")
        print("=" * 75)
        for name, col_o, col_r in [('A', 'A_outcome', 'A_return'),
                                     ('B', 'B_outcome', 'B_return')]:
            if col_o in merged.columns:
                valid = merged.dropna(subset=[col_o, col_r])
                if len(valid) > 0:
                    n_win = (valid[col_o] == 1).sum()
                    n_loss = (valid[col_o] == -1).sum()
                    win_rate = n_win / len(valid)
                    loss_rate = n_loss / len(valid)
                    avg_win = valid[valid[col_o] == 1][col_r].mean() if n_win > 0 else 0
                    avg_loss = valid[valid[col_o] == -1][col_r].mean() if n_loss > 0 else 0
                    ev = win_rate * avg_win + loss_rate * avg_loss
                    print(f"\n{name} 시나리오:")
                    print(f"  Win Rate: {win_rate*100:.2f}%  Avg Win: {avg_win*100:+.2f}%")
                    print(f"  Loss Rate: {loss_rate*100:.2f}%  Avg Loss: {avg_loss*100:+.2f}%")
                    print(f"  Expected Value: {ev*100:+.4f}%")
                    if ev > 0:
                        print(f"  → 무작위 진입해도 평균 양수 ✓")
                    else:
                        print(f"  → 무작위 진입은 손실. 모델이 alpha 만들어야 함 ⚠️")

        # 연도별 분포
        print("\n" + "=" * 75)
        print("연도별 B_outcome 분포")
        print("=" * 75)
        merged['year'] = pd.to_datetime(merged['date']).dt.year
        yearly = pd.crosstab(merged['year'], merged['B_outcome'].fillna(-99), 
                              normalize='index') * 100
        print(yearly.tail(15).round(2))

        # 종목별 win rate 분포
        print("\n" + "=" * 75)
        print("종목별 B 승률 분포")
        print("=" * 75)
        per_ticker = merged.dropna(subset=['B_outcome']).groupby('ticker').apply(
            lambda g: (g['B_outcome'] == 1).sum() / len(g) * 100,
            include_groups=False
        )
        print(f"  min: {per_ticker.min():.1f}%  median: {per_ticker.median():.1f}%  max: {per_ticker.max():.1f}%")
        print(f"  최악 5: {per_ticker.nsmallest(5).round(1).to_dict()}")
        print(f"  최고 5: {per_ticker.nlargest(5).round(1).to_dict()}")

    # 5. 저장
    print(f"\n저장: {output_path}")
    merged.to_parquet(output_path, index=False)
    print(f"  최종 shape: {merged.shape}")
    print(f"  컬럼: {list(merged.columns)}")

    print(f"\n총 시간: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
