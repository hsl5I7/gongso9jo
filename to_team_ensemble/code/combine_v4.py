"""
combine_v4.py
====================

merge_labels_v4.py 결과(230 종목 라벨 통합) + 기존 output_v3_yt_top70 (99 피처)를
(date, ticker) 기준으로 merge.

핵심:
- label_binary 생성: B_outcome == 1 → 1, 나머지 → 0, NaN → NaN
- 기존 99 피처(top70 + 시그널) 그대로 사용
- 학습용 X, y, meta 분리 저장

사용법:
    python combine_v4.py \
        --labels ..\\output_v4\\labels_merged.parquet \
        --features ..\\output_v3_yt_top70\\df_full.parquet \
        --output_dir ..\\output_v4_features
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=str, required=True,
                   help="merge_labels_v4.py 통합 결과 (labels_merged.parquet)")
    p.add_argument("--features", type=str, required=True,
                   help="기존 99피처 (output_v3_yt_top70/df_full.parquet)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--label_source", type=str, default="B_outcome",
                   choices=["B_outcome", "A_outcome"],
                   help="어떤 라벨 컬럼으로 label_binary 만들지 (기본: B_outcome)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # 1. 라벨 데이터 로드
    print("=" * 75)
    print("데이터 로드")
    print("=" * 75)
    labels = pd.read_parquet(args.labels)
    labels['date'] = pd.to_datetime(labels['date'])
    labels['ticker'] = labels['ticker'].astype(str)
    print(f"\n[라벨] shape: {labels.shape}")
    print(f"  종목: {labels['ticker'].nunique()}")
    print(f"  기간: {labels['date'].min().date()} ~ {labels['date'].max().date()}")

    # 2. 피처 데이터 로드
    features = pd.read_parquet(args.features)
    features['date'] = pd.to_datetime(features['date'])
    features['ticker'] = features['ticker'].astype(str)
    print(f"\n[피처] shape: {features.shape}")
    print(f"  종목: {features['ticker'].nunique()}")
    print(f"  기간: {features['date'].min().date()} ~ {features['date'].max().date()}")

    # 3. 라벨에서 필요한 컬럼만 추출
    label_cols_keep = ['date', 'ticker', args.label_source]
    # 백테스트용 메타도 같이 보관
    label_meta = ['entry_dt', 'entry_price', 'stop_price',
                  f'{args.label_source[0]}_exit_dt',
                  f'{args.label_source[0]}_exit_price',
                  f'{args.label_source[0]}_holding_days',
                  f'{args.label_source[0]}_return',
                  'atr', 'tp1_threshold']
    for c in label_meta:
        if c in labels.columns:
            label_cols_keep.append(c)
    labels_slim = labels[label_cols_keep].copy()

    # 4. label_binary 생성
    print(f"\n[label_binary 생성: {args.label_source} 기반]")
    labels_slim['label_binary'] = (labels_slim[args.label_source] == 1).astype('Int8')
    # NaN은 유지
    nan_mask = labels_slim[args.label_source].isna()
    labels_slim.loc[nan_mask, 'label_binary'] = pd.NA

    vc = labels_slim['label_binary'].value_counts(dropna=False)
    total = len(labels_slim)
    print(f"  분포 (n={total:,}):")
    for k, v in vc.items():
        print(f"    {k}: {v:>8,} ({v/total*100:5.2f}%)")

    # 5. Merge (피처 기준 left join — 우리 피처 데이터에 라벨 붙임)
    print(f"\n" + "=" * 75)
    print("Merge")
    print("=" * 75)
    # features에서 기존 label 컬럼이 있으면 제거 (덮어쓰기 위함)
    drop_old = [c for c in ['label_binary', 'label_3class'] if c in features.columns]
    if drop_old:
        print(f"  기존 라벨 컬럼 제거: {drop_old}")
        features = features.drop(columns=drop_old)

    merged = features.merge(labels_slim, on=['date', 'ticker'], how='left')
    print(f"\n  merge 결과: {merged.shape}")
    print(f"  label_binary 매칭률: {(~merged['label_binary'].isna()).sum()/len(merged)*100:.2f}%")

    # 6. 컬럼 분류
    # ⚠️ days_to_event, realized_ret은 기존 v3 라벨 메타. 피처에서 제외 필수 (leakage)
    meta_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                 'label_binary', args.label_source,
                 'days_to_event', 'realized_ret']  # ← leakage 방지
    meta_cols += [c for c in label_meta if c in merged.columns]
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    print(f"\n[컬럼 분류]")
    print(f"  메타: {len(meta_cols)}개")
    print(f"  피처: {len(feature_cols)}개")
    print(f"\n  피처 카테고리 분포:")
    cat_count = {
        'sig_* (유튜버 시그널)': sum(1 for c in feature_cols if c.startswith('sig_')),
        '*_cs (CS 정규화)': sum(1 for c in feature_cols if c.endswith('_cs')),
        'inter_* (상호작용)': sum(1 for c in feature_cols if c.startswith('inter_')),
        '기타': len(feature_cols),
    }
    cat_count['기타'] -= (cat_count['sig_* (유튜버 시그널)'] +
                          cat_count['*_cs (CS 정규화)'] +
                          cat_count['inter_* (상호작용)'])
    for k, v in cat_count.items():
        print(f"    {k}: {v}")

    # 7. 결측치 확인
    print(f"\n[결측치 (상위 10개)]")
    nan_count = merged[feature_cols].isna().sum()
    nan_count = nan_count[nan_count > 0].sort_values(ascending=False)
    if len(nan_count) > 0:
        for col, cnt in nan_count.head(10).items():
            print(f"  {col:<35} {cnt:>8,} ({cnt/len(merged)*100:.2f}%)")
    else:
        print("  없음")

    # 8. 저장 (전체 + 분리)
    print(f"\n[저장]")
    # 전체
    merged.to_parquet(output_dir / "df_full.parquet", index=False)
    print(f"  df_full.parquet: {merged.shape}")

    # X (피처만)
    X = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    X.to_parquet(output_dir / "X.parquet", index=False)
    print(f"  X.parquet: {X.shape}")

    # y (라벨)
    y = merged[['label_binary', args.label_source]]
    y.to_parquet(output_dir / "y.parquet", index=False)
    print(f"  y.parquet: {y.shape}")

    # meta (날짜, 종목, 백테스트용 정보)
    meta = merged[[c for c in meta_cols if c not in [args.label_source, 'label_binary']]]
    meta.to_parquet(output_dir / "meta.parquet", index=False)
    print(f"  meta.parquet: {meta.shape}")

    # 피처 목록
    pd.DataFrame({'feature': feature_cols}).to_csv(
        output_dir / "feature_list.csv", index=False)
    print(f"  feature_list.csv ({len(feature_cols)}개)")

    print(f"\n총 시간: {time.time()-t_start:.1f}s")
    print(f"\n다음 단계: python train_walkforward_v4.py "
          f"--data_dir {output_dir} --ohlcv_dir ..\\data\\processed\\ohlcv "
          f"--output_dir ..\\results\\wf_v4_baseline")


if __name__ == "__main__":
    main()
