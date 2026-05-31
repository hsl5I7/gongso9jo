"""
combine_features.py
====================

Permutation importance로 선정된 top70 피처 + youtuber 29개 시그널을 결합.

입력:
- output_v3_yt/df_full.parquet : 시그널 29개 포함 (전체 162 v3 + 29 sig = 191 컬럼)
- results/perm_sel/selected_features.csv : top70 피처 목록

출력:
- output_v3_yt_top70/df_full.parquet : top70 + 29 sig = 99 피처
- output_v3_yt_top70/X.parquet
- output_v3_yt_top70/y.parquet, meta.parquet

사용법:
    python combine_features.py --signal_data ..\output_v3_yt\df_full.parquet ^
        --top70_csv ..\results\perm_sel\selected_features.csv ^
        --output_dir ..\output_v3_yt_top70
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signal_data", type=str, required=True,
                   help="시그널 추가된 df_full.parquet (output_v3_yt)")
    p.add_argument("--top70_csv", type=str, required=True,
                   help="Permutation selected_features.csv")
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print("=" * 70)
    print("데이터 로드")
    print("=" * 70)
    df = pd.read_parquet(args.signal_data)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  shape: {df.shape}")
    print(f"  컬럼 수: {df.shape[1]}")

    # top70 피처 로드
    top70_df = pd.read_csv(args.top70_csv)
    if 'feature' in top70_df.columns:
        top70_features = top70_df['feature'].tolist()
    else:
        top70_features = top70_df.iloc[:, 0].tolist()
    print(f"\n  top70 피처: {len(top70_features)}개 로드")

    # 시그널 컬럼 추출
    sig_cols = [c for c in df.columns if c.startswith('sig_')]
    print(f"  시그널 컬럼: {len(sig_cols)}개")

    # 메타 컬럼
    meta_cols = ['date', 'ticker', 'label_3class', 'label_binary',
                 'days_to_event', 'realized_ret']
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    keep_cols = []
    for c in meta_cols + ohlcv_cols:
        if c in df.columns:
            keep_cols.append(c)

    # 사용할 피처 확정
    # top70 중 데이터에 있는 것만
    available_top70 = [c for c in top70_features if c in df.columns]
    missing_top70 = set(top70_features) - set(available_top70)
    if missing_top70:
        print(f"\n  [WARN] top70 중 누락 피처: {missing_top70}")

    final_features = available_top70 + sig_cols
    print(f"\n  최종 피처: {len(final_features)}개")
    print(f"    - top70 (Permutation): {len(available_top70)}")
    print(f"    - 유튜버 시그널: {len(sig_cols)}")

    # 피처 카테고리 분석
    print(f"\n[유튜버 시그널 그룹별]")
    groups = {
        'A. 역배열 반등 (영매공파)': [c for c in sig_cols if c.startswith('sig_yo') or c in
                                       ['sig_maejip', 'sig_gonguri', 'sig_paran', 'sig_ma112',
                                        'sig_yg_all', 'sig_yg_strength', 'sig_yokbae']],
        'B. 정배열 눌림목': [c for c in sig_cols if c.startswith('sig_pullback')],
        'C. 수렴 후 돌파': [c for c in sig_cols if c.startswith('sig_squeeze')],
        'D. 추세선 변곡': [c for c in sig_cols if c.startswith('sig_trend')],
        'E. 종가+거래량 돌파': [c for c in sig_cols if c.startswith('sig_close')],
    }
    for name, cols in groups.items():
        print(f"  {name:<30} {len(cols)}개")

    # 데이터 정리
    df_out = df[keep_cols + final_features].copy()
    print(f"\n[출력 데이터]")
    print(f"  shape: {df_out.shape}")
    print(f"  컬럼: {df_out.shape[1]} (메타 {len(keep_cols)} + 피처 {len(final_features)})")

    # 결측치 확인
    nan_features = df_out[final_features].isna().sum()
    nan_features = nan_features[nan_features > 0].sort_values(ascending=False)
    if len(nan_features) > 0:
        print(f"\n[결측치가 있는 피처 (상위 10개)]")
        print(nan_features.head(10).to_string())

    # 저장
    print(f"\n[저장]")
    df_out.to_parquet(output_dir / "df_full.parquet", index=False)
    print(f"  df_full.parquet: {output_dir / 'df_full.parquet'}")

    # X, y, meta 분리 저장
    X = df_out[final_features].replace([np.inf, -np.inf], np.nan)
    y_3class = df_out['label_3class'].fillna(-1).astype(int) if 'label_3class' in df_out.columns else None
    meta = df_out[['date', 'ticker']]

    X.to_parquet(output_dir / "X.parquet", index=False)
    print(f"  X.parquet: {X.shape}")

    if y_3class is not None:
        y_binary = (df_out['label_3class'] == 2).astype('Int64')
        pd.DataFrame({'label_3class': y_3class, 'label_binary': y_binary}).to_parquet(
            output_dir / "y.parquet", index=False
        )
        print(f"  y.parquet: ({len(y_3class)},)")

    meta.to_parquet(output_dir / "meta.parquet", index=False)
    print(f"  meta.parquet: {meta.shape}")

    # 피처 목록도 저장
    pd.DataFrame({'feature': final_features}).to_csv(
        output_dir / "feature_list.csv", index=False
    )
    print(f"  feature_list.csv ({len(final_features)}개)")

    print(f"\n총 시간: {time.time()-t_start:.1f}s")
    print(f"\n다음 단계:")
    print(f"  1. Optuna 튜닝:")
    print(f"     python tune_hyperparameters.py --data_dir {output_dir} \\")
    print(f"         --ohlcv_dir ..\\data\\processed\\ohlcv \\")
    print(f"         --output_dir ..\\results\\tune_99 --n_trials 30")
    print(f"  2. 최종 학습:")
    print(f"     python train_walkforward_v2.py --data_dir {output_dir} \\")
    print(f"         --ohlcv_dir ..\\data\\processed\\ohlcv \\")
    print(f"         --output_dir ..\\results\\wf_v2_99")


if __name__ == "__main__":
    main()
