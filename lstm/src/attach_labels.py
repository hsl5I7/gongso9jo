"""
attach_labels.py
================

data/processed/labels/ 의 Turtle Trading 라벨(B_outcome 계열)을
output/general/, output/youtube/ 의 데이터에 (date, ticker) 기준으로 join.

B_outcome 정의 (Faith 2003 + Le Beau&Lucas 1992 Chandelier):
  -1 = 손절 (2 ATR 이탈)
  +1 = 익절 (Chandelier TP2: 누적최고가 - 3 ATR 이탈)
   0 = 미도달 (시계열 끝까지 둘 다 미발생)

매핑:
  label_3class:  -1→0(손절), +1→1(익절), 0→2(미도달)
  label_binary:  +1→1, 그 외→0

사용법:
    python attach_labels.py \
        --labels_dir ../data/processed/labels \
        --ticker_list ../actually_used_230_tickers.txt \
        --targets ../output/general ../output/youtube
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


KEEP_LABEL_COLS = ["B_outcome", "B_return", "B_holding_days"]


def load_ticker_list(path: Path) -> list[str]:
    tickers = []
    with open(path) as f:
        for line in f:
            t = line.strip()
            if t:
                tickers.append(t)
    return tickers


def load_all_labels(labels_dir: Path, tickers: list[str] | None = None) -> pd.DataFrame:
    files = sorted(labels_dir.glob("*.parquet"))
    if tickers is not None:
        tset = set(tickers)
        files = [f for f in files if f.stem in tset]
        missing = tset - {f.stem for f in files}
        if missing:
            print(f"  [WARN] {len(missing)}개 종목 라벨 파일 없음: {list(missing)[:5]}...")

    print(f"  라벨 파일 로드: {len(files)}개 종목")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp, columns=["dt"] + KEEP_LABEL_COLS)
        df["ticker"] = fp.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"dt": "date"})
    df["date"] = pd.to_datetime(df["date"])

    # label_3class / label_binary 생성
    b = df["B_outcome"].astype("Float64")
    label_3class = pd.Series(pd.NA, index=df.index, dtype="Int8")
    label_3class[b == -1] = 0
    label_3class[b == 1] = 1
    label_3class[b == 0] = 2
    df["label_3class"] = label_3class

    label_binary = pd.Series(pd.NA, index=df.index, dtype="Int8")
    label_binary[b == 1] = 1
    label_binary[(b == -1) | (b == 0)] = 0
    df["label_binary"] = label_binary

    return df[["date", "ticker"] + KEEP_LABEL_COLS + ["label_3class", "label_binary"]]


def attach_to_target(target_dir: Path, labels: pd.DataFrame, verbose: bool = True):
    df_full_path = target_dir / "df_full.parquet"
    if not df_full_path.exists():
        print(f"  [SKIP] {df_full_path} 없음")
        return

    print(f"\n[{target_dir}]")
    df_full = pd.read_parquet(df_full_path)
    df_full["date"] = pd.to_datetime(df_full["date"])
    before_shape = df_full.shape

    # 기존 라벨/Triple Barrier 흔적 제거
    drop_cols = [c for c in df_full.columns if c in
                 {"label_3class", "label_binary", "days_to_event", "realized_ret",
                  "B_outcome", "B_return", "B_holding_days"}]
    if drop_cols:
        print(f"  기존 라벨 컬럼 제거: {drop_cols}")
        df_full = df_full.drop(columns=drop_cols)

    # merge
    df_full = df_full.merge(labels, on=["date", "ticker"], how="left")
    print(f"  shape: {before_shape} → {df_full.shape}")

    # 분포
    if verbose:
        dist = df_full["label_3class"].value_counts(dropna=False).sort_index()
        print(f"  label_3class 분포:")
        names = {0: "손절(-1)", 1: "익절(+1)", 2: "미도달(0)", pd.NA: "NaN"}
        for k, v in dist.items():
            name = names.get(k, "NaN")
            pct = v / len(df_full) * 100
            print(f"    {name:14s}: {v:>10,} ({pct:5.2f}%)")
        ymean = df_full["label_binary"].dropna().astype(float).mean()
        print(f"  label_binary 양성 비율: {ymean:.4f}")

    # 저장
    df_full.to_parquet(df_full_path, index=False)
    print(f"  → df_full.parquet 갱신")

    # y.parquet 갱신 (meta와 row alignment 맞춤)
    meta_path = target_dir / "meta.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        meta["date"] = pd.to_datetime(meta["date"])
        label_only = df_full[["date", "ticker", "label_3class", "label_binary"]]
        y_aligned = meta.merge(label_only, on=["date", "ticker"], how="left")[
            ["label_3class", "label_binary"]
        ]
        y_aligned.to_parquet(target_dir / "y.parquet", index=False)
        print(f"  → y.parquet 갱신 (행 수 {len(y_aligned):,}, meta/X와 alignment)")
    else:
        y_df = df_full[["label_3class", "label_binary"]].copy()
        y_df.to_parquet(target_dir / "y.parquet", index=False)
        print(f"  → y.parquet 갱신 (행 수 {len(y_df):,})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", type=str, required=True)
    p.add_argument("--ticker_list", type=str, default=None)
    p.add_argument("--targets", type=str, nargs="+", required=True,
                   help="라벨을 join할 대상 디렉토리 (예: output/general output/youtube)")
    args = p.parse_args()

    t_start = time.time()
    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir 없음: {labels_dir}")

    print("=" * 70)
    print("Turtle 라벨 join (B_outcome 계열)")
    print("=" * 70)

    tickers = load_ticker_list(Path(args.ticker_list)) if args.ticker_list else None
    print(f"종목 수: {len(tickers) if tickers else '전체'}")

    labels = load_all_labels(labels_dir, tickers)
    print(f"통합 라벨 shape: {labels.shape}")
    print(f"  기간: {labels['date'].min().date()} ~ {labels['date'].max().date()}")

    for tgt in args.targets:
        attach_to_target(Path(tgt), labels)

    print(f"\n총 시간: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
