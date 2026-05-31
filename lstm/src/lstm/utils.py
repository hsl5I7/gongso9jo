"""공통 유틸 — seed, device, 데이터 로딩, 전처리."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_general_dataset(general_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """output/general/{X, y, meta}.parquet 로딩. row alignment 보장됨 (attach_labels.py 출력)."""
    general_dir = Path(general_dir)
    X = pd.read_parquet(general_dir / "X.parquet")
    y_df = pd.read_parquet(general_dir / "y.parquet")
    meta = pd.read_parquet(general_dir / "meta.parquet")
    meta["date"] = pd.to_datetime(meta["date"])

    assert len(X) == len(y_df) == len(meta), \
        f"shape mismatch: X={len(X)} y={len(y_df)} meta={len(meta)}"

    y = y_df["label_binary"]
    return X, y, meta


def add_cs_zscore(df: pd.DataFrame, features: list[str], clip: float = 5.0) -> pd.DataFrame:
    """feature_v3_transform.add_cs_zscore 동일 로직.
    같은 date 내 종목간 z-score → ±clip 클리핑 → '_cs' 접미사로 새 컬럼 추가."""
    df = df.copy()
    available = [c for c in features if c in df.columns]
    if not available:
        return df
    grouped = df.groupby("date")[available]
    means = grouped.transform("mean")
    stds = grouped.transform("std")
    for c in available:
        z = (df[c] - means[c]) / (stds[c] + 1e-10)
        df[f"{c}_cs"] = z.clip(-clip, clip)
    return df


# binary/ordinal — cs_zscore 적용 없이 그대로 유지
KEEP_AS_IS_FEATURES = {"is_kospi"}
# 추가로 prefix 매칭으로 keep-as-is 처리할 컬럼 (예: sig_* 유튜버 시그널 binary/strength)
KEEP_AS_IS_PREFIXES = ("sig_",)


def _is_keep_as_is(col_name: str) -> bool:
    return col_name in KEEP_AS_IS_FEATURES or col_name.startswith(KEEP_AS_IS_PREFIXES)


def load_selected_features(path: Path) -> list[str]:
    """permutation_selection.py 산출물 selected_features.csv 로딩."""
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"{path}에 'feature' 컬럼 없음. cols={list(df.columns)}")
    return df["feature"].tolist()


def load_combined_dataset(
    general_dir: Path, youtube_dir: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """general (date,ticker) 기준에 youtube X를 left join.

    youtube이 general의 모든 (date,ticker) 행을 포함한다고 검증됨 → 손실 0.
    반환: (X_combined, y_general, meta_general)
        - X_combined: general 컬럼 + youtube 컬럼 (모두 raw 상태)
        - y/meta는 general 기준 그대로
    """
    X_g, y_g, meta_g = load_general_dataset(general_dir)
    X_y = pd.read_parquet(Path(youtube_dir) / "X.parquet")
    meta_y = pd.read_parquet(Path(youtube_dir) / "meta.parquet")
    meta_y["date"] = pd.to_datetime(meta_y["date"])

    df_g = pd.concat([meta_g[["date", "ticker"]].reset_index(drop=True),
                      X_g.reset_index(drop=True)], axis=1)
    df_y = pd.concat([meta_y[["date", "ticker"]].reset_index(drop=True),
                      X_y.reset_index(drop=True)], axis=1)

    # 컬럼 충돌 방지 (이론상 없음 — sig_* vs 그 외)
    yt_cols = [c for c in X_y.columns if c not in X_g.columns]
    if len(yt_cols) != len(X_y.columns):
        skipped = [c for c in X_y.columns if c in X_g.columns]
        print(f"[load_combined] general과 youtube 컬럼명 충돌 — youtube 제외: {skipped}")
    df_y = df_y[["date", "ticker"] + yt_cols]

    df = df_g.merge(df_y, on=["date", "ticker"], how="left")
    assert len(df) == len(df_g), f"left join 결과 행 수 변경: {len(df_g)} → {len(df)}"

    X_cols = [c for c in df.columns if c not in ("date", "ticker")]
    X_combined = df[X_cols].reset_index(drop=True)
    meta_combined = df[["date", "ticker"]].reset_index(drop=True)
    return X_combined, y_g.reset_index(drop=True), meta_combined


def preprocess_features(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    selected: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """X.parquet → LSTM 입력 array.

    공통 규칙 (Downloads/src/feature_v3_transform.add_cs_zscore만 사용):
    - binary (is_kospi 등 KEEP_AS_IS_FEATURES)는 그대로
    - _cs로 끝나는 컬럼은 이미 정규화됨 → 그대로
    - 그 외 numerical raw 컬럼은 add_cs_zscore 적용 (date별 z-score + clip ±5)

    selected=None (전체 모드):
        - _cs 짝 있는 원본 → drop (중복)
        - _cs 짝 없는 raw → add_cs_zscore 후 _cs 만 사용
        - 결과: 103차원 (기존 _cs 60 + 새 _cs 42 + is_kospi 1)

    selected=[…] (selected 모드, permutation_selection.py 산출물):
        - selected 리스트의 컬럼명 그대로 유지 (XGB importance 기준 보존)
        - raw 컬럼은 add_cs_zscore 적용 후 그 정규화된 값을 raw 컬럼명에 덮어씀
        - 결과: len(selected) 차원
    """
    df = pd.concat(
        [meta[["date", "ticker"]].reset_index(drop=True), X.reset_index(drop=True)],
        axis=1,
    )

    if selected is None:
        return _preprocess_full(df, X.columns.tolist())
    return _preprocess_selected(df, X.columns.tolist(), selected)


def _preprocess_full(
    df: pd.DataFrame, all_orig_cols: list[str]
) -> tuple[np.ndarray, list[str]]:
    cs_cols = [c for c in all_orig_cols if c.endswith("_cs")]
    keep_as_is = [c for c in all_orig_cols if _is_keep_as_is(c)]
    to_normalize = [
        c for c in all_orig_cols
        if not c.endswith("_cs")
        and f"{c}_cs" not in all_orig_cols
        and c not in keep_as_is
    ]
    df = add_cs_zscore(df, to_normalize, clip=5.0)
    new_cs_cols = [f"{c}_cs" for c in to_normalize]
    final_features = cs_cols + new_cs_cols + keep_as_is

    df[final_features] = df.groupby("ticker", sort=False)[final_features].ffill()
    df[final_features] = df[final_features].fillna(0.0)
    arr = df[final_features].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, final_features


def _preprocess_selected(
    df: pd.DataFrame, all_orig_cols: list[str], selected: list[str]
) -> tuple[np.ndarray, list[str]]:
    missing = [c for c in selected if c not in all_orig_cols]
    if missing:
        raise ValueError(f"selected 중 X.parquet에 없는 컬럼: {missing}")

    # raw 컬럼만 추출 (_cs 접미사 아닌 것), binary/sig_*는 제외
    raw_to_normalize = [
        c for c in selected
        if not c.endswith("_cs") and not _is_keep_as_is(c)
    ]
    df = add_cs_zscore(df, raw_to_normalize, clip=5.0)

    # raw 컬럼명에 _cs 값을 덮어씀 (selected 이름은 그대로 유지)
    for c in raw_to_normalize:
        cs_col = f"{c}_cs"
        if cs_col in df.columns:
            df[c] = df[cs_col]

    df[selected] = df.groupby("ticker", sort=False)[selected].ffill()
    df[selected] = df[selected].fillna(0.0)
    arr = df[selected].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, list(selected)
