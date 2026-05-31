"""sliding window Dataset — ticker 경계 안에서만 lookback 잘라냄."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_ticker_index(meta: pd.DataFrame) -> tuple[dict, dict, np.ndarray]:
    """meta는 (ticker, date)로 정렬되어 있다고 가정.
    return: (ticker_starts, ticker_ends, starts_per_row)
    """
    n = len(meta)
    ticker_arr = meta["ticker"].to_numpy()
    change_idx = np.where(ticker_arr[1:] != ticker_arr[:-1])[0] + 1
    boundaries = np.concatenate(([0], change_idx, [n]))

    ticker_starts: dict[str, int] = {}
    ticker_ends: dict[str, int] = {}
    starts_per_row = np.empty(n, dtype=np.int64)
    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        t = ticker_arr[s]
        ticker_starts[t] = s
        ticker_ends[t] = e
        starts_per_row[s:e] = s
    return ticker_starts, ticker_ends, starts_per_row


def build_valid_indices(
    meta: pd.DataFrame,
    y: pd.Series,
    starts_per_row: np.ndarray,
    lookback: int,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
) -> np.ndarray:
    """조건: date in [date_min, date_max] AND y not NaN AND row-pos in ticker >= lookback-1."""
    dates = meta["date"].to_numpy()
    label_valid = y.notna().to_numpy()

    dmin = np.datetime64(pd.Timestamp(date_min).to_datetime64())
    dmax = np.datetime64(pd.Timestamp(date_max).to_datetime64())
    date_mask = (dates >= dmin) & (dates <= dmax)

    pos_in_ticker = np.arange(len(meta), dtype=np.int64) - starts_per_row
    pos_mask = pos_in_ticker >= (lookback - 1)

    valid = date_mask & label_valid & pos_mask
    return np.where(valid)[0].astype(np.int64)


class WindowDataset(Dataset):
    """X_arr이 (N, F)로 ticker-date 정렬되어 있을 때, valid_idx 각 t에 대해
    [t-lookback+1, t] 시퀀스와 label y[t] 반환."""

    def __init__(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        valid_indices: np.ndarray,
        lookback: int,
    ):
        assert X_arr.dtype == np.float32, f"X dtype={X_arr.dtype}"
        self.X = X_arr
        self.y = y_arr.astype(np.float32, copy=False)
        self.valid_idx = valid_indices
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int):
        t = int(self.valid_idx[idx])
        seq = self.X[t - self.lookback + 1 : t + 1]
        label = self.y[t]
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.float32)
