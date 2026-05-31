"""Walk-forward fold 구성: expanding train + 1y val + 1y test, step 1y, purge 60d."""
from __future__ import annotations

import pandas as pd


def make_walk_forward_folds(
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
    first_test_year: int = 2012,
    step_years: int = 1,
    purge_days: int = 60,
    val_test_purge_days: int | None = None,
    val_years: int = 1,
    test_years: int = 1,
) -> list[dict]:
    """각 fold:
        train: [date_min, val_start - purge_days - 1d]
        val:   [val_start, val_end]
        test:  [val_end + val_test_purge_days + 1d, test_end]

    val_test_purge_days: val 끝과 test 시작 사이 간격(라벨이 미래로 늘어지는 케이스 차단).
        None 이면 purge_days 와 동일 값 사용. test 기간이 1년에서 그만큼 짧아진다.
    """
    if val_test_purge_days is None:
        val_test_purge_days = purge_days
    folds: list[dict] = []
    year = first_test_year
    purge = pd.Timedelta(days=purge_days)
    vt_purge = pd.Timedelta(days=val_test_purge_days)
    while True:
        val_start = pd.Timestamp(year=year - val_years, month=1, day=1)
        val_end = pd.Timestamp(year=year - 1, month=12, day=31)
        test_start = val_end + vt_purge + pd.Timedelta(days=1)
        test_end = pd.Timestamp(year=year + test_years - 1, month=12, day=31)
        if test_end > date_max:
            break
        train_start = date_min
        train_end = val_start - purge - pd.Timedelta(days=1)
        folds.append({
            "train_start": train_start, "train_end": train_end,
            "val_start": val_start, "val_end": val_end,
            "test_start": test_start, "test_end": test_end,
        })
        year += step_years
    return folds
