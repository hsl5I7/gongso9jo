#!/usr/bin/env python3
# Turtle Trading Rules 기반 손절 + 두 익절 전략 라벨링.
# PDF: Faith (2003) "The Original Turtle Trading Rules"
# PDF 외 출처: Le Beau & Lucas (1992) Chandelier Exit

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 상수 (§8 명시)
ATR_PERIOD = 20          # Wilder ATR 기간 — PDF p.13
LOW_LOOKBACK = 10        # TP1 직전 N일 일중 최저 — PDF p.26
CHANDELIER_MULT = 3.0    # TP2 Chandelier 배수 — PDF 외 (Le Beau & Lucas 1992)
STOP_MULT = 2.0          # 손절 ATR 배수 — PDF p.22


def load_data(src: Path) -> pd.DataFrame:
    ext = src.suffix.lower()
    if ext == ".parquet":
        try:
            df = pd.read_parquet(src)
        except Exception:
            # pyarrow가 일부 파일을 못 읽는 경우 fastparquet로 폴백
            df = pd.read_parquet(src, engine="fastparquet")
    elif ext == ".csv":
        df = pd.read_csv(src)
    else:
        raise ValueError(f"지원하지 않는 확장자: {ext} (parquet/csv만 허용)")

    required = ["dt", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    # §9.1 사용자 결정: NaN drop
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] 필수 컬럼 NaN 행 {dropped}개를 drop 했습니다.", file=sys.stderr)

    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float64")
    df["volume"] = df["volume"].astype("int64")
    if "tradeable" in df.columns:
        df["tradeable"] = df["tradeable"].astype(bool)
    return df


def compute_tr(df: pd.DataFrame) -> np.ndarray:
    # PDF p.13: TR = max(H-L, |H-PDC|, |PDC-L|).
    # 인덱스 0은 PDC 부재 → H-L (ATR 초기 평균에선 제외)
    n = len(df)
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    tr = np.empty(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = h[0] - lo[0]
    for i in range(1, n):
        pdc = c[i - 1]
        tr[i] = max(h[i] - lo[i], abs(h[i] - pdc), abs(pdc - lo[i]))
    return tr


def compute_atr(tr: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    # PDF p.13: 초기 ATR = TR 20개의 단순평균(인덱스 1~20),
    #          이후 ATR[t] = (19*ATR[t-1] + TR[t]) / 20  (Wilder RMA)
    # pandas.ewm 사용 금지 — α가 다르다.
    n = len(tr)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return atr
    atr[period] = float(np.mean(tr[1 : period + 1]))
    for t in range(period + 1, n):
        atr[t] = ((period - 1) * atr[t - 1] + tr[t]) / period
    return atr


def compute_tp1_threshold(low: np.ndarray, lookback: int = LOW_LOOKBACK) -> np.ndarray:
    # 직전 lookback일의 일중 최저 (현재 행 미포함) — PDF p.26
    n = len(low)
    thr = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback, n):
        thr[i] = float(np.min(low[i - lookback : i]))
    return thr


def _exit_with_gap(open_j: float, threshold: float) -> float:
    # PDF p.18 진입 갭 규칙을 청산에 확장 적용:
    # 시가가 이미 임계값 너머로 벌어졌으면 시가 체결, 아니면 임계값 체결
    return float(open_j) if open_j < threshold else float(threshold)


def _outcome_from_return(ret: float) -> int:
    # 정책: 수익률 > 0 → +1, 수익률 < 0 → -1, 수익률 == 0 → -1 (보수적)
    return 1 if ret > 0 else -1


def simulate_a(t, stop_price, tp1_thr, low, open_, entry_price):
    # 시나리오 A: 손절 vs TP1 — outcome은 수익률 기반, mechanism은 별도 반환
    n = len(low)
    for j in range(t + 1, n):
        thr_j = tp1_thr[j]
        stop_hit = low[j] <= stop_price  # PDF p.21 "traded at the stop" → <=
        tp_hit = (not np.isnan(thr_j)) and (low[j] < thr_j)  # PDF p.19 strict break → <
        if stop_hit:
            # 같은 날 동시 발생 시 손절 우선 (§4 보수적 처리)
            exit_px = _exit_with_gap(open_[j], stop_price)
            ret = (exit_px - entry_price) / entry_price
            return _outcome_from_return(ret), j, exit_px, "stop"
        if tp_hit:
            exit_px = _exit_with_gap(open_[j], thr_j)
            ret = (exit_px - entry_price) / entry_price
            return _outcome_from_return(ret), j, exit_px, "tp1"
    return 0, -1, np.nan, "none"


def simulate_b(t, stop_price, atr, low, high, open_, entry_price):
    # 시나리오 B: 손절 vs TP2 (Chandelier, PDF 외) — outcome은 수익률 기반
    # TP2[j] = max(High[t+1 : j+1]) - 3 * atr[j]
    n = len(low)
    highest = -np.inf
    for j in range(t + 1, n):
        if high[j] > highest:
            highest = float(high[j])
        if np.isnan(atr[j]):
            tp_hit = False
            tp2_j = np.nan
        else:
            tp2_j = highest - CHANDELIER_MULT * atr[j]
            tp_hit = low[j] < tp2_j
        stop_hit = low[j] <= stop_price
        if stop_hit:
            exit_px = _exit_with_gap(open_[j], stop_price)
            ret = (exit_px - entry_price) / entry_price
            return _outcome_from_return(ret), j, exit_px, "stop"
        if tp_hit:
            exit_px = _exit_with_gap(open_[j], tp2_j)
            ret = (exit_px - entry_price) / entry_price
            return _outcome_from_return(ret), j, exit_px, "tp2"
    return 0, -1, np.nan, "none"


def simulate_tp1_only(t, tp1_thr, low, open_):
    # 손절 무시, TP1만
    n = len(low)
    for j in range(t + 1, n):
        thr_j = tp1_thr[j]
        if np.isnan(thr_j):
            continue
        if low[j] < thr_j:
            return 1, j, _exit_with_gap(open_[j], thr_j)
    return 0, -1, np.nan


def simulate_tp2_only(t, atr, high, low, open_):
    # 손절 무시, TP2(Chandelier)만
    n = len(low)
    highest = -np.inf
    for j in range(t + 1, n):
        if high[j] > highest:
            highest = float(high[j])
        if np.isnan(atr[j]):
            continue
        tp2_j = highest - CHANDELIER_MULT * atr[j]
        if low[j] < tp2_j:
            return 1, j, _exit_with_gap(open_[j], tp2_j)
    return 0, -1, np.nan


def label(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    dt = df["dt"].to_numpy()
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    if "tradeable" in df.columns:
        tradeable = df["tradeable"].to_numpy(dtype=bool)
    else:
        # 지수 등 tradeable 컬럼이 없는 경우: 사용자 결정 (b)는 종목 OHLCV용,
        # 지수는 모두 진입 가능으로 간주 (사용자 결정)
        tradeable = np.ones(n, dtype=bool)

    tr = compute_tr(df)
    atr = compute_atr(tr, ATR_PERIOD)
    tp1_thr = compute_tp1_threshold(low, LOW_LOOKBACK)

    nat = np.datetime64("NaT")
    entry_dt_arr = np.full(n, nat, dtype="datetime64[ns]")
    entry_price = np.full(n, np.nan)
    stop_price = np.full(n, np.nan)
    stop_dist = np.full(n, np.nan)

    A_outcome = np.full(n, np.nan)
    A_exit_dt = np.full(n, nat, dtype="datetime64[ns]")
    A_exit_p = np.full(n, np.nan)
    A_hold = np.full(n, np.nan)
    A_return = np.full(n, np.nan)
    A_exit_mechanism = np.full(n, None, dtype=object)  # 'stop'/'tp1'/'none', 진입 불가는 None

    B_outcome = np.full(n, np.nan)
    B_exit_dt = np.full(n, nat, dtype="datetime64[ns]")
    B_exit_p = np.full(n, np.nan)
    B_hold = np.full(n, np.nan)
    B_return = np.full(n, np.nan)
    B_chand_at_entry = np.full(n, np.nan)
    B_exit_mechanism = np.full(n, None, dtype=object)  # 'stop'/'tp2'/'none', 진입 불가는 None

    tp1_only_label = np.full(n, np.nan)
    tp1_only_exit_dt = np.full(n, nat, dtype="datetime64[ns]")
    tp1_only_hold = np.full(n, np.nan)
    tp1_only_ret = np.full(n, np.nan)

    tp2_only_label = np.full(n, np.nan)
    tp2_only_exit_dt = np.full(n, nat, dtype="datetime64[ns]")
    tp2_only_hold = np.full(n, np.nan)
    tp2_only_ret = np.full(n, np.nan)

    for t in range(n):
        # 진입 제외 조건 (§3)
        if np.isnan(atr[t]):
            continue
        if t + 1 >= n:
            continue
        # 사용자 결정 (b): tradeable[t+1] == False면 진입 불가
        if not bool(tradeable[t + 1]):
            continue

        ep = float(open_[t + 1])
        sp = ep - STOP_MULT * atr[t]
        entry_dt_arr[t] = dt[t + 1]
        entry_price[t] = ep
        stop_price[t] = sp
        stop_dist[t] = (ep - sp) / ep

        # 시나리오 A
        oa, ja, xa, mech_a = simulate_a(t, sp, tp1_thr, low, open_, ep)
        A_outcome[t] = oa
        A_exit_mechanism[t] = mech_a
        if oa != 0:
            A_exit_dt[t] = dt[ja]
            A_exit_p[t] = xa
            A_hold[t] = ja - (t + 1)
            A_return[t] = (xa - ep) / ep
        # oa==0 (미도달): holding_days/return 모두 NaN 유지, mechanism='none'

        # 시나리오 B
        ob, jb, xb, mech_b = simulate_b(t, sp, atr, low, high, open_, ep)
        B_outcome[t] = ob
        B_exit_mechanism[t] = mech_b
        if ob != 0:
            B_exit_dt[t] = dt[jb]
            B_exit_p[t] = xb
            B_hold[t] = jb - (t + 1)
            B_return[t] = (xb - ep) / ep

        # 진입일의 TP2 임계값 (참고용)
        if not np.isnan(atr[t + 1]):
            B_chand_at_entry[t] = float(high[t + 1]) - CHANDELIER_MULT * atr[t + 1]

        # TP1만
        l1, j1, x1 = simulate_tp1_only(t, tp1_thr, low, open_)
        tp1_only_label[t] = l1
        if l1 != 0:
            tp1_only_exit_dt[t] = dt[j1]
            tp1_only_hold[t] = j1 - (t + 1)
            tp1_only_ret[t] = (x1 - ep) / ep

        # TP2만
        l2, j2, x2 = simulate_tp2_only(t, atr, high, low, open_)
        tp2_only_label[t] = l2
        if l2 != 0:
            tp2_only_exit_dt[t] = dt[j2]
            tp2_only_hold[t] = j2 - (t + 1)
            tp2_only_ret[t] = (x2 - ep) / ep

    out = pd.DataFrame(
        {
            "dt": pd.to_datetime(df["dt"]),
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": df["volume"].astype("int64"),
            "tr": tr,
            "atr": atr,
            "tp1_threshold": tp1_thr,
            "entry_dt": pd.to_datetime(entry_dt_arr),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "stop_distance_pct": stop_dist,
            "A_outcome": pd.array(A_outcome, dtype="Int8"),
            "A_exit_dt": pd.to_datetime(A_exit_dt),
            "A_exit_price": A_exit_p,
            "A_holding_days": A_hold,
            "A_return": A_return,
            "A_exit_mechanism": A_exit_mechanism,
            "B_outcome": pd.array(B_outcome, dtype="Int8"),
            "B_exit_dt": pd.to_datetime(B_exit_dt),
            "B_exit_price": B_exit_p,
            "B_holding_days": B_hold,
            "B_return": B_return,
            "B_chand_thr_at_entry": B_chand_at_entry,
            "B_exit_mechanism": B_exit_mechanism,
            "tp1_only_label": pd.array(tp1_only_label, dtype="Int8"),
            "tp1_only_exit_dt": pd.to_datetime(tp1_only_exit_dt),
            "tp1_only_holding_days": tp1_only_hold,
            "tp1_only_return": tp1_only_ret,
            "tp2_only_label": pd.array(tp2_only_label, dtype="Int8"),
            "tp2_only_exit_dt": pd.to_datetime(tp2_only_exit_dt),
            "tp2_only_holding_days": tp2_only_hold,
            "tp2_only_return": tp2_only_ret,
        }
    )
    return out


def print_stats(df: pd.DataFrame, out: pd.DataFrame) -> None:
    n = len(df)
    print("=== 데이터 요약 ===")
    print(f"총 행: {n}")
    if n > 0:
        print(f"기간: {df['dt'].iloc[0].date()} ~ {df['dt'].iloc[-1].date()}")
    if n > ATR_PERIOD:
        print(f"ATR 산출 시작 인덱스: {ATR_PERIOD} (= {df['dt'].iloc[ATR_PERIOD].date()})")
    expected = max(n - (ATR_PERIOD + 1), 0)
    actual_enterable = out["entry_price"].notna().sum()
    print(f"진입 가능 행: {actual_enterable}  (프롬프트 공식 N-21 = {expected};")
    print(f"  사용자 결정 (b)로 tradeable[t+1]=False 행이 추가 제외되어 실제값은 더 작을 수 있음)")
    print()

    def _block(name, outcome_col, ret_col, hold_col, mech_col, tp_name):
        out_sub = out[outcome_col].dropna()
        wins = int((out_sub == 1).sum())
        losses = int((out_sub == -1).sum())
        nones = int((out_sub == 0).sum())
        reached = wins + losses
        wr = (wins / reached * 100.0) if reached > 0 else float("nan")
        ret_reached = out.loc[out[outcome_col].isin([-1, 1]), ret_col]
        hold_reached = out.loc[out[outcome_col].isin([-1, 1]), hold_col]
        avg_ret = ret_reached.mean() * 100.0 if len(ret_reached) > 0 else float("nan")
        avg_hold = hold_reached.mean() if len(hold_reached) > 0 else float("nan")
        print(f"=== {name} ===")
        print(f"진입 가능 행: {len(out_sub)}")
        print(f"[수익률 기반 outcome]")
        print(f"  win(+1): {wins}, loss(-1): {losses}, none(0): {nones}")
        print(f"  승률 (none 제외): {wr:.2f}%")
        print(f"  평균 수익률 (도달건만): {avg_ret:.2f}%")
        print(f"  평균 보유일 (도달건만): {avg_hold:.2f}")

        # 메커니즘별 분포
        print(f"\n[메커니즘별 분포]")
        mech_series = out[mech_col]
        for mech in ("stop", tp_name, "none"):
            mask = mech_series == mech
            cnt = int(mask.sum())
            if mech == "none":
                print(f"  none(미도달): {cnt}건")
            else:
                ret_mech = out.loc[mask, ret_col]
                avg_r = ret_mech.mean() * 100.0 if len(ret_mech.dropna()) > 0 else float("nan")
                print(f"  {mech} 발동: {cnt}건, 평균 수익률: {avg_r:.2f}%")

        # 정책 변경 영향 확인
        print(f"\n[정책 변경 영향 확인]")
        tp_mask = (mech_series == tp_name) & out[ret_col].notna()
        tp_total = int(tp_mask.sum())
        tp_neg = int((tp_mask & (out[ret_col] <= 0)).sum())
        tp_neg_pct = (tp_neg / tp_total * 100.0) if tp_total > 0 else float("nan")
        print(f"  {tp_name} 발동 중 수익률 ≤ 0 인 케이스: {tp_neg}건 ({tp_neg_pct:.2f}%)")
        print(f"    → 기존 정책에서는 +1이었으나 새 정책에서는 -1로 분류")

        stop_mask = (mech_series == "stop") & out[ret_col].notna()
        stop_total = int(stop_mask.sum())
        stop_pos = int((stop_mask & (out[ret_col] > 0)).sum())
        stop_pos_pct = (stop_pos / stop_total * 100.0) if stop_total > 0 else float("nan")
        print(f"  stop 발동 중 수익률 > 0 인 케이스: {stop_pos}건 ({stop_pos_pct:.2f}%)")
        print(f"    → 기존 정책에서는 -1이었으나 새 정책에서는 +1로 분류")
        print()

    _block(
        "시나리오 A: 손절(-2N) vs TP1(직전 10일 일중 low)",
        "A_outcome", "A_return", "A_holding_days", "A_exit_mechanism", "tp1",
    )
    _block(
        "시나리오 B: 손절(-2N) vs TP2(Chandelier -3N)",
        "B_outcome", "B_return", "B_holding_days", "B_exit_mechanism", "tp2",
    )

    def _tp_only(name, label_col, ret_col, hold_col):
        sub = out[label_col].dropna()
        reach = int((sub == 1).sum())
        total = int(len(sub))
        rate = (reach / total * 100.0) if total > 0 else float("nan")
        avg_ret = out.loc[out[label_col] == 1, ret_col].mean() * 100.0 if reach > 0 else float("nan")
        avg_hold = out.loc[out[label_col] == 1, hold_col].mean() if reach > 0 else float("nan")
        print(f"{name} 도달률: {rate:.2f}%, 평균 수익률: {avg_ret:.2f}%, 평균 보유일: {avg_hold:.2f}")

    print("=== 참고: 손절 무시한 익절만의 도달률 ===")
    _tp_only("TP1", "tp1_only_label", "tp1_only_return", "tp1_only_holding_days")
    _tp_only("TP2", "tp2_only_label", "tp2_only_return", "tp2_only_holding_days")


def main():
    ap = argparse.ArgumentParser(description="Turtle ATR 손절 + TP1/TP2 라벨링")
    ap.add_argument("--src", required=True, type=Path, help="입력 parquet/csv 경로")
    ap.add_argument("--out", required=True, type=Path, help="출력 parquet 경로 (.csv 동시 저장)")
    ap.add_argument("--quiet", action="store_true", help="통계 출력 생략")
    args = ap.parse_args()

    # §9.3 수정주가 경고
    print(
        "[NOTE] 입력 데이터는 수정주가(adjusted price) 기준으로 가정합니다 "
        "(사용자 확인됨). 비수정주가 데이터를 넣으면 결과가 부정확할 수 있습니다.",
        file=sys.stderr,
    )

    df = load_data(args.src)
    out = label(df)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    csv_path = out_path.with_suffix(".csv")
    out.to_csv(csv_path, index=False)

    if not args.quiet:
        print_stats(df, out)


if __name__ == "__main__":
    main()
