"""
주식예측 데이터 파이프라인 v2 (피처 빌더)
=====================================

실제 데이터 스키마 반영:
- 컬럼: dt, open, high, low, close, volume, trade_value, turnover_rt, chg_sig, chg, tradeable
- 269종 × 1993~2026 (현재 살아있는 종목)
- 수정주가, 거래정지 플래그 포함

라벨링은 본 모듈 외부(scripts/label_turtle.py + src/attach_labels.py)에서 처리.

주요 기능:
1. 실제 컬럼명(dt 등) 반영
2. tradeable=False 자동 제외
3. turnover_rt → 발행주식수 → 시가총액 역산
4. 다종목 동시 로딩
5. 시장 proxy 피처 (시장지수 데이터 없는 환경용)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union


# ============================================================
# 1. 데이터 로딩
# ============================================================

def load_single_ticker(filepath: Union[str, Path]) -> pd.DataFrame:
    """단일 종목 parquet 로드 + ticker 컬럼 추가"""
    filepath = Path(filepath)
    df = pd.read_parquet(filepath)
    df['ticker'] = filepath.stem  # 파일명 = 종목코드 (예: '000390')
    return df


def load_ticker_list(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    종목 리스트 파일 로드. 형식 자동 감지:
    - .csv: 첫 컬럼이 종목코드 (stk_cd, ticker, code, symbol 중 자동 감지)
            추가 컬럼(stk_nm, market, sector 등) 있으면 같이 반환
    - .txt: 한 줄에 한 종목코드 (콤마/공백 구분도 처리)

    Returns:
        DataFrame with at minimum 'ticker' column.
        Optionally: 'stk_nm', 'market', 'sector' 등 추가 메타.
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath, encoding='utf-8-sig', dtype=str)
        # 종목코드 컬럼 자동 감지
        code_cols = ['stk_cd', 'ticker', 'code', 'symbol', '종목코드']
        ticker_col = None
        for c in code_cols:
            if c in df.columns:
                ticker_col = c
                break
        if ticker_col is None:
            # 못 찾으면 첫 컬럼 사용
            ticker_col = df.columns[0]
            print(f"[ticker_list][WARN] 종목코드 컬럼 자동 감지 실패. "
                  f"첫 컬럼 '{ticker_col}' 사용.")

        df = df.rename(columns={ticker_col: 'ticker'})
        # 6자리 패딩
        df['ticker'] = df['ticker'].astype(str).str.strip().apply(
            lambda x: x.zfill(6) if x.isdigit() and len(x) < 6 else x
        )
        # 빈 값/NaN 제거
        df = df[df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'nan')]
        df = df.reset_index(drop=True)

        meta_cols = [c for c in ['stk_nm', 'market', 'sector'] if c in df.columns]
        print(f"[ticker_list] {filepath.name}에서 {len(df)}개 종목 로드 (CSV)")
        if meta_cols:
            print(f"  메타 컬럼: {meta_cols}")
        print(f"  샘플: {df['ticker'].head(5).tolist()}")
        return df

    # txt 처리
    text = filepath.read_text(encoding='utf-8-sig')
    tickers = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        for sep in [',', '\t', ' ']:
            if sep in line:
                line = line.split(sep)[0].strip()
                break
        if line and not line.lower().startswith(('ticker', 'code', 'symbol', '종목', 'stk_cd')):
            tickers.append(line)

    tickers = [t.zfill(6) if t.isdigit() and len(t) < 6 else t for t in tickers]
    print(f"[ticker_list] {filepath.name}에서 {len(tickers)}개 종목 로드 (TXT)")
    if len(tickers) > 0:
        print(f"  샘플: {tickers[:5]}")
    return pd.DataFrame({'ticker': tickers})


def load_universe(
    data_dir: Union[str, Path],
    pattern: str = "*.parquet",
    ticker_list: Optional[Union[list, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    종목 데이터 동시 로드. 컬럼명을 표준화하고 ticker 추가.

    Args:
        data_dir: parquet 파일들 폴더
        pattern: 파일 패턴
        ticker_list:
            - list: 종목코드 리스트
            - DataFrame: 'ticker' 컬럼 + 선택적으로 stk_nm/market/sector
              → 메타 정보가 결과에 자동 병합됨
            - None: 폴더 내 전체 사용

    Returns: 통합 DataFrame
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    # ticker_list 정규화
    meta_df = None
    if ticker_list is not None:
        if isinstance(ticker_list, pd.DataFrame):
            tickers_set = set(ticker_list['ticker'])
            if any(c in ticker_list.columns for c in ['stk_nm', 'market', 'sector']):
                meta_df = ticker_list.copy()
        else:
            tickers_set = set(ticker_list)

        files_filtered = [f for f in files if f.stem in tickers_set]
        missing = tickers_set - {f.stem for f in files_filtered}
        if missing:
            print(f"[load_universe][WARN] ticker_list 중 {len(missing)}개 파일 없음: "
                  f"{list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        files = files_filtered
        print(f"[load_universe] ticker_list 필터: 전체 → {len(files)}개 종목")

    if not files:
        raise FileNotFoundError("필터 후 로드할 파일 없음")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        df['ticker'] = fp.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={'dt': 'date'})
    df['date'] = pd.to_datetime(df['date'])

    # 메타 정보 병합
    if meta_df is not None:
        df = df.merge(meta_df, on='ticker', how='left')
        meta_cols = [c for c in ['stk_nm', 'market', 'sector'] if c in df.columns]
        print(f"[load_universe] 메타 정보 병합: {meta_cols}")

    print(f"[load_universe] {len(files)}개 종목, {len(df):,} 행 로드 완료")
    return df


# ============================================================
# 2. 데이터 정제
# ============================================================

def clean_ohlcv(
    df: pd.DataFrame,
    verbose: bool = True,
    max_daily_change: float = 0.30,
    drop_extreme_rows: bool = True,
    drop_extreme_tickers_threshold: int = 3,
) -> pd.DataFrame:
    """
    OHLCV 정제:
    - tradeable=False 제거 (거래정지·관리종목)
    - 가격 sanity check
    - 중복 제거
    - 극단 변동 처리 (액면분할/배당 미조정 의심)

    Args:
        max_daily_change: 정상 일간 변동 한계 (기본 30% — 한국 가격제한폭)
        drop_extreme_rows: max_daily_change 초과 행 제거 여부
        drop_extreme_tickers_threshold: 한 종목에서 극단변동이 이 횟수 이상이면
                                         종목 전체 제거 (None이면 행만 제거)
    """
    df = df.copy()
    n0 = len(df)

    # 1. 거래정지 제거
    if 'tradeable' in df.columns:
        df = df[df['tradeable'] == True]
        if verbose:
            print(f"[clean] 거래정지 제거: {n0:,} → {len(df):,}")

    # 2. 정렬 + 중복 제거
    df = df.sort_values(['ticker', 'date']).drop_duplicates(['ticker', 'date'], keep='last')

    # 3. 가격 sanity
    price_cols = ['open', 'high', 'low', 'close']
    df = df.dropna(subset=price_cols)
    df = df[(df[price_cols] > 0).all(axis=1)]
    df = df[df['high'] >= df[['open', 'close']].max(axis=1)]
    df = df[df['low'] <= df[['open', 'close']].min(axis=1)]

    # 4. 극단 변동 진단
    df['_ret'] = df.groupby('ticker')['close'].pct_change()
    extreme_mask = df['_ret'].abs() > max_daily_change
    extreme_count = extreme_mask.sum()

    if extreme_count > 0:
        # 종목별 극단변동 빈도
        bad_tickers = df.loc[extreme_mask, 'ticker'].value_counts()
        if verbose:
            print(f"[clean][WARN] 일간 ±{max_daily_change:.0%} 초과 변동 {extreme_count}건 "
                  f"(액면분할/배당 미조정 의심)")
            print(f"  영향 종목: {len(bad_tickers)}개")
            if len(bad_tickers) > 0:
                print(f"  빈도 상위: {bad_tickers.head(5).to_dict()}")

        # 종목 단위 제거
        if drop_extreme_tickers_threshold is not None:
            tickers_to_drop = bad_tickers[bad_tickers >= drop_extreme_tickers_threshold].index
            if len(tickers_to_drop) > 0:
                n_before = len(df)
                df = df[~df['ticker'].isin(tickers_to_drop)]
                if verbose:
                    print(f"  → {len(tickers_to_drop)}개 종목 통째 제거 "
                          f"(극단변동 ≥{drop_extreme_tickers_threshold}회): "
                          f"{n_before:,} → {len(df):,} 행")
                    print(f"  → 제거된 종목: {list(tickers_to_drop)[:10]}")
                # 마스크 재계산
                extreme_mask = df['_ret'].abs() > max_daily_change

        # 남은 극단 변동 행 제거
        if drop_extreme_rows:
            n_before = len(df)
            df = df[~extreme_mask]
            if verbose:
                print(f"  → 잔여 극단변동 행 제거: {n_before:,} → {len(df):,}")

    df = df.drop(columns='_ret')

    if verbose:
        print(f"[clean] 최종: {len(df):,} 행, {df['ticker'].nunique()}개 종목")
    return df.reset_index(drop=True)


def estimate_mktcap(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    turnover_rt(%)로 발행주식수 역산 후 시총 추정.
    turnover_rt = volume / shares_outstanding × 100
    → shares = volume / (turnover_rt / 100)

    각 종목별로 최빈 발행주식수를 forward-fill로 보강.
    """
    df = df.copy()

    # turnover_rt > 0 인 경우만 추정
    valid = df['turnover_rt'] > 0
    df.loc[valid, 'shares_est'] = (
        df.loc[valid, 'volume'] / (df.loc[valid, 'turnover_rt'] / 100)
    )

    # 종목별 forward fill + backward fill (발행주식수는 거의 일정)
    df['shares_est'] = df.groupby('ticker')['shares_est'].transform(
        lambda x: x.ffill().bfill()
    )

    df['mktcap_est'] = df['close'] * df['shares_est']
    df['log_mktcap'] = np.log1p(df['mktcap_est'].clip(lower=1))

    if verbose:
        miss = df['mktcap_est'].isna().mean()
        print(f"[mktcap] 시총 추정 완료 (결측률 {miss:.4f})")

    return df


def apply_universe_filter(
    df: pd.DataFrame,
    min_turnover_20d_mn: float = 1000,   # 10억원 = 1000 백만원
    min_mktcap: Optional[float] = 5e10,  # 500억 (None이면 적용 안 함)
    min_history_days: int = 250,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Universe 필터: 거래대금·시총·히스토리 길이

    Args:
        min_turnover_20d_mn: 20일 평균 거래대금 최소값 (단위: 백만원)
                             trade_value 컬럼이 백만원 단위.
        min_mktcap: 최소 시가총액 (단위: 원)
    """
    df = df.copy()

    # 20일 평균 거래대금 (백만원 단위)
    df['trade_value_ma20'] = df.groupby('ticker')['trade_value'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )

    n0 = len(df)
    df = df[df['trade_value_ma20'] >= min_turnover_20d_mn]
    if verbose:
        print(f"[filter] 거래대금 ≥{min_turnover_20d_mn:,.0f}백만원: {n0:,} → {len(df):,}")

    if min_mktcap and 'mktcap_est' in df.columns:
        n1 = len(df)
        df = df[df['mktcap_est'] >= min_mktcap]
        if verbose:
            print(f"[filter] 시총 ≥{min_mktcap:.0e}원: {n1:,} → {len(df):,}")

    # 종목별 최소 거래일수
    counts = df.groupby('ticker').size()
    valid = counts[counts >= min_history_days].index
    df = df[df['ticker'].isin(valid)]

    if verbose:
        print(f"[filter] 최종: {len(df):,} 행, {df['ticker'].nunique()}개 종목")
    return df.reset_index(drop=True)


# ============================================================
# 3. 종목별 피처 (이전 버전과 거의 동일)
# ============================================================

def _add_returns(g: pd.DataFrame) -> pd.DataFrame:
    c = g['close']
    log_c = np.log(c)

    for n in [1, 3, 5, 10, 20, 60]:
        g[f'ret_{n}d'] = log_c.diff(n)

    daily = log_c.diff(1)
    for lag in [1, 2, 3, 5, 10, 20, 30]:
        g[f'ret_lag_{lag}'] = daily.shift(lag)

    for n in [5, 10, 20, 60, 120]:
        ma = c.rolling(n, min_periods=max(2, n // 2)).mean()
        g[f'close_to_ma_{n}'] = c / ma - 1

    ma5, ma20, ma60 = c.rolling(5).mean(), c.rolling(20).mean(), c.rolling(60).mean()
    g['ma_ratio_5_20'] = ma5 / ma20 - 1
    g['ma_ratio_20_60'] = ma20 / ma60 - 1
    return g


def _add_momentum(g: pd.DataFrame) -> pd.DataFrame:
    c, h, l = g['close'], g['high'], g['low']

    for n in [14, 21]:
        diff = c.diff()
        gain = diff.clip(lower=0).rolling(n).mean()
        loss = (-diff.clip(upper=0)).rolling(n).mean()
        rs = gain / (loss + 1e-10)
        g[f'rsi_{n}'] = 100 - 100 / (1 + rs)

    n = 14
    low_n = l.rolling(n).min()
    high_n = h.rolling(n).max()
    fastK = 100 * (c - low_n) / (high_n - low_n + 1e-10)
    g['stoch_fastK'] = fastK
    g['stoch_fastD'] = fastK.rolling(3).mean()
    g['stoch_slowD'] = g['stoch_fastD'].rolling(3).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    g['macd'] = ema12 - ema26
    g['macd_signal'] = g['macd'].ewm(span=9, adjust=False).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']

    for n in [5, 10, 20, 60]:
        g[f'roc_{n}'] = c.pct_change(n)

    g['williams_r_14'] = -100 * (high_n - c) / (high_n - low_n + 1e-10)

    n = 20
    tp = (h + l + c) / 3
    tp_ma = tp.rolling(n).mean()
    tp_md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    g['cci_20'] = (tp - tp_ma) / (0.015 * tp_md + 1e-10)

    for n in [1, 5, 10, 15]:
        g[f'mom_{n}'] = c.pct_change(n)  # 절대 차이 대신 % 변화로 (스케일 통일)

    return g


def _add_volatility(g: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o = g['close'], g['high'], g['low'], g['open']
    log_ret = np.log(c / c.shift(1))

    for n in [5, 10, 20, 60]:
        g[f'vol_{n}'] = log_ret.rolling(n, min_periods=max(2, n // 2)).std()

    n = 20
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std()
    upper, lower = ma + 2 * sd, ma - 2 * sd
    g['bb_upper_dev'] = upper / c - 1
    g['bb_lower_dev'] = lower / c - 1
    g['bb_pctB'] = (c - lower) / (upper - lower + 1e-10)
    g['bb_bandwidth'] = (upper - lower) / (ma + 1e-10)

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    g['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    g['atr_pct_14'] = g['atr_14'] / c

    pk = (np.log(h / l) ** 2) / (4 * np.log(2))
    g['parkinson_vol_20'] = np.sqrt(pk.rolling(20).mean())

    gk = 0.5 * (np.log(h / l) ** 2) - (2 * np.log(2) - 1) * (np.log(c / o) ** 2)
    g['gk_vol_20'] = np.sqrt(gk.rolling(20).mean().clip(lower=0))

    g['vol_of_vol_20'] = g['vol_20'].rolling(20).std()

    # 변동성 단위로 본 15% 상승까지의 거리 (피처용 스케일링 상수)
    g['dist_to_15pct'] = 0.15 / (g['vol_20'] * np.sqrt(30) + 1e-10)

    return g


def _add_volume(g: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = g['close'], g['high'], g['low'], g['volume']
    tv = g['trade_value']  # 데이터에 있는 거래대금 사용

    g['vol_ratio_20'] = v / (v.rolling(20).mean() + 1e-10)
    g['trade_value_ratio_20'] = tv / (tv.rolling(20).mean() + 1e-10)

    direction = np.sign(c.diff()).fillna(0)
    g['obv_chg_5'] = (direction * v).cumsum().pct_change(5)

    tp = (h + l + c) / 3
    vwap = (tp * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
    g['vwap_dev'] = (c - vwap) / vwap

    n = 14
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(1), 0).rolling(n).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(n).sum()
    g['mfi_14'] = 100 - 100 / (1 + pos / (neg + 1e-10))

    mfv = ((c - l) - (h - c)) / (h - l + 1e-10) * v
    g['cmf_20'] = mfv.rolling(20).sum() / (v.rolling(20).sum() + 1e-10)

    g['vol_cv_20'] = v.rolling(20).std() / (v.rolling(20).mean() + 1e-10)

    # turnover_rt 자체도 피처 (유동성·관심도)
    if 'turnover_rt' in g.columns:
        g['turnover_rt_ma20'] = g['turnover_rt'].rolling(20).mean()
        g['turnover_rt_chg'] = g['turnover_rt'] / (g['turnover_rt_ma20'] + 1e-10)

    return g


def _add_trend(g: pd.DataFrame) -> pd.DataFrame:
    c, h, l = g['close'], g['high'], g['low']

    up = h.diff()
    down = -l.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    g['adx_14'] = dx.ewm(alpha=1/14, adjust=False).mean()
    g['plus_di_14'] = plus_di
    g['minus_di_14'] = minus_di

    high_252 = h.rolling(252, min_periods=60).max()
    low_252 = l.rolling(252, min_periods=60).min()
    g['pos_in_52w'] = (c - low_252) / (high_252 - low_252 + 1e-10)
    g['dist_to_52w_high'] = c / (high_252 + 1e-10) - 1
    g['dist_to_52w_low'] = c / (low_252 + 1e-10) - 1
    g['price_percentile_60d'] = c.rolling(60).rank(pct=True)

    return g


def _add_target_specific(g: pd.DataFrame) -> pd.DataFrame:
    c = g['close']
    fwd_max_30 = c.rolling(30).max() / c.shift(30) - 1
    fwd_min_30 = c.rolling(30).min() / c.shift(30) - 1

    g['past_15pct_hits_1y'] = (fwd_max_30 >= 0.15).rolling(252).sum()
    g['past_drawdown_neg35_1y'] = (fwd_min_30 <= -0.035).rolling(252).sum()
    g['mfe_30d_mean'] = fwd_max_30.rolling(60).mean()
    g['mae_30d_mean'] = fwd_min_30.rolling(60).mean()
    return g


def build_per_ticker_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """종목별 시계열 피처 생성"""
    def per_ticker(g):
        g = _add_returns(g)
        g = _add_momentum(g)
        g = _add_volatility(g)
        g = _add_volume(g)
        g = _add_trend(g)
        g = _add_target_specific(g)
        return g

    if verbose:
        n_tickers = df['ticker'].nunique()
        print(f"[features] 종목별 피처 생성 시작 ({n_tickers}개 종목)...")

    result = df.groupby('ticker', group_keys=True).apply(per_ticker, include_groups=False)
    result = result.reset_index()
    if 'level_1' in result.columns:
        result = result.drop(columns=['level_1'])

    if verbose:
        print(f"[features] 완료: {result.shape}")
    return result.reset_index(drop=True)


# ============================================================
# 4. Cross-sectional + 시장 proxy 피처
# ============================================================

def build_market_proxy_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    269종 집계로 의사 시장지수 만들기 (KOSPI/VKOSPI 없는 환경용).
    매일 종목들의 동일가중 평균을 시장 대용으로 사용.
    """
    df = df.copy()
    # 시장 수익률 = 그날 모든 종목의 평균 일간 수익률
    market = df.groupby('date').agg(
        market_ret_1d=('ret_1d', 'mean'),
        market_vol_20=('vol_20', 'mean'),
        market_breadth=('ret_1d', lambda x: (x > 0).mean()),  # 상승 종목 비율
    ).reset_index()

    # 다중 horizon 시장 수익률
    market = market.sort_values('date').reset_index(drop=True)
    market['market_ret_5d'] = market['market_ret_1d'].rolling(5).sum()
    market['market_ret_20d'] = market['market_ret_1d'].rolling(20).sum()
    market['market_vol_chg'] = market['market_vol_20'].pct_change(20)

    df = df.merge(market, on='date', how='left')

    # 종목 - 시장 상대수익률
    df['rel_ret_5d'] = df['ret_5d'] - df['market_ret_5d']
    df['rel_ret_20d'] = df['ret_20d'] - df['market_ret_20d']

    # 베타 (60일 rolling) — 종목 수익률 vs 시장 수익률
    def rolling_beta(g, window=60):
        x = g['market_ret_1d']
        y = g['ret_1d']
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var()
        return cov / (var + 1e-10)

    df['beta_60'] = df.groupby('ticker', group_keys=False).apply(
        lambda g: rolling_beta(g, 60)
    ).reset_index(level=0, drop=True)

    if verbose:
        print(f"[market_proxy] 시장 대용 피처 추가 완료")
    return df


def build_cross_sectional_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """같은 날짜 종목 간 rank 기반 피처 (Alpha101 계열)"""
    df = df.copy()

    df['cs_rank_ret_1d'] = df.groupby('date')['ret_1d'].rank(pct=True)
    df['cs_rank_ret_5d'] = df.groupby('date')['ret_5d'].rank(pct=True)
    df['cs_rank_ret_20d'] = df.groupby('date')['ret_20d'].rank(pct=True)
    df['cs_rank_volume'] = df.groupby('date')['volume'].rank(pct=True)
    df['cs_rank_vol_20'] = df.groupby('date')['vol_20'].rank(pct=True)
    if 'mktcap_est' in df.columns:
        df['cs_rank_mktcap'] = df.groupby('date')['mktcap_est'].rank(pct=True)

    # ts_argmax(close, 20): 최근 20일 중 고점이 며칠 전이었는지
    df['ts_argmax_close_20'] = df.groupby('ticker')['close'].transform(
        lambda x: x.rolling(20).apply(lambda w: 20 - 1 - np.argmax(w.values), raw=False)
    )

    if verbose:
        print(f"[cs_features] cross-sectional 피처 추가 완료")
    return df


def build_sector_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Sector/Market 기반 피처 (메타 정보 있을 때만).
    - 업종 평균 수익률 대비 상대수익률
    - 업종 내 momentum rank
    - market(KOSPI/KOSDAQ) one-hot
    """
    df = df.copy()

    if 'sector' in df.columns and df['sector'].notna().any():
        # 같은 날짜+업종 평균 수익률
        sector_ret = df.groupby(['date', 'sector'])['ret_1d'].transform('mean')
        df['sector_ret_1d'] = sector_ret
        df['sector_rel_ret_1d'] = df['ret_1d'] - sector_ret

        sector_ret_5 = df.groupby(['date', 'sector'])['ret_5d'].transform('mean')
        df['sector_rel_ret_5d'] = df['ret_5d'] - sector_ret_5

        sector_ret_20 = df.groupby(['date', 'sector'])['ret_20d'].transform('mean')
        df['sector_rel_ret_20d'] = df['ret_20d'] - sector_ret_20

        # 업종 내 모멘텀 rank
        df['sector_rank_ret_20d'] = df.groupby(['date', 'sector'])['ret_20d'].rank(pct=True)
        df['sector_rank_vol_20'] = df.groupby(['date', 'sector'])['vol_20'].rank(pct=True)

        # 업종 더미 (XGBoost는 카테고리도 처리 가능)
        df['sector_cat'] = df['sector'].astype('category').cat.codes

        if verbose:
            print(f"[sector] sector 기반 피처 추가 (업종 {df['sector'].nunique()}개)")

    if 'market' in df.columns and df['market'].notna().any():
        # KOSPI=1, KOSDAQ=0 (또는 다른 시장)
        df['is_kospi'] = (df['market'] == 'KOSPI').astype(int)
        if verbose:
            print(f"[sector] market 더미 추가: {df['market'].value_counts().to_dict()}")

    return df


# ============================================================
# 5. 학습 데이터 준비 (X, meta) — 라벨은 attach_labels.py에서 처리
# ============================================================

def prepare_train_data(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    extra_drop: Optional[list] = None,
):
    """X, (선택)y, meta 분리 + 누수 가능 컬럼 제거.

    label_col=None 이면 라벨 없이 X와 meta만 만든다.
    실제 라벨링은 scripts/label_turtle.py + src/attach_labels.py 에서 수행.
    """
    default_drop = [
        # 식별자·날짜
        'date', 'ticker',
        # 시간 누수 위험 (regime overfitting)
        'year', 'month', 'day', 'quarter', 'dayofweek', '_year',
        # 원본 절대값 (비율 피처만 사용)
        'open', 'high', 'low', 'close', 'volume', 'trade_value',
        'chg_sig', 'chg', 'tradeable',
        # 추정 절대값
        'shares_est', 'mktcap_est',
        'trade_value_ma20', 'turnover_rt_ma20',
        # 텍스트 메타 (sector_cat, is_kospi로 변환됨)
        'stk_nm', 'market', 'sector',
        # 라벨 (외부 파이프라인이 별도 부착)
        'label_3class', 'label_binary', 'days_to_event', 'realized_ret',
        'B_outcome', 'B_return', 'B_holding_days',
    ]
    if extra_drop:
        default_drop += extra_drop
    drop_cols = list(set(default_drop))

    if label_col is not None:
        df = df.dropna(subset=[label_col]).copy()
        y = df[label_col].astype(int)
    else:
        df = df.copy()
        y = None

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan)
    meta = df[['date', 'ticker']].reset_index(drop=True)
    return X, y, meta


# ============================================================
# 7. End-to-end 파이프라인
# ============================================================

def run_pipeline(
    data_dir: Union[str, Path],
    min_turnover_20d_mn: float = 1000,   # 10억원 (백만원 단위)
    min_mktcap: float = 5e10,
    start_date: Optional[str] = None,
    verbose: bool = True,
):
    """피처 빌드 전체 흐름 (라벨링은 attach_labels.py 단계에서 부착)."""
    df = load_universe(data_dir)
    df = df.rename(columns={'dt': 'date'}) if 'dt' in df.columns else df
    if start_date:
        df = df[df['date'] >= start_date]

    df = clean_ohlcv(df, verbose)
    df = estimate_mktcap(df, verbose)
    df = apply_universe_filter(df, min_turnover_20d_mn, min_mktcap, verbose=verbose)
    df = build_per_ticker_features(df, verbose)
    df = build_market_proxy_features(df, verbose)
    df = build_cross_sectional_features(df, verbose)
    X, _, meta = prepare_train_data(df, label_col=None)

    if verbose:
        print(f"\n[done] X={X.shape}, 피처 수: {X.shape[1]}")
    return X, meta, df


# ============================================================
# 데모: 업로드된 단일 종목 파일로 테스트
# ============================================================

if __name__ == "__main__":
    # 단일 종목으로 빠른 테스트
    fp = '/mnt/user-data/uploads/000390.parquet'
    df = load_single_ticker(fp)
    df = df.rename(columns={'dt': 'date'})
    df['date'] = pd.to_datetime(df['date'])

    print(f"\n원본: {df.shape}")
    df = clean_ohlcv(df)
    df = estimate_mktcap(df)
    # 이 종목 시총이 1700억대라 시총 필터 None, 거래대금만 (100백만원 = 1억)
    df = apply_universe_filter(df, min_turnover_20d_mn=100, min_mktcap=None)
    df = build_per_ticker_features(df)
    # 단일 종목이라 cross-sectional/market_proxy는 의미 없으므로 스킵
    X, _, meta = prepare_train_data(df, label_col=None)

    print(f"\n=== 최종 결과 ===")
    print(f"X: {X.shape}")
    print(f"\n피처 일부:\n{X.columns.tolist()[:30]}")
