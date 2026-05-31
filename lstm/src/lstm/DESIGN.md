# LSTM 익절 확률 예측 — 설계 문서

## 1. 목표
- **입력**: 종목 t시점까지의 최근 N일(lookback) 시계열 피처
- **출력**: 해당 시점 t의 익절 확률 `P(B_outcome = +1)` ∈ [0,1]
- **활용**: 트레이딩 시그널을 위한 확률 스코어 (XGBoost/룰 기반과 비교 가능한 baseline)

본 문서는 **general 피처만 사용하는 LSTM**의 설계를 다룬다. general+youtube 결합 모델은 후속 단계.

---

## 2. 데이터 입력
출처: `output/general/` (run_pipeline + attach_labels 결과물)

| 파일 | shape | 비고 |
|---|---|---|
| X.parquet | (441,920, 163) | 피처 행렬 |
| y.parquet | (441,920, 2)   | `label_binary`, `label_3class` |
| meta.parquet | (441,920, 2) | `date`, `ticker` |

- 종목 230개, 기간 1999-06-28 ~ 2026-05-06
- 피처 163개 = base 97 + cross-sectional 정규화(`_cs`) + interaction(`inter_*`)

---

## 3. 라벨 정의 (Turtle B_outcome, 매우 중요)

`data/processed/labels/` 의 Turtle Trading 라벨 사용 ([scripts/label_turtle.py](../../scripts/label_turtle.py)).

### 진입·청산 룰
- 진입가 = 익일 시가 (`open[t+1]`)
- 손절선 = 진입가 − 2 × ATR(20) (PDF p.22)
- 익절선 = 누적최고가 − 3 × ATR (Chandelier, Le Beau & Lucas 1992)
- 매 시점 t에서 시계열 끝까지 시뮬레이션, **먼저 닿는 쪽**으로 outcome 결정

### 매핑 (attach_labels.py)
| B_outcome | label_3class | label_binary | 의미 |
|---|---|---|---|
| −1 | 0 | 0 | 손절 (2 ATR 이탈) |
| +1 | 1 | **1** | 익절 (Chandelier TP2) |
|  0 | 2 | 0 | 시계열 끝까지 미도달 |
| NaN | NaN | NaN | 라벨 미생성 |

### ⚠️ horizon 없음
일반적인 triple-barrier(fixed horizon)와 **다름**. 보유일 통계 (output/general 기준):

| outcome | n | median hold | p90 | max |
|---|---|---|---|---|
| 손절 | 156,496 | 6d | 14d | 46d |
| 익절 | 281,513 | 16d | 35d | **380d** |
| 미도달 | 2,155 | — | — | — (시계열 끝) |

**학습 시 시사점**:
- 30일 horizon이라 가정하면 안 됨
- walk-forward purge gap 결정 시 익절 p90=35d 기준
- max 380d는 극단치 → 별도 처리하지 않고 purge 60d로 다수 케이스만 커버

### 학습 타깃
- **`label_binary`** 사용 (1 = 익절, 0 = 손절·미도달)
- NaN인 1,986행은 학습 샘플에서 제외
- 양성 비율 **63.70%** (양성에 약간 치우침, class_weight 미적용)

---

## 4. 모델 아키텍처

```
Input        (batch, lookback, 70)   # selected 모드 (default)
              (batch, lookback, 103)  # full 모드 (--selected_features 미지정)
  │
  │  (sklearn scaler 없음 — add_cs_zscore 결과 그대로, 자세한 내용은 §6)
  ▼
LSTM         hidden=128, num_layers=2, dropout=0.2, batch_first
  │
  │  last hidden state (batch, 128)
  ▼
Head         Dropout(0.3)
             ├─ Linear(128 → 64)
             ├─ ReLU
             └─ Linear(64 → 1)
  │
  ▼
Output       logit  ─(sigmoid)→  P(익절) ∈ [0,1]
```

### 하이퍼파라미터 (초기값)
| 항목 | 값 | 근거 |
|---|---|---|
| Loss | `BCEWithLogitsLoss` | sigmoid 통합, 수치 안정 |
| Optimizer | `AdamW(lr=1e-4, weight_decay=1e-5)` | lr=1e-3은 MPS+LSTM에서 발산 확인 |
| Grad clip | `max_norm=5.0` | 표준적 LSTM 안정화 |
| Batch | 256 | MPS 메모리 + 속도 균형 |
| Epoch | 최대 30 | early stop으로 자동 조기종료 |
| Patience | val AUC 5 epoch 정체 | overfit 방지 |
| Seed | 42 | 재현성 |

---

## 5. Walk-forward 검증

### 구조 (Expanding train + 1y val + 1y test)
| 항목 | 값 |
|---|---|
| 첫 test 연도 | 2012 |
| train 시작 | data_min (≈1999-06) |
| val 길이 | 1년 |
| test 길이 | 1년 |
| step | 1년 |
| **purge** | 60일 (val_start − 60d까지가 train) |
| fold 수 | 14 (test 2012 ~ 2025) |

### 예시 fold (fold 0)
```
train  : 1999-06-30 ~ 2010-11-01   (1y val − 60d purge)
val    : 2011-01-01 ~ 2011-12-31
test   : 2012-01-01 ~ 2012-12-31
```

### purge 60d 근거
익절 라벨 결정에 걸리는 보유일 p90 = 35d. 안전마진 2배로 60d. 극단치 max 380d까지 커버하려면 1y+ 필요하지만 학습 데이터 손실 큼 → 다수 케이스(p90)만 커버.

### 시간 누설(leakage) 분석
- 정합: 각 fold의 train/val/test 시점 disjoint + purge gap
- 잔여 위험: train 마지막 ~60일에 진입한 익절 라벨 중 60d 초과 케이스(약 10%)가 val 기간에 도달 → 미세한 라벨 누설. 실용적으론 무시.

---

## 6. 데이터 처리

### 피처 NaN
- 원인: `past_*_1y` 등 lookback 1년 필요 피처가 ticker 초기 구간에 NaN (≈ 13%)
- 처리: ticker별 forward-fill → 남은 NaN은 0
- indicator(mask) 컬럼은 추가하지 않음 (단순화)

### 라벨 NaN
- 1,986행 (0.45%) — 라벨 파일에 없는 (date, ticker)
- 학습 샘플 생성 시 `y.notna()` 조건으로 제외

### 피처 선택 (Permutation Importance, 상위 70개)
Downloads/src/`permutation_selection.py`를 prj 데이터로 직접 실행해서 상위 70개 피처를 선별.

| 설정 | 값 |
|---|---|
| 측정 모델 | XGBoost (n_estimators=500, max_depth=6, lr=0.05) |
| 측정 fold | 2023, 2024, 2025, 2026 (4 fold) |
| repeat | 피처당 3회 (Sharpe 하락 측정, 노이즈 평균) |
| 입력 | `output/perm_input/df_full.parquet` (B_outcome/B_return/B_holding_days 제거 — 라벨 누설 방지) |
| 결과 | `output/perm_sel/selected_features.csv` (70개) |

**선별된 70개 카테고리 분포**: `_cs` 28 / raw 35 / `inter_*` 2 / `cs_rank_*` 4 / `sector_*` 1

**Top 5**: `pos_in_52w` (+0.944), `dist_to_52w_low_cs` (+0.781), `mfe_30d_mean_cs` (+0.760), `macd_signal` (+0.629), `ret_20d` (+0.621)

⚠️ permutation_selection.py 원본 `exclude` 리스트에 라벨 파생 컬럼 (`B_outcome`, `B_return`, `B_holding_days`)이 누락되어 있어 prj에서는 임시 df_full(누설 컬럼 drop)을 만들어 실행.

### Scaling / Normalization (feature_v3_transform.add_cs_zscore 만 사용)
LSTM 입력 정규화는 **`feature_v3_transform.py`의 `add_cs_zscore` 함수만 사용**. sklearn scaler 미사용.

**add_cs_zscore 규칙** (Downloads/src/feature_v3_transform.py:78):
```
z = (x - mean_t) / (std_t + 1e-10)    # mean/std는 같은 date의 종목간
z = z.clip(-5, 5)                      # ±5 클리핑
df[f'{c}_cs'] = z                      # 원본은 유지, _cs 접미사로 추가
```

**X.parquet 컬럼 구성** (총 163):
| 분류 | 개수 | 처리 |
|---|---|---|
| `_cs` 접미사 (이미 cs_zscore 적용됨) | 60 | 그대로 사용 |
| _cs 짝 있는 원본 | 60 | **drop** (중복) |
| binary (`is_kospi`) | 1 | 그대로 |
| 나머지 numerical (inter_*, cs_rank_*, sector_*, raw 18개) | 42 | **add_cs_zscore 추가 적용 후 _cs만 사용** |

→ 최종 LSTM 입력 **103차원** (기존 _cs 60 + 새 _cs 42 + is_kospi 1)

**왜 inter_*, cs_rank_*, sector_*도 cs_zscore 추가 적용?**
이름만 보면 이미 정규화된 듯하지만 실제 분포 확인 결과:
- `inter_smallcap_vol` (-141~75): `-log_mktcap × vol_20_cs`라 log_mktcap (24~33)이 곱해져 큰 값
- `sector_cat` (0~20): ordinal 인코딩, ±5 밖
- `cs_rank_*` (0~1): [0,1] 범위지만 일관성을 위해 동일 처리

이 모두에 add_cs_zscore (date별 z-score + clip ±5)를 적용해 LSTM 입력 분포 통일.

**처리 순서** ([utils.py](utils.py) `preprocess_features`):

`selected=None` (full 모드):
```
X.parquet + meta
  ↓ _cs 짝 없는 모든 numerical에 add_cs_zscore (clip ±5)
  ↓ _cs 짝 있는 원본 60개 drop
  ↓ 남는 컬럼: 기존 _cs 60 + 새 _cs 42 + is_kospi 1 = 103
LSTM 입력 (lookback, 103)  — 분포: mean≈0, std≈1, ±5 saturate 0.28%
```

`selected=[…70개]` (selected 모드, **기본 권장**):
```
X.parquet + meta
  ↓ selected 70개 중 raw 컬럼에 add_cs_zscore (clip ±5)
  ↓ raw 컬럼명에 _cs 값을 덮어씀 (이름은 selected_features.csv 순서대로 유지)
  ↓ ticker별 ffill → 남은 NaN은 0 → np.nan_to_num
LSTM 입력 (lookback, 70)   — 분포: mean≈0, std≈1, ±5 saturate 0.22%
```

⚠️ `add_cs_zscore` 적용 후에도 약 0.2% 행이 ±5에 saturate (정상 — clip(±5)의 기대 동작). 정보 손실 미미.

### 시퀀스 샘플 생성
정렬: (ticker, date) — 같은 ticker 행이 연속.

각 row index t가 valid 샘플인 조건:
1. `t - ticker_start[ticker(t)] ≥ lookback − 1` (ticker 내 충분한 과거)
2. `meta.date[t]` ∈ fold 기간 (train / val / test 각각)
3. `y[t]` not NaN

샘플:
- 입력: `X_input[t − lookback + 1 : t + 1]`  → shape (lookback, **70** selected / 103 full)
- 타깃: `y[t]` ∈ {0, 1}

stride = 1 (모든 valid t 사용)

---

## 7. 실험 계획

### 1차 실험: lookback 비교
| lookback | 의미 | 예상 학습 시간 (1 fold, 230 종목) |
|---|---|---|
| 30 | horizon과 비슷한 단기 | 빠름 (~3분) |
| 60 | momentum + 일부 추세 | 중간 (~5분) |
| 90 | 분기 추세 | 느림 (~7분) |
| 120 | MA_120과 일치하는 장기 | 가장 느림 (~10분) |

* 시간은 풀 데이터 1 fold smoke 후 측정 필요. 위 값은 추정.

총 학습 세션: **4 lookback × 14 fold = 56**

### 결과 보고 형식
```
lookback별 walk-forward summary (fold 14개 평균)
                AUC          AP         best fold     worst fold
  lookback_30   x.xxx±x.xxx  x.xxx±x.xxx  20yy: x.xxx   20yy: x.xxx
  lookback_60   ...
  lookback_90   ...
  lookback_120  ...
```

### 2차 실험 (별도 문서)
- general + youtube 결합 모델 (단일 LSTM concat vs dual-branch)

---

## 8. 평가 지표
- **1차**: ROC-AUC, Average Precision (PR-AUC)
- **2차** (후속): 
  - threshold tuning (Youden's J, F1 max)
  - precision@top-k%  (실용적 시그널 관점)
  - 백테스트: prob > τ에서 진입 → 실제 B_return 추적

---

## 9. 출력 디렉토리 구조
```
models/lstm/general/
  lookback_30/
    args.json                       # 실행 args 기록
    walk_forward_summary.csv        # fold별 AUC/AP/n 요약
    fold_00_test2012/
      model.pt                      # best val state dict
      test_predictions.parquet      # (date, ticker, prob_익절, label_binary)
    fold_01_test2013/
      ...
    fold_13_test2025/
  lookback_60/
    ...
  lookback_90/
    ...
  lookback_120/
    ...
```

---

## 10. 하드웨어 / 프레임워크
- PyTorch 2.8.0
- 디바이스: Apple Silicon **MPS** (fallback CPU)
  - `PYTORCH_ENABLE_MPS_FALLBACK=1` 자동 설정
  - 일부 LSTM 연산이 MPS 미지원 시 자동 CPU fallback
- num_workers = 0 (macOS fork 이슈 회피)

---

## 11. 알려진 리스크 / Open Questions
1. **MPS 안정성**: lr=1e-4 + add_cs_zscore 정규화 입력에서도 MPS LSTM이 첫 epoch에 발산하는 케이스 확인 (CPU에선 정상 학습). PyTorch 2.8 + MPS + LSTM 조합의 알려진 수치 안정성 이슈. **풀 학습은 `--device cpu` 권장** (속도 trade-off). MPS 사용하려면 lr 추가 축소(1e-5)와 grad_clip 강화 필요.
2. **양성 비율 64%**: 다소 imbalanced. baseline accuracy 64%부터 시작 → AUC가 0.55+여야 의미 있음.
3. **시간 추정**: 56 학습 세션 → 총 8~15시간 예상. 백그라운드 실행 + checkpoint 기반 재개 필요할 수도. (현재 미구현)
4. **라벨 누설 (max 380d)**: 극단 익절 케이스 일부가 val 기간에 도달. 영향 미미하다고 판단했지만, 정량적 측정은 안 함.
5. **threshold**: 본 모델은 확률 출력만. 실제 트레이딩 시그널 변환 (τ 결정)은 후속 작업.

---

## 12. 의사결정 요약 (선택지 vs 채택)

| 결정 | 채택 | 이유 |
|---|---|---|
| 출력 | binary sigmoid (P(익절)) | 직관적, 미도달 0.5% 무시 가능 |
| 입력 결합 | 일단 general only | 4 lookback baseline 먼저 |
| lookback | {30, 60, 90, 120} 모두 비교 | optimal 미정 |
| split | walk-forward expanding 14 fold | 시간 누설 방지 + 견고성 |
| NaN 라벨 | drop | 0.45%, 정보 손실 미미 |
| NaN 피처 | ticker별 ffill → 0 | 단순, indicator 미추가 |
| 피처 선별 | permutation_selection.py 상위 70개 | XGBoost 기반 importance, 4 fold × 3 repeat, B_* 누설 컬럼 제거 후 실행 |
| Scaling | add_cs_zscore (z-score + clip ±5)만 적용 | feature_v3_transform.py 방식, sklearn scaler 미사용. selected raw 35개에 추가 적용해 saturate 방지 |
| 시퀀스 stride | 1 | 데이터 최대 활용 |
| Framework | PyTorch | 유연성, walk-forward 구현 용이 |
| Device | MPS (fallback CPU) | M-시리즈 5~10x 속도 |
