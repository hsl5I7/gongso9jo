# XGBoost 예측 결과 — 앙상블용

> XGBoost 모델 (Phase 4) 예측 결과 공유 — LSTM과 앙상블 위한 자료

---

## 📁 파일 구성

```
predictions/
├── predictions_all.parquet              ← 전체 통합 (가장 중요)
├── predictions_fold_2018.parquet         ← fold별 분리
├── predictions_fold_2019.parquet
├── predictions_fold_2020.parquet
├── predictions_fold_2021.parquet
├── predictions_fold_2022.parquet
├── predictions_fold_2023.parquet
├── predictions_fold_2024.parquet
├── predictions_fold_2025.parquet
└── predictions_fold_2026.parquet
```

---

## 컬럼 설명

| 컬럼 | 타입 | 의미 |
|------|------|------|
| `date` | datetime | 진입일 |
| `ticker` | str | 종목코드 |
| `p_pos` | float | **XGBoost 예측 익절 확률 [0, 1]** ⭐ |
| `label_binary` | int | 실제 라벨 (1=익절, 0=손절) |
| `test_year` | int | 어느 fold의 test인지 |

---

## 모델 정보

### 학습 환경
- **모델**: XGBoost (binary:logistic)
- **피처**: 99개 (Top70 Permutation + 유튜버 시그널 29)
- **라벨**: 터틀 B_outcome (Chandelier 익절, -2N 손절)
- **CV**: Walk-forward 9 fold (test_year 2018~2026)
- **데이터**: 230 종목, 1999~2026

### 하이퍼파라미터 (baseline)
```python
n_estimators=500
max_depth=6
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
```

### 모델 성능
- Mean AUC: **0.576**
- Mean Alpha (TOP - RANDOM): **+0.79%** per trade
- 6/9 fold 양수 alpha

### 캘리브레이션 (중요)
| p_pos | 실제 익절률 | 평균 수익 |
|-------|------------|----------|
| 0.30~0.50 | ~45-50% | +3~4% |
| 0.55~0.65 | 48~54% | +7~10% |
| **0.65~0.70** | **68%** | **+9.85%** |
| **0.70+** | **86~100%** | **+14~17%** |

→ 확률 높을수록 진짜 잘함 (단조 증가)

---

## 앙상블 사용 예시

### 1. 데이터 로드

```python
import pandas as pd

# XGBoost 예측
xgb_pred = pd.read_parquet('predictions_all.parquet')
xgb_pred = xgb_pred.rename(columns={'p_pos': 'p_xgb'})

# 본인 LSTM 예측 (가정)
lstm_pred = pd.read_parquet('your_lstm_predictions.parquet')
lstm_pred = lstm_pred.rename(columns={'p_pos': 'p_lstm'})

# Merge on (date, ticker)
merged = xgb_pred.merge(
    lstm_pred[['date', 'ticker', 'p_lstm']],
    on=['date', 'ticker'],
    how='inner'  # 양쪽 다 있는 것만
)
print(f"Merged: {len(merged):,}")
```

### 2. 앙상블 방법들

```python
# Method A: 단순 평균
merged['p_ensemble_avg'] = (merged['p_xgb'] + merged['p_lstm']) / 2

# Method B: 가중 평균 (validation으로 가중치 결정)
w_xgb, w_lstm = 0.4, 0.6  # 예시
merged['p_ensemble_weighted'] = (
    w_xgb * merged['p_xgb'] + w_lstm * merged['p_lstm']
)

# Method C: 둘 다 높을 때만 (보수적)
merged['p_ensemble_strict'] = (merged['p_xgb'] > 0.6) & (merged['p_lstm'] > 0.6)

# Method D: 둘 중 하나라도 높으면 (공격적)
merged['p_ensemble_loose'] = (merged['p_xgb'] > 0.5) | (merged['p_lstm'] > 0.5)

# Method E: 로지스틱 회귀 (메타 모델)
from sklearn.linear_model import LogisticRegression
# train fold에서 학습 → test fold에서 적용
```

### 3. 평가

```python
from sklearn.metrics import roc_auc_score

# AUC 비교
print(f"XGBoost AUC: {roc_auc_score(merged['label_binary'], merged['p_xgb']):.4f}")
print(f"LSTM AUC:    {roc_auc_score(merged['label_binary'], merged['p_lstm']):.4f}")
print(f"Avg AUC:     {roc_auc_score(merged['label_binary'], merged['p_ensemble_avg']):.4f}")

# 거래 단위 평가 (B_return은 라벨 데이터에서 가져와야 함)
threshold = 0.55
selected = merged[merged['p_ensemble_avg'] > threshold]
print(f"\nThreshold {threshold}:")
print(f"  거래 수: {len(selected):,}")
print(f"  실제 익절률: {(selected['label_binary']==1).mean()*100:.2f}%")
```

---

## 주의사항

### 1. 같은 데이터 분할 사용 필수

XGBoost는 walk-forward 9 fold (2018~2026) 사용. **LSTM도 동일한 분할 사용해야 함**.

- 같은 train/valid/test 기간
- 같은 (date, ticker) 매칭
- 같은 라벨 (B_outcome)

### 2. Test fold 예측만 결합

`predictions_all.parquet`은 **각 fold의 test 결과만** 모음. Train/valid는 포함 X.

### 3. 라벨 일치 확인

merge 후 `label_binary` 컬럼이 양쪽 (XGBoost, LSTM)에서 일치하는지 확인:
```python
assert (merged['label_binary_x'] == merged['label_binary_y']).all()
```

### 4. 결측치

XGBoost 예측에는 label NaN인 행도 포함됨 (예측은 됨, label만 NaN).
앙상블 평가 시 `label_binary.notna()` 필터링 필요.

---

## 권장 진행 순서

1. **본인 LSTM 학습** (같은 walk-forward 구조)
2. **LSTM 예측 결과 저장** (같은 형식: date, ticker, p_pos, label_binary)
3. **이 XGBoost 예측과 merge**
4. **앙상블 방법 비교** (평균/가중/스트릭트 등)
5. **최적 가중치 결정** (validation fold에서)
6. **test fold에서 최종 평가**

---

## 질문 있으면

XGBoost 학습 코드 (`train_walkforward_v4.py`)도 공유 가능.
같은 데이터 (`output_v4_features/`)도 공유 가능.

필요한 거 말씀하세요.
