# XGBoost 모델 패키지 — LSTM 앙상블용

> Phase 4 XGBoost binary 모델 + 재현용 코드 패키지

---

## 📦 패키지 구성

```
to_team_ensemble/
├── README.md                       # 이 파일
├── docs/
│   └── ENSEMBLE_README.md          # 앙상블 가이드 (필독)
├── code/                           # 재현용 코드 (10개)
│   ├── feature_pipeline_v2.py
│   ├── run_pipeline.py
│   ├── feature_v3_transform.py
│   ├── permutation_selection.py
│   ├── youtuber_signals.py
│   ├── combine_features.py
│   ├── merge_labels_v4.py
│   ├── combine_v4.py
│   ├── train_walkforward_v4.py    # ⭐ 핵심 학습 코드
│   └── visualize_results.py
└── predictions/                    # 예측값 (별도 첨부)
    ├── predictions_all.parquet     # 전체 통합 (앙상블에 사용)
    └── predictions_fold_*.parquet  # fold별 분리
```

---

## 🎯 빠른 시작

### 옵션 A: 예측값만 사용 (앙상블만 하면 됨)

```python
import pandas as pd

xgb = pd.read_parquet('predictions/predictions_all.parquet')
print(xgb.head())
# date, ticker, p_pos (XGBoost 익절 확률), label_binary, test_year

# 본인 LSTM 예측과 merge
my_lstm = pd.read_parquet('your_lstm_pred.parquet')
merged = xgb.merge(my_lstm, on=['date', 'ticker'])
merged['p_ensemble'] = (merged['p_pos'] + merged['p_pos_lstm']) / 2
```

자세한 앙상블 가이드: `docs/ENSEMBLE_README.md`

### 옵션 B: 코드로 재현

전체 파이프라인 재실행하려면 데이터 필요:
- 원본 OHLCV (230 종목)
- 라벨링 데이터 (`output_v4/labels_merged.parquet`)

이 데이터는 별도 공유 (큰 용량). 요청 시 전달.

---

## 🧠 XGBoost 모델 정보

### 학습 환경
- **모델**: XGBoost (binary:logistic)
- **피처**: 99개 (Top70 Permutation + 유튜버 시그널 29)
- **라벨**: B_outcome (터틀 Chandelier 익절, -2N 손절)
- **CV**: Walk-forward 9 fold (test_year 2018~2026)
- **데이터**: 230 종목, 1999~2026

### 하이퍼파라미터 (baseline)
```python
{
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'random_state': 42,
}
```

→ Optuna 30 trial 시도했지만 overfitting으로 **baseline이 더 robust**

### 성능 요약
| 메트릭 | 값 |
|--------|-----|
| Mean AUC | 0.576 |
| Mean Alpha (TOP - RANDOM) | +0.79% per trade |
| Positive alpha folds | 6/9 |
| p > 0.65 실제 익절률 | 67.92% |
| p > 0.70 실제 익절률 | 100% (n=14) |

---

## 🔄 재현 흐름 (코드 사용 시)

원본 데이터부터 시작:

```powershell
# Phase 1-3: 피처 + 시그널 생성
python run_pipeline.py --data_dir <ohlcv폴더> --ticker_list <티커csv> --output_dir output
python feature_v3_transform.py --input output\df_full.parquet --output_dir output_v3
python permutation_selection.py --data_dir output_v3 --ohlcv_dir <ohlcv> --output_dir results\perm_sel --top_n 70
python youtuber_signals.py --input output_v3\df_full.parquet --output output_v3_yt\df_full.parquet
python combine_features.py --signal_data output_v3_yt\df_full.parquet --top70_csv results\perm_sel\selected_features.csv --output_dir output_v3_yt_top70

# Phase 4: 터틀 라벨링 + binary 학습
python merge_labels_v4.py --labels_dir <라벨폴더> --output output_v4\labels_merged.parquet
python combine_v4.py --labels output_v4\labels_merged.parquet --features output_v3_yt_top70\df_full.parquet --output_dir output_v4_features
python train_walkforward_v4.py --data_dir output_v4_features --output_dir results\wf_v4_new_label

# 시각화
python visualize_results.py --results_dir results\wf_v4_new_label
```

---

## ⚠️ 앙상블 시 주의사항

### 데이터 정합성 확인 필수

1. **같은 230 종목**: KOSPI/KOSDAQ 지수(kosdaq, kospi)는 제외됨
2. **같은 라벨링**: B_outcome (1=익절, 0=손절/미도달)
3. **같은 walk-forward 구조**: test_year 2018~2026 (9 fold)
4. **같은 (date, ticker) 매칭**: merge 시 inner join

### Merge 후 검증

```python
merged = xgb.merge(lstm, on=['date', 'ticker'], suffixes=('_xgb', '_lstm'))
# 라벨 일치 확인
assert (merged['label_binary_xgb'] == merged['label_binary_lstm']).all()
print(f"매칭된 행: {len(merged):,}")
```

매칭 행 적으면 데이터 분할 다를 가능성 → 사전 조율 필요

---

## 📋 컬럼 명세 (predictions_all.parquet)

| 컬럼 | 타입 | 의미 |
|------|------|------|
| `date` | datetime64 | 진입일 |
| `ticker` | str | 종목코드 (6자리) |
| `p_pos` | float | **XGBoost 익절 확률 [0, 1]** ⭐ |
| `label_binary` | int | 실제 라벨 (1=익절, 0=손절/미도달) |
| `test_year` | int | 어느 fold의 test (2018~2026) |

행 수: 약 280,000 (9 fold × 평균 31,000)

---

## 💡 핵심 인사이트 (참고)

### 1. 모델 캘리브레이션 좋음
p_pos 값이 높을수록 진짜 잘함 (단조 증가)

| p_pos | 실제 익절률 | 평균 수익 |
|-------|------------|----------|
| 0.30~0.50 | ~45-50% | +3~4% |
| 0.55~0.65 | 48~54% | +7~10% |
| 0.65~0.70 | **68%** | +9.85% |
| 0.70+ | **86~100%** | +14~17% |

→ 앙상블 시 **threshold 0.65 이상이 의미 있음**

### 2. AUC 0.576은 약함
- 라벨 EV가 -0.58%인 어려운 게임
- 무작위 진입 시 평균 손실
- 모델이 +0.79% alpha 만든 것은 의미 있음

### 3. 유튜버 시그널 29개 포함
- 영매공파(A 그룹)만 진짜 alpha 검증됨 (Lift 1.84x)
- 나머지 B/C/D/E는 음수 lift지만 모델이 활용 가능

### 4. Fold별 성능 변동
- 좋은 해: 2018, 2020, 2021, 2022 (alpha +1~3%)
- 약한 해: 2019, 2024, 2026 (alpha -1~0%)
- 2024 AUC 0.533 최저 (regime 변화?)

---

## 🤝 협업 권장 순서

1. **본인 LSTM 학습** (같은 walk-forward 구조로)
2. **LSTM 예측 결과 저장** (같은 형식: date, ticker, p_pos, label_binary)
3. **(date, ticker) merge 확인** (매칭률 90%+ 목표)
4. **앙상블 방법 비교**
   - 단순 평균
   - 가중 평균 (validation에서 가중치 결정)
   - 메타 모델 (로지스틱 회귀)
5. **threshold별 평가** (0.5, 0.55, 0.6, 0.65, 0.7)
6. **최종 백테스트**

---

## 📞 질문이 있다면

- 코드: 주석 잘 달려있음 (각 파일 상단 docstring 참고)
- 데이터 공유: 추가 데이터 필요 시 별도 요청

이번 주는 정리 완료. 다음 주 앙상블 진행 권장.
