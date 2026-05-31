# 앙상블 핸드오프 문서 (3-model linear ensemble)

> 새 세션이 이 문서만 읽고 바로 이어서 작업할 수 있도록 정리한 자기완결 컨텍스트.
> 최종 갱신: 2026-05-31 · 채택: **raw walk-forward** · repo: github.com/hsl5I7/gongso9jo

---

## 0. TL;DR

- **문제**: 한국주식 "터틀" 진입거래가 익절(label_binary=1)할지 손절(0)할지 예측. walk-forward(연도별), 230종목, 2018~2025.
- **3개 독립 모델**(XGBoost·LSTM·CNN) × 2개 피처변종(**baseline**=가격/기술지표, **youtube**=baseline+유튜버시그널 77feat_Aonly)을 `(date,ticker)`로 합쳐 **선형 앙상블**. baseline끼리·youtube끼리 → 앙상블 2개.
- **매수 전략**: 매일 앙상블점수 상위 k종목 중 점수≥임계값 a인 것 매수.
- **채택 결론(raw walk-forward OOS)**: baseline 평균 **+3.94%/거래**(2020제외 +0.20%), youtube +7.53%(2020제외 −0.34%). **2020(강세장) 빼면 거의 본전.** 진짜 알파는 승률(45~52% vs base 31%).
- **rank-norm은 폐기**: in-sample 대박이었으나 leak-free OOS에서 raw와 동급으로 수렴(과적합/lookahead였음).

---

## 1. 데이터

### 1.1 앙상블 모델 데이터 (이미 생성됨)
`outputs/ensemble/ensemble_3model_{baseline,youtube}.parquet` — **206,165행, test_year 2018~2025, base_rate 0.313, 3모델 라벨일치 100%.**

| 컬럼 | 의미 |
|---|---|
| `date, ticker` | 진입일, 종목(6자리) |
| `test_year` | walk-forward fold |
| `label_binary` | 1=익절, 0=손절 (정답) |
| `p_xgb, p_lstm, p_cnn` | 각 모델 익절확률 [0,1] ⭐ |
| `p_avg, p_avg_rank, p_w_*` 등 | 참고용 사전계산 선형결합(앙상블 가중치 실험용) |

선형 앙상블 = `s = w_xgb·p_xgb + w_lstm·p_lstm + w_cnn·p_cnn` (가중치 합=1).

### 1.2 원본 예측 소스 (모델 재생성 시 필요)
- XGB baseline: `xgboost/wf_v4_70feat_thr05/predictions_all_70feat_noYT.parquet` (`p_pos`)
- XGB youtube: `xgboost/wf_v4_77feat_thr05/predictions_all_77feat_Aonly.parquet` (77feat = 70base + 7 A그룹 유튜버시그널, LSTM youtube와 동일 피처셋)
- LSTM: `lstm/models/lstm/general_v2{,_yt}/lookback_30/fold_*/test_predictions.parquet` (`prob_익절`)
- CNN: `outputs/cnn/{baseline,youtube}/predictions_test_pattern_*ty{YYYY}.csv` (`prob_익절`)
- ⚠️ 스케일 주의: CNN 확률은 ~0.5중심, XGB/LSTM은 ~0.3중심. raw 확률 선형결합 시 CNN이 임계값을 지배.

### 1.3 실현수익률 소스 (수익률 분석/WF에 필요)
`lstm/data/processed/labels/{ticker}.parquet` → **`B_return`** (key: `dt`,ticker). `label_binary==1 ⟺ B_return>0`, 커버리지 100%. 전체 universe 평균수익률 −0.58%(음수 EV 게임).

---

## 2. 현재 파일 (정리 후 잔존)

```
src/ensemble/
├── README.md                       # 이 문서
├── ensemble_3model.py              # ⭐ 앙상블 모델 빌더 (소스→merged parquet)
└── topk_threshold_walkforward.py   # ⭐ 채택: raw walk-forward OOS 검증
outputs/ensemble/
├── ensemble_3model_baseline.parquet   # 현재 앙상블 모델 데이터
├── ensemble_3model_youtube.parquet
├── wf_oos_perfold_{baseline,youtube}.csv  # WF 결과 (연도별 + 선택된 정책)
└── wf_oos_summary.csv                     # WF 종합
```
그 외 모든 실험 스크립트/산출물은 정리됨(§6, git 복구 가능).

---

## 3. 채택 방법: raw walk-forward

**프로토콜** (`topk_threshold_walkforward.py`): test fold Y마다 **과거 fold(<Y)만으로** 정책 `(w_xgb,w_lstm,w_cnn, k, a)`를 선택(목적=과거 pooled B_return 최대화, 제약=과거 모든 fold에서 ≥30거래) → fold Y에 그대로 적용해 OOS 측정. 가중치 simplex step 0.02, k∈{10,20,30}, a∈{0,0.50,0.55,0.60,0.65}(raw 확률 임계값).

실행: `python src/ensemble/topk_threshold_walkforward.py --weight-step 0.02`
(읽는 것: `outputs/ensemble/ensemble_3model_*.parquet` + `lstm/data/processed/labels/`. 결과 CSV 갱신.)

---

## 4. 결과 (raw walk-forward OOS)

| 변종 | OOS 거래 | 승률 | 평균수익/거래 | 2020제외 | 양수fold |
|---|---|---|---|---|---|
| **baseline** | 3,611 | 46.5% | **+3.94%** | **+0.20%** | 5/7 |
| youtube | 1,644 | 52.8% | +7.53% | −0.34% | 5/7 |

**baseline 연도별** (선택된 정책 w/k/a, OOS):
| fold | w(X/L/C) | k | a | n | 승률% | 수익% | base% | random% |
|---|---|---|---|---|---|---|---|---|
|2019|0.52/0.14/0.34|10|0.55|231|45.9|+2.01|28.7|−2.36|
|2020|0.60/0.06/0.34|30|0.55|683|63.7|+19.95|46.0|+2.45|
|2021|0.44/0.06/0.50|30|0.65|48|27.1|−1.73|32.5|−1.60|
|2022|0.42/0.08/0.50|30|0.65|824|54.2|+2.54|28.0|−2.84|
|2023|0.42/0.08/0.50|30|0.65|261|50.6|+0.79|30.7|−2.13|
|2024|0.32/0.18/0.50|30|0.65|1297|29.6|−2.36|23.0|−3.78|
|2025|0.44/0.06/0.50|30|0.65|267|60.7|+3.65|32.7|−0.79|

(youtube는 `wf_oos_perfold_youtube.csv` 참조. 2021·2025 fold가 1거래뿐이라 불안정.)

---

## 5. 핵심 결론 (정직한 판정)

1. **모든 방식에서 "2020 제외 ≈ 0%"** (−0.34 ~ +0.38%). 강세장(2020)이 사실상 유일 수익원.
2. **진짜·견고한 알파는 "random 대비"**: OOS 승률 45~52%(base 31%, random 32%), 매 fold random(~−1.5~−2%)을 +2~4%p 능가. 단 **손익비**(승 ~+11% / 패 ~−7.4%, 손익분기 승률 ~40%) 탓에 승률 우위가 절대수익으로 잘 전환 안 됨.
3. **baseline > youtube**(견고성). youtube는 평균수익 높아 보이나 2020집중·일부 fold 1거래로 불안정.
4. **rank-norm 폐기**: 각 모델 fold별 백분위로 정규화→가중하면 in-sample 압도(baseline +9.56%/ex2020 +3.88%)였으나, leak-free WF에선 +2.46%/+0.14%로 raw(+3.94%/+0.20%)와 동급/이하. in-sample 우위는 intra-fold lookahead + 과적합. → **복잡한 rank-norm 불필요, raw로 충분.**
5. **수익성은 레짐 의존적** — 강세장 수익/평시 본전/약세장(2024) 소폭 손실.

---

## 6. git / 복구

- 원격: **github.com/hsl5I7/gongso9jo** (branch main)
- `fc8fa42` = 정리 **전** 전체 상태 (삭제된 실험 스크립트·산출물 전부 보존)
- `55cbf49` = 정리 후
- 폐기 실험 복구: `git checkout fc8fa42 -- src/ensemble/<파일>` (예: 폐기된 in-sample 결합탐색 `topk_threshold_search.py`, rank-norm WF `topk_threshold_walkforward_rank.py`, `topk_weight_search.py`, `calibrated_ensemble.py`, `find_best_weights.py`, `optimal_weight_search.py`, `ensemble_weight_sweep.py`, `auroc_analysis.py`, `threshold_metrics.py`, `ensemble_apply_weights.py`, `ensemble_xgb_lstm.py`)
- `.gitignore`로 대용량 데이터 제외: `lstm/output`,`lstm/data`,`lstm/models`,`xgboost/*.parquet`,`outputs/cnn`,`*.zip`

### (참고) 폐기된 in-sample 결합탐색 결과
`topk_threshold_search.py`(git 복구)로 (가중치×k×a) in-sample 최적을 찾았던 결과 — **상한치(낙관)**, 배포기준 아님:
- baseline 견고: w(0.60,0.06,0.34) k=30 a=0.55 → in-sample +3.59%/ex2020 +1.07%, 양수 7/8.
- 교훈: in-sample 수치는 WF로 검증 전엔 신뢰 불가.

---

## 7. 다음 단계 후보

- **레짐 필터 결합**: 시장추세(예: 지수 MA) 상승 구간에만 진입 → 2024형 약세장 손실 회피, 알파를 수익으로 전환.
- **손절폭 민감도**: 터틀 −2N 손절 폭 조정으로 손익비 개선(승률 알파 수익화).
- **시각화/리포트**: WF 누적·복리 수익곡선, MDD, fold별 비교.
- **youtube 안정화**: 1거래 fold 문제(min_fold_n 제약 강화) 또는 youtube 포기.
- (선택) ensemble_apply_weights 류로 고정가중치 산출물 재생성 — 단 WF가 가중치를 동적 선택하므로 우선순위 낮음.

---

## 8. 빠른 시작 (새 세션용)

```bash
# (필요시) 앙상블 모델 재생성 — 원본 예측 소스 필요
python src/ensemble/ensemble_3model.py            # → ensemble_3model_{baseline,youtube}.parquet

# 채택된 walk-forward OOS 재실행/검증
python src/ensemble/topk_threshold_walkforward.py --weight-step 0.02   # → wf_oos_*.csv
```
앙상블 데이터/결과는 이미 `outputs/ensemble/`에 있으므로 분석만 할 거면 바로 그 parquet/csv를 읽으면 됨.
