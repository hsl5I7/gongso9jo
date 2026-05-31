# 77개 피처 XGBoost 학습 — 실행 방법

`train_walkforward_v4_ablation.py` 한 개 파일로 학습합니다.
A그룹(영매공파) 7개 시그널만 남기고 나머지 유튜버 시그널(B/C/D/E 22개)은 제외해서
**Top70 + A그룹 7개 = 77개 피처**로 학습합니다.

## 실행 명령

`code/` 폴더에서 실행:

```powershell
python train_walkforward_v4_ablation.py --data_dir ..\output_v4_features --output_dir ..\results\wf_v4_77feat_thr05 --keep_a_only --fixed_threshold --threshold 0.5
```

**옵션 의미**
- `--data_dir ..\output_v4_features` : 입력 데이터 (X.parquet, y.parquet, meta.parquet) 폴더
- `--output_dir ..\results\wf_v4_77feat_thr05` : 결과 저장 폴더 (자동 생성)
- `--keep_a_only` : A그룹(영매공파) 7개만 유지, B/C/D/E 22개 제외
- `--fixed_threshold --threshold 0.5` : 자동 threshold 끄고 0.5로 고정

## 실행 확인 포인트

학습 시작 직후 이 로그가 보이면 정상:

```
X: (265541, 99)
[keep_a_only] A그룹(영매공파) 7개만 유지, B/C/D/E 22개 제외
  유지된 A컬럼: ['sig_yokbae', 'sig_maejip', 'sig_gonguri', 'sig_paran',
                'sig_ma112', 'sig_yg_all', 'sig_yg_strength']
X (제외 후): (265541, 77)

[출력 태그] 77feat_Aonly
```

소요 시간: 약 1~2시간 (9 fold × fold당 5~15분).

## 출력 파일

`..\results\wf_v4_77feat_thr05\` 폴더에 다음이 생성됨:

```
walk_forward_summary_77feat_Aonly.csv     ← fold별 성능 요약 (이게 핵심)
predictions_all_77feat_Aonly.parquet      ← 9 fold 통합 예측
last_model_77feat_Aonly.json              ← 마지막 fold(2026) 학습된 모델
all_trades_top_77feat_Aonly.csv           ← TOP 매수 거래 로그
predictions/predictions_fold_{YYYY}_77feat_Aonly.parquet   ← fold별 9개
```
