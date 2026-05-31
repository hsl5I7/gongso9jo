"""XGBoost walk-forward baseline.

LSTM과 동일한 fold·동일한 selected_features로 학습.
시퀀스가 아니라 각 valid 시점의 단일 row(현재 시점 피처)를 입력으로 사용.

사용법:
    python scripts/xgb_baseline.py \
        --selected_features output/perm_sel/selected_features.csv \
        --lookback 30 \
        --output_dir models/xgb/general_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "src" / "lstm"))

from dataset import build_ticker_index, build_valid_indices  # noqa: E402
from utils import (  # noqa: E402
    load_general_dataset,
    load_selected_features,
    preprocess_features,
    set_seed,
)
from walk_forward import make_walk_forward_folds  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--general_dir", default="output/general")
    p.add_argument("--output_dir", default="models/xgb/general_v1")
    p.add_argument("--selected_features", type=str,
                   default="output/perm_sel/selected_features.csv")
    p.add_argument("--lookback", type=int, default=30,
                   help="LSTM과 동일 sample 필터링용. XGBoost는 시퀀스 사용 안 함.")
    p.add_argument("--first_test_year", type=int, default=2012)
    p.add_argument("--purge_days", type=int, default=60)
    p.add_argument("--val_test_purge_days", type=int, default=None)
    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--early_stopping_rounds", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only_fold", type=int, default=None)
    p.add_argument("--start_fold", type=int, default=0)
    p.add_argument("--tree_method", type=str, default="hist")
    p.add_argument("--device", type=str, default="cuda",
                   help="xgboost device: cuda | cpu")
    return p.parse_args()


def _safe_metric(fn, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(fn(y_true, y_score))


def main():
    args = parse_args()
    set_seed(args.seed)

    X_df, y_ser, meta = load_general_dataset(Path(args.general_dir))
    order = meta.sort_values(["ticker", "date"], kind="stable").index.to_numpy()
    X_df = X_df.iloc[order].reset_index(drop=True)
    y_ser = y_ser.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    print(f"[data] rows={len(meta):,}  tickers={meta['ticker'].nunique()}  "
          f"date={meta['date'].min().date()}~{meta['date'].max().date()}")

    selected = load_selected_features(Path(args.selected_features))
    X_input, feat_names = preprocess_features(X_df, meta, selected=selected)
    print(f"[features] {len(feat_names)}개")

    y_arr = y_ser.astype("float32").to_numpy()
    _, _, starts_per_row = build_ticker_index(meta)

    folds = make_walk_forward_folds(
        date_min=meta["date"].min(),
        date_max=meta["date"].max(),
        first_test_year=args.first_test_year,
        purge_days=args.purge_days,
        val_test_purge_days=args.val_test_purge_days,
    )
    print(f"[folds] {len(folds)}개")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as fp:
        json.dump(vars(args), fp, indent=2, default=str)

    results: list[dict] = []
    for fold_idx, fold in enumerate(folds):
        if args.only_fold is not None and fold_idx != args.only_fold:
            continue
        if args.only_fold is None and fold_idx < args.start_fold:
            continue
        print(f"\n=== fold {fold_idx} (test {fold['test_start'].year}) ===")

        train_idx = build_valid_indices(meta, y_ser, starts_per_row,
                                        args.lookback, fold["train_start"], fold["train_end"])
        val_idx = build_valid_indices(meta, y_ser, starts_per_row,
                                      args.lookback, fold["val_start"], fold["val_end"])
        test_idx = build_valid_indices(meta, y_ser, starts_per_row,
                                       args.lookback, fold["test_start"], fold["test_end"])
        print(f"  samples: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            print("  [SKIP] empty fold")
            continue

        X_tr, y_tr = X_input[train_idx], y_arr[train_idx]
        X_va, y_va = X_input[val_idx], y_arr[val_idx]
        X_te, y_te = X_input[test_idx], y_arr[test_idx]

        train_pos = float(y_tr.mean())
        scale_pos = (1.0 - train_pos) / train_pos if 0 < train_pos < 1 else 1.0
        print(f"  train_pos={train_pos:.4f}  scale_pos_weight={scale_pos:.3f}")

        t0 = time.time()
        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            scale_pos_weight=scale_pos,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method=args.tree_method,
            device=args.device,
            early_stopping_rounds=args.early_stopping_rounds,
            random_state=args.seed,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = clf.best_iteration
        fit_secs = time.time() - t0

        val_prob = clf.predict_proba(X_va)[:, 1]
        test_prob = clf.predict_proba(X_te)[:, 1]
        val_auc = _safe_metric(roc_auc_score, y_va, val_prob)
        val_ap = _safe_metric(average_precision_score, y_va, val_prob)
        test_auc = _safe_metric(roc_auc_score, y_te, test_prob)
        test_ap = _safe_metric(average_precision_score, y_te, test_prob)
        print(f"  fit={fit_secs:.1f}s  best_iter={best_iter}  "
              f"val_auc={val_auc:.4f}  val_ap={val_ap:.4f}  "
              f"test_auc={test_auc:.4f}  test_ap={test_ap:.4f}  "
              f"test_pos={y_te.mean():.4f}")

        fold_dir = output_dir / f"fold_{fold_idx:02d}_test{fold['test_start'].year}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        clf.save_model(fold_dir / "model.json")

        pred_meta = meta.iloc[test_idx][["date", "ticker"]].copy()
        pred_meta["prob_익절"] = test_prob
        pred_meta["label_binary"] = y_te.astype(int)
        pred_meta.to_parquet(fold_dir / "test_predictions.parquet", index=False)

        results.append({
            "fold": fold_idx,
            "test_year": fold["test_start"].year,
            "train_n": int(len(train_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "best_iter": int(best_iter),
            "val_auc": val_auc,
            "test_auc": test_auc,
            "test_ap": test_ap,
        })

    if results:
        res_df = pd.DataFrame(results)
        summary_path = output_dir / "walk_forward_summary.csv"
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            old = old[~old["fold"].isin(res_df["fold"])]
            res_df = pd.concat([old, res_df], ignore_index=True).sort_values("fold")
        print("\n=== walk-forward summary ===")
        print(res_df.to_string(index=False))
        res_df.to_csv(summary_path, index=False)
        print(f"\n[mean] test_auc={res_df['test_auc'].mean():.4f}  "
              f"test_ap={res_df['test_ap'].mean():.4f}")


if __name__ == "__main__":
    main()
