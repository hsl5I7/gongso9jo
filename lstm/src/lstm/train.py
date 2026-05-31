"""general-only LSTM walk-forward 학습.

사용법:
    # smoke test (fold 1개, epoch 3)
    python -m src.lstm.train --smoke

    # 단일 lookback 전체 fold 학습
    python src/lstm/train.py --lookback 60

    # 4개 lookback 순회
    for L in 30 60 90 120; do python src/lstm/train.py --lookback $L; done
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from dataset import WindowDataset, build_ticker_index, build_valid_indices  # noqa: E402
from model import LSTMBinary  # noqa: E402
from utils import (  # noqa: E402
    get_device,
    load_combined_dataset,
    load_general_dataset,
    load_selected_features,
    preprocess_features,
    set_seed,
)
from walk_forward import make_walk_forward_folds  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--general_dir", default="output/general")
    p.add_argument("--youtube_dir", default=None,
                   help="지정 시 general + youtube 결합 모델. 예: output/youtube")
    p.add_argument("--output_dir", default="models/lstm/general")
    p.add_argument("--selected_features", type=str, default=None,
                   help="permutation_selection.py 산출물 selected_features.csv 경로. "
                        "지정 시 해당 피처만 사용 (raw는 add_cs_zscore 적용). "
                        "미지정 시 전체 모드 (103차원)")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--first_test_year", type=int, default=2012)
    p.add_argument("--purge_days", type=int, default=60,
                   help="train↔val 사이 purge (일)")
    p.add_argument("--val_test_purge_days", type=int, default=None,
                   help="val↔test 사이 purge (일). 미지정 시 purge_days와 동일.")
    p.add_argument("--train_start_date", type=str, default=None,
                   help="train 데이터 시작 날짜 (YYYY-MM-DD). data_min 보다 늦추면 그만큼 잘림.")
    p.add_argument("--pos_weight", type=str, default="auto",
                   help="BCEWithLogitsLoss pos_weight. 'auto'(=train 양성비로 fold별 동적), 'none', 또는 양수 값")
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--head_dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0,
                   help="macOS에서는 0 권장 (fork 이슈 회피)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto",
                   help="auto: MPS > CUDA > CPU 우선순위. mps 발산 시 cpu로 강제 권장")
    p.add_argument("--smoke", action="store_true",
                   help="빠른 검증: fold 1개, epoch 3, 종목 일부")
    p.add_argument("--only_fold", type=int, default=None,
                   help="특정 fold 인덱스만 학습 (디버깅용)")
    p.add_argument("--start_fold", type=int, default=0,
                   help="이 fold부터 학습 시작 (이전 fold는 skip). only_fold가 우선.")
    p.add_argument("--max_tickers", type=int, default=None,
                   help="첫 N개 종목만 사용 (smoke / 디버깅)")
    return p.parse_args()


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).detach().cpu().numpy()
            all_logits.append(out)
            all_labels.append(yb.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def _safe_metric(fn, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(fn(y_true, y_score))


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = get_device(prefer_mps=True)
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[device] {device}  [lookback] {args.lookback}")

    # --- 데이터 ---
    if args.youtube_dir:
        X_df, y_ser, meta = load_combined_dataset(
            Path(args.general_dir), Path(args.youtube_dir)
        )
        print(f"[combined] general + youtube → X shape={X_df.shape}")
    else:
        X_df, y_ser, meta = load_general_dataset(Path(args.general_dir))
    # (ticker, date) 정렬
    order = meta.sort_values(["ticker", "date"], kind="stable").index.to_numpy()
    X_df = X_df.iloc[order].reset_index(drop=True)
    y_ser = y_ser.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    if args.max_tickers is not None:
        keep_tickers = meta["ticker"].drop_duplicates().head(args.max_tickers).tolist()
        mask = meta["ticker"].isin(keep_tickers).to_numpy()
        X_df = X_df.loc[mask].reset_index(drop=True)
        y_ser = y_ser.loc[mask].reset_index(drop=True)
        meta = meta.loc[mask].reset_index(drop=True)
        print(f"[max_tickers] {len(keep_tickers)} 종목으로 제한, rows={len(meta):,}")

    print(f"[data] raw_features={X_df.shape[1]}  rows={len(meta):,}  "
          f"tickers={meta['ticker'].nunique()}  "
          f"date={meta['date'].min().date()}~{meta['date'].max().date()}")

    selected = None
    if args.selected_features:
        selected = load_selected_features(Path(args.selected_features))
        print(f"[selected] {args.selected_features} → {len(selected)}개 피처 사용")

    # feature_v3_transform.add_cs_zscore 만 사용해서 정규화.
    X_input, feature_names = preprocess_features(X_df, meta, selected=selected)
    input_size = X_input.shape[1]
    mode = f"selected={len(selected)}" if selected else "full"
    print(f"[features] LSTM 입력: {input_size}개 ({mode} 모드)")
    y_arr_f32 = y_ser.astype("float32").to_numpy()
    ticker_starts, ticker_ends, starts_per_row = build_ticker_index(meta)

    # --- fold ---
    folds = make_walk_forward_folds(
        date_min=meta["date"].min(),
        date_max=meta["date"].max(),
        first_test_year=args.first_test_year,
        purge_days=args.purge_days,
        val_test_purge_days=args.val_test_purge_days,
    )
    user_train_start = pd.Timestamp(args.train_start_date) if args.train_start_date else None
    print(f"[folds] {len(folds)}개"
          f"{f'  (train_start_date={user_train_start.date()})' if user_train_start else ''}")
    for i, f in enumerate(folds):
        ts = max(f['train_start'], user_train_start) if user_train_start else f['train_start']
        print(f"  fold {i:02d}: "
              f"train {ts.date()}~{f['train_end'].date()} | "
              f"val {f['val_start'].date()}~{f['val_end'].date()} | "
              f"test {f['test_start'].date()}~{f['test_end'].date()}")

    if args.smoke:
        folds = folds[:1]
        args.epochs = min(args.epochs, 3)
        print("[SMOKE] fold=1, epochs=3")

    output_dir = Path(args.output_dir) / f"lookback_{args.lookback}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as fp:
        json.dump(vars(args), fp, indent=2, default=str)

    # --- 각 fold 학습 ---
    results: list[dict] = []
    for fold_idx, fold in enumerate(folds):
        if args.only_fold is not None and fold_idx != args.only_fold:
            continue
        if args.only_fold is None and fold_idx < args.start_fold:
            continue
        print(f"\n=== fold {fold_idx} (test {fold['test_start'].year}) ===")

        train_start_eff = max(fold["train_start"], user_train_start) if user_train_start else fold["train_start"]
        if train_start_eff > fold["train_end"]:
            print(f"  [SKIP] train_start_date({train_start_eff.date()}) > train_end({fold['train_end'].date()})")
            continue
        train_idx = build_valid_indices(meta, y_ser, starts_per_row,
                                        args.lookback, train_start_eff, fold["train_end"])
        val_idx = build_valid_indices(meta, y_ser, starts_per_row,
                                      args.lookback, fold["val_start"], fold["val_end"])
        test_idx = build_valid_indices(meta, y_ser, starts_per_row,
                                       args.lookback, fold["test_start"], fold["test_end"])
        print(f"  samples: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            print("  [SKIP] empty fold")
            continue

        # Downloads/src 방침에 따라 fold별 scaler fit/transform 없음 — X_input 그대로 사용.
        train_ds = WindowDataset(X_input, y_arr_f32, train_idx, args.lookback)
        val_ds = WindowDataset(X_input, y_arr_f32, val_idx, args.lookback)
        test_ds = WindowDataset(X_input, y_arr_f32, test_idx, args.lookback)

        pin = (device.type == "cuda")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=pin)

        model = LSTMBinary(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
        ).to(device)

        # pos_weight: 'auto' = fold별 train 양성비 기반 동적, 'none' = 미적용, 숫자 = 고정
        train_pos = float(y_arr_f32[train_idx].mean())
        if args.pos_weight == "auto":
            if 0 < train_pos < 1:
                pw_val = (1.0 - train_pos) / train_pos
                pw_tensor = torch.tensor([pw_val], dtype=torch.float32, device=device)
            else:
                pw_val, pw_tensor = None, None
        elif args.pos_weight == "none":
            pw_val, pw_tensor = None, None
        else:
            pw_val = float(args.pos_weight)
            pw_tensor = torch.tensor([pw_val], dtype=torch.float32, device=device)
        print(f"  train_pos={train_pos:.4f}  pos_weight={pw_val if pw_val is not None else 'none'}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        criterion_eval = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

        best_val_auc = -1.0
        no_improve = 0
        best_state = None
        history: list[dict] = []
        fold_dir = output_dir / f"fold_{fold_idx:02d}_test{fold['test_start'].year}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        history_path = fold_dir / "history.csv"
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            train_loss = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)
            train_loss /= max(n_seen, 1)

            val_logits, val_labels = _predict(model, val_loader, device)
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            with torch.no_grad():
                vl_t = torch.from_numpy(val_logits).to(device)
                vy_t = torch.from_numpy(val_labels).to(device)
                val_loss = float(criterion_eval(vl_t, vy_t).item())
            val_auc = _safe_metric(roc_auc_score, val_labels, val_probs)
            val_ap = _safe_metric(average_precision_score, val_labels, val_probs)

            epoch_secs = time.time() - t0
            print(f"  epoch {epoch+1:2d}: loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  "
                  f"val_ap={val_ap:.4f}  ({epoch_secs:.1f}s)")

            improved = val_auc > best_val_auc
            if improved:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            history.append({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "val_ap": float(val_ap),
                "best": bool(improved),
                "epoch_secs": float(epoch_secs),
            })
            pd.DataFrame(history).to_csv(history_path, index=False)

            if not improved and no_improve >= args.patience:
                print(f"  [early stop] epoch {epoch+1}")
                break

        # 테스트
        if best_state is not None:
            model.load_state_dict(best_state)
        test_logits, test_labels = _predict(model, test_loader, device)
        test_probs = 1.0 / (1.0 + np.exp(-test_logits))
        test_auc = _safe_metric(roc_auc_score, test_labels, test_probs)
        test_ap = _safe_metric(average_precision_score, test_labels, test_probs)
        print(f"  [TEST] AUC={test_auc:.4f}  AP={test_ap:.4f}  pos_rate={test_labels.mean():.4f}")

        # 저장 (fold_dir 는 epoch 루프 전에 이미 mkdir 됨)
        torch.save(best_state, fold_dir / "model.pt")

        pred_meta = meta.iloc[test_idx][["date", "ticker"]].copy()
        pred_meta["prob_익절"] = test_probs
        pred_meta["label_binary"] = test_labels.astype(int)
        pred_meta.to_parquet(fold_dir / "test_predictions.parquet", index=False)

        results.append({
            "fold": fold_idx,
            "test_year": fold["test_start"].year,
            "train_n": int(len(train_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
            "test_ap": float(test_ap),
        })

    if results:
        res_df = pd.DataFrame(results)
        summary_path = output_dir / "walk_forward_summary.csv"
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            # 새 fold가 기존 fold를 갱신하면 새 결과로 덮어씀
            old = old[~old["fold"].isin(res_df["fold"])]
            res_df = pd.concat([old, res_df], ignore_index=True).sort_values("fold")
        print("\n=== walk-forward summary (누적) ===")
        print(res_df.to_string(index=False))
        res_df.to_csv(summary_path, index=False)
        print(f"\n[mean] test_auc={res_df['test_auc'].mean():.4f}  "
              f"test_ap={res_df['test_ap'].mean():.4f}")


if __name__ == "__main__":
    main()
