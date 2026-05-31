"""LSTM walk-forward 학습곡선 시각화.

각 fold 의 epoch 단위 (train_loss, val_loss, val_auc, val_ap) 를 그린다.
- 우선 `models/lstm/<run>/lookback_<L>/fold_NN_testYYYY/history.csv` 를 읽고,
- 없으면 stdout 로그 파일을 정규식으로 파싱한다 (fold 0 처럼 history 가 없는 경우).

사용법:
    python src/lstm/plot_learning_curves.py
        # 기본: models/lstm/general/lookback_60 + models/lstm/general_yt/lookback_60 자동 탐색

    python src/lstm/plot_learning_curves.py \\
        --runs models/lstm/general/lookback_60 \\
        --log-fallback models/lstm/general/lookback_60:0:output/lstm_smoke_full.log
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 로그 라인 예:
#   epoch  1: loss=0.6522  val_auc=0.5363  (45.5s)
#   epoch  1: loss=0.6590  val_loss=0.65  val_auc=0.5429  val_ap=0.7  (97.3s)
EPOCH_RE = re.compile(
    r"epoch\s+(\d+):\s+loss=([\d.]+)"
    r"(?:\s+val_loss=([\d.]+))?"
    r"\s+val_auc=([\d.]+)"
    r"(?:\s+val_ap=([\d.]+))?"
)
FOLD_HEADER_RE = re.compile(r"=== fold (\d+) \(test (\d{4})\) ===")


def parse_log(path: Path) -> dict[int, pd.DataFrame]:
    """stdout 로그를 fold → history DataFrame 으로 파싱."""
    text = path.read_text()
    out: dict[int, list[dict]] = defaultdict(list)
    cur_fold: int | None = None
    for line in text.splitlines():
        h = FOLD_HEADER_RE.search(line)
        if h:
            cur_fold = int(h.group(1))
            continue
        m = EPOCH_RE.search(line)
        if not m or cur_fold is None:
            continue
        epoch, loss, val_loss, val_auc, val_ap = m.groups()
        out[cur_fold].append({
            "epoch": int(epoch),
            "train_loss": float(loss),
            "val_loss": float(val_loss) if val_loss else math.nan,
            "val_auc": float(val_auc),
            "val_ap": float(val_ap) if val_ap else math.nan,
        })
    return {k: pd.DataFrame(v) for k, v in out.items() if v}


def load_run_histories(run_dir: Path, log_fallback: dict[int, Path] | None = None
                       ) -> dict[int, tuple[pd.DataFrame, str]]:
    """run_dir 의 모든 fold 에 대해 (history_df, source) 반환.

    source 는 'history.csv' 또는 'log:<path>'.
    """
    result: dict[int, tuple[pd.DataFrame, str]] = {}
    for fold_dir in sorted(run_dir.glob("fold_*")):
        m = re.match(r"fold_(\d+)_test(\d{4})", fold_dir.name)
        if not m:
            continue
        fold_idx = int(m.group(1))
        hist_path = fold_dir / "history.csv"
        if hist_path.exists():
            result[fold_idx] = (pd.read_csv(hist_path), "history.csv")

    # fallback 적용 — history.csv 가 없는 fold 에만
    if log_fallback:
        for fold_idx, log_path in log_fallback.items():
            if fold_idx in result:
                continue
            parsed = parse_log(log_path)
            if fold_idx in parsed:
                result[fold_idx] = (parsed[fold_idx], f"log:{log_path.name}")

    return result


def plot_single_run(histories: dict[int, tuple[pd.DataFrame, str]],
                    title: str, out_path: Path) -> None:
    folds = sorted(histories.keys())
    n = len(folds)
    if n == 0:
        print(f"  [skip] {title}: no fold data")
        return

    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows),
                              squeeze=False, sharex=False)

    for ax_idx, fold_idx in enumerate(folds):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        df, source = histories[fold_idx]
        ax2 = ax.twinx()

        l1, = ax.plot(df["epoch"], df["train_loss"], color="tab:blue",
                      marker="o", markersize=3, label="train_loss")
        lines = [l1]
        if "val_loss" in df.columns and df["val_loss"].notna().any():
            l2, = ax.plot(df["epoch"], df["val_loss"], color="tab:cyan",
                          marker="s", markersize=3, linestyle="--",
                          label="val_loss")
            lines.append(l2)
        l3, = ax2.plot(df["epoch"], df["val_auc"], color="tab:red",
                       marker="^", markersize=3, label="val_auc")
        lines.append(l3)

        # best epoch 표시
        best_row = df.loc[df["val_auc"].idxmax()]
        ax2.axvline(best_row["epoch"], color="tab:red", alpha=0.25,
                    linestyle=":", linewidth=1)
        ax2.scatter([best_row["epoch"]], [best_row["val_auc"]],
                    s=80, facecolors="none", edgecolors="tab:red",
                    linewidths=1.5, zorder=5)

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss", color="tab:blue")
        ax2.set_ylabel("val_auc", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.grid(True, alpha=0.25)

        subtitle = (f"fold {fold_idx:02d}  "
                    f"best_auc={best_row['val_auc']:.4f}@ep{int(best_row['epoch'])}  "
                    f"({source})")
        ax.set_title(subtitle, fontsize=10)
        ax.legend(lines, [ln.get_label() for ln in lines],
                  loc="lower left", fontsize=8)

    # 남은 subplot 비우기
    for ax_idx in range(n, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}  ({n} folds)")


def plot_comparison(per_run_hist: dict[str, dict[int, tuple[pd.DataFrame, str]]],
                    out_path: Path) -> None:
    """두 run 의 동일 fold 를 좌우로 비교 (val_auc 중심)."""
    if not per_run_hist:
        return
    # 공통 fold
    common_folds = sorted(
        set.intersection(*[set(h.keys()) for h in per_run_hist.values()])
    )
    if not common_folds:
        print("  [skip] comparison: no common folds")
        return

    n = len(common_folds)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows),
                              squeeze=False, sharex=False)
    colors = {"general": "tab:blue", "general_yt": "tab:red"}
    for ax_idx, fold_idx in enumerate(common_folds):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        for run_name, hist in per_run_hist.items():
            if fold_idx not in hist:
                continue
            df, _ = hist[fold_idx]
            col = colors.get(run_name, None)
            ax.plot(df["epoch"], df["val_auc"], marker="o", markersize=3,
                    label=run_name, color=col)
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val_auc")
        ax.set_title(f"fold {fold_idx:02d}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)

    for ax_idx in range(n, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].axis("off")
    fig.suptitle("val_auc: general vs general_yt", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}  ({n} common folds)")


def parse_fallback_arg(s: str) -> tuple[Path, int, Path]:
    """run_dir:fold_idx:log_path 형식."""
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--log-fallback 형식은 run_dir:fold_idx:log_path 이다 (got {s})"
        )
    return Path(parts[0]), int(parts[1]), Path(parts[2])


def main():
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+",
                   default=[
                       "models/lstm/general/lookback_60",
                       "models/lstm/general_yt/lookback_60",
                   ],
                   help="시각화할 run 디렉토리들 (lookback_N 단위)")
    p.add_argument("--log-fallback", nargs="*", default=None,
                   type=parse_fallback_arg,
                   help="history.csv 없는 fold 에 대한 로그 fallback. "
                        "형식: run_dir:fold_idx:log_path")
    p.add_argument("--out-dir", default="output/lstm_learning_curves")
    args = p.parse_args()

    # 기본 fallback: 기존 fold 0 stdout 로그
    if args.log_fallback is None:
        candidates = [
            ("models/lstm/general/lookback_60", 0, "output/lstm_smoke_full.log"),
            ("models/lstm/general_yt/lookback_60", 0, "output/lstm_yt_fold0.log"),
        ]
        args.log_fallback = []
        for run, fi, lp in candidates:
            run_p = repo_root / run
            log_p = repo_root / lp
            if run_p.exists() and log_p.exists():
                args.log_fallback.append((run_p, fi, log_p))

    # fallback 그룹화: run_dir -> {fold_idx: log_path}
    fb_by_run: dict[Path, dict[int, Path]] = defaultdict(dict)
    for run_dir, fold_idx, log_path in args.log_fallback:
        run_dir = (run_dir if run_dir.is_absolute()
                   else (repo_root / run_dir)).resolve()
        log_path = (log_path if log_path.is_absolute()
                    else (repo_root / log_path)).resolve()
        fb_by_run[run_dir][fold_idx] = log_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    per_run_hist: dict[str, dict[int, tuple[pd.DataFrame, str]]] = {}
    for run in args.runs:
        run_p = (Path(run) if Path(run).is_absolute() else repo_root / run).resolve()
        if not run_p.exists():
            print(f"  [skip] {run_p}: not found")
            continue
        run_name = run_p.parent.name  # general / general_yt
        lookback = run_p.name  # lookback_60
        fallback = fb_by_run.get(run_p)
        hist = load_run_histories(run_p, fallback)
        if not hist:
            print(f"  [warn] {run_p}: no fold history found")
            continue
        per_run_hist[run_name] = hist

        out_path = out_dir / f"{run_name}_{lookback}.png"
        plot_single_run(
            hist,
            title=f"{run_name} / {lookback}  —  learning curves",
            out_path=out_path,
        )

    if len(per_run_hist) >= 2:
        plot_comparison(per_run_hist, out_dir / "compare_val_auc.png")


if __name__ == "__main__":
    main()
