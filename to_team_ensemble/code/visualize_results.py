"""
visualize_results.py
=====================

학습 결과(wf_v4_new_label)를 한 장의 그림으로 시각화.

사용법:
    python visualize_results.py --results_dir ..\\results\\wf_v4_new_label

출력:
    - performance_overview.png (4 panel)
    - calibration.png (캘리브레이션 곡선)
    - fold_comparison.png (fold별 비교)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True,
                   help="walk_forward_summary.csv와 all_trades_top.csv가 있는 폴더")
    p.add_argument("--output_dir", type=str, default=None,
                   help="기본값: results_dir")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 로드
    summary = pd.read_csv(results_dir / "walk_forward_summary.csv")
    trades = pd.read_csv(results_dir / "all_trades_top.csv")
    trades['date'] = pd.to_datetime(trades['date'])
    trades['year'] = trades['date'].dt.year

    # 한글 폰트 (Windows에서)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # ============================================================
    # Figure 1: 4-panel overview
    # ============================================================
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Fold별 AUC
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(summary['test_year'].astype(int), summary['auc'],
            color='steelblue', alpha=0.7)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax1.axhline(y=summary['auc'].mean(), color='green', linestyle='-',
                alpha=0.5, label=f'Mean ({summary["auc"].mean():.3f})')
    ax1.set_title('Fold별 AUC', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Test Year')
    ax1.set_ylabel('AUC')
    ax1.set_ylim(0.45, max(0.7, summary['auc'].max() + 0.05))
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Fold별 Alpha (TOP - RANDOM)
    ax2 = fig.add_subplot(gs[0, 1])
    alpha = summary['top_total_ret'] - summary['random_total_ret']
    colors = ['green' if a > 0 else 'red' for a in alpha]
    ax2.bar(summary['test_year'].astype(int), alpha, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axhline(y=alpha.mean(), color='blue', linestyle='-',
                alpha=0.5, label=f'Mean ({alpha.mean():+.2f}%)')
    ax2.set_title('Fold별 Model Alpha (TOP - RANDOM)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Test Year')
    ax2.set_ylabel('Alpha (% per trade)')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3: 거래 수
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(summary['test_year'].astype(int), summary['top_n_trades'],
            color='orange', alpha=0.7)
    ax3.set_title('Fold별 거래 수 (TOP)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Test Year')
    ax3.set_ylabel('# Trades')
    ax3.grid(alpha=0.3)
    # 텍스트
    for i, (yr, n) in enumerate(zip(summary['test_year'].astype(int),
                                       summary['top_n_trades'])):
        ax3.text(yr, n + 30, f'{int(n)}', ha='center', fontsize=8)

    # Panel 4: Best threshold
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(summary['test_year'].astype(int), summary['best_threshold'],
             'o-', color='purple', linewidth=2, markersize=10)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Fold별 Auto-selected Threshold', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Test Year')
    ax4.set_ylabel('Threshold')
    ax4.set_ylim(0.3, 1.0)
    ax4.grid(alpha=0.3)
    for yr, t in zip(summary['test_year'].astype(int),
                      summary['best_threshold']):
        ax4.annotate(f'{t:.2f}', (yr, t), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=8)

    fig.suptitle(f'모델 성능 Overview — Mean AUC: {summary["auc"].mean():.3f}, '
                 f'Mean Alpha: {alpha.mean():+.2f}%',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.savefig(output_dir / 'performance_overview.png', dpi=120, bbox_inches='tight')
    print(f"저장: {output_dir / 'performance_overview.png'}")
    plt.close()

    # ============================================================
    # Figure 2: 캘리브레이션 — p_pos vs 실제 익절률
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 구간 binning
    bins = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 1.0]
    trades['p_bin'] = pd.cut(trades['p_pos'], bins=bins, include_lowest=True)
    bin_stats = trades.groupby('p_bin', observed=True).agg(
        n=('return', 'count'),
        win_rate=('return', lambda x: (x > 0).mean() * 100),
        avg_return=('return', lambda x: x.mean() * 100),
        median_return=('return', lambda x: x.median() * 100),
    )
    bin_centers = [(b.left + b.right) / 2 for b in bin_stats.index]

    # 좌: p_pos vs win_rate
    ax_l = axes[0]
    bars = ax_l.bar(range(len(bin_centers)), bin_stats['win_rate'],
                     color='steelblue', alpha=0.7, edgecolor='black')
    # 거래수 텍스트
    for i, (n, w) in enumerate(zip(bin_stats['n'], bin_stats['win_rate'])):
        ax_l.text(i, w + 1, f'n={int(n)}', ha='center', fontsize=8)
    ax_l.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (no skill)')
    ax_l.set_xticks(range(len(bin_centers)))
    ax_l.set_xticklabels([f'{b.left:.2f}-{b.right:.2f}' for b in bin_stats.index],
                          rotation=45, ha='right')
    ax_l.set_xlabel('Predicted Probability (p_pos)')
    ax_l.set_ylabel('Actual Win Rate (%)')
    ax_l.set_title('Calibration: p_pos vs 실제 익절률', fontsize=12, fontweight='bold')
    ax_l.legend()
    ax_l.grid(alpha=0.3, axis='y')

    # 우: p_pos vs avg_return
    ax_r = axes[1]
    colors_r = ['green' if r > 0 else 'red' for r in bin_stats['avg_return']]
    bars_r = ax_r.bar(range(len(bin_centers)), bin_stats['avg_return'],
                       color=colors_r, alpha=0.7, edgecolor='black')
    for i, (n, r) in enumerate(zip(bin_stats['n'], bin_stats['avg_return'])):
        offset = 0.5 if r > 0 else -1.5
        ax_r.text(i, r + offset, f'n={int(n)}', ha='center', fontsize=8)
    ax_r.axhline(y=0, color='black', linewidth=0.8)
    ax_r.set_xticks(range(len(bin_centers)))
    ax_r.set_xticklabels([f'{b.left:.2f}-{b.right:.2f}' for b in bin_stats.index],
                          rotation=45, ha='right')
    ax_r.set_xlabel('Predicted Probability (p_pos)')
    ax_r.set_ylabel('Average Return (%)')
    ax_r.set_title('Calibration: p_pos vs 평균 수익률', fontsize=12, fontweight='bold')
    ax_r.grid(alpha=0.3, axis='y')

    fig.suptitle('모델 캘리브레이션 — 확률 높을수록 진짜 잘하나?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration.png', dpi=120, bbox_inches='tight')
    print(f"저장: {output_dir / 'calibration.png'}")
    plt.close()

    # ============================================================
    # Figure 3: TOP vs RANDOM vs BOTTOM 비교
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(summary))
    width = 0.27

    ax.bar(x - width, summary['top_total_ret'], width, label='TOP',
           color='green', alpha=0.7)
    ax.bar(x, summary['random_total_ret'], width, label='RANDOM',
           color='gray', alpha=0.7)
    ax.bar(x + width, summary['bottom_total_ret'], width, label='BOTTOM',
           color='red', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary['test_year'].astype(int))
    ax.set_xlabel('Test Year')
    ax.set_ylabel('Average Return per Trade (%)')
    ax.set_title('Fold별 TOP/RANDOM/BOTTOM 비교',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'fold_comparison.png', dpi=120, bbox_inches='tight')
    print(f"저장: {output_dir / 'fold_comparison.png'}")
    plt.close()

    # ============================================================
    # 텍스트 요약
    # ============================================================
    print("\n" + "=" * 70)
    print("성능 요약")
    print("=" * 70)
    print(f"  Mean AUC:        {summary['auc'].mean():.4f}")
    print(f"  Mean AP:         {summary['ap'].mean():.4f}")
    print(f"  Mean Prec@10%:   {summary['prec_top10'].mean():.4f}")
    print(f"  Mean TOP avg:    {summary['top_total_ret'].mean():+.2f}%")
    print(f"  Mean RANDOM avg: {summary['random_total_ret'].mean():+.2f}%")
    print(f"  Mean Alpha:      {alpha.mean():+.2f}%")
    print(f"  Positive folds:  {(alpha > 0).sum()}/{len(alpha)}")
    print(f"  Total trades:    {summary['top_n_trades'].sum()}")
    print(f"\n  결과 그래프: {output_dir}")


if __name__ == "__main__":
    main()
