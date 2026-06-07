# plotting.py — 可视化函数
#
# 包含：路径图、学习曲线、路径质量柱状图、学习效率图、训练稳定性图、Q过估计图
# 所有函数接受显式参数（wind、reward、seeds），不依赖全局变量。

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from metrics import smooth


# ══════════════════════════════════════════════════════════════════════════════
# 路径图（单种子，三算法并排）
# ══════════════════════════════════════════════════════════════════════════════
def plot_seed_paths(seed, result, out_dir, wind, reward):
    """
    绘制单种子下 VI / DQN-Basic / DQN-Improved 三条路径，
    叠加风场箭头、障碍物和目标格。
    """
    paths_info = result['_paths']
    configs    = [
        ('vi',       'VI',           '#7f8c8d'),
        ('basic',    'DQN-Basic',    '#2980b9'),
        ('improved', 'DQN-Improved', '#e67e22'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=150)
    fig.suptitle(f"Paths — Seed {seed} | {wind} wind · {reward} reward",
                 fontsize=13, fontweight='bold')

    for ax, (key, label, color) in zip(axes, configs):
        path, env = paths_info[key]
        G = env.G
        ax.set_facecolor('white')
        for i in range(G + 1):
            ax.axhline(i, color='black', lw=0.4, zorder=1)
            ax.axvline(i, color='black', lw=0.4, zorder=1)
        for (c, r) in env.obstacles:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor='black', zorder=3))
        gc, gr = env.goal
        ax.add_patch(plt.Rectangle((gc, gr), 1, 1, facecolor='#2ecc71', zorder=3))
        sc, sr = env.start
        ax.plot(sc + 0.5, sr + 0.5, 'o', color='#e67e22', markersize=9, zorder=5)

        xs, ys, us, vs = [], [], [], []
        for (wc, wr), (wmag, wang) in env.wind_field.items():
            ang_rad = np.radians(wang)
            xs.append(wc + 0.5); ys.append(wr + 0.5)
            us.append(wmag * np.sin(ang_rad) * 0.12)
            vs.append(wmag * np.cos(ang_rad) * 0.12)
        ax.quiver(xs, ys, us, vs, color='red', width=0.006, alpha=0.7, zorder=4)

        px = [p[0] + 0.5 for p in path]
        py = [p[1] + 0.5 for p in path]
        ax.plot(px, py, color=color, lw=2.5, zorder=7, alpha=0.9)

        m         = result[key]
        goal_mark = "ok" if m['reached_goal'] else "no"
        ax.set_title(f"{label}  {goal_mark}  ({m['steps']} steps)", fontsize=10)
        ax.set_xlim(0, G); ax.set_ylim(0, G)
        ax.set_xticks(np.arange(G) + 0.5); ax.set_xticklabels(range(G), fontsize=7)
        ax.set_yticks(np.arange(G) + 0.5); ax.set_yticklabels(range(G), fontsize=7)
        ax.set_aspect('equal')

    plt.tight_layout()
    import os
    out = os.path.join(out_dir, f"{wind}_wind_{reward}_reward_paths_seed_{seed}.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ 路径图已保存: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 路径质量均值柱状图
# ══════════════════════════════════════════════════════════════════════════════
def plot_metrics_mean(stats, path, seeds, wind, reward):
    """绘制 Energy / Time / Distance / Steps 四个路径质量指标的均值±标准差。"""
    algos     = ['VI', 'DQN-Basic', 'DQN-Improved']
    algo_keys = ['vi', 'basic', 'improved']
    colors    = ['#7f8c8d', '#2980b9', '#e67e22']
    metrics   = [
        ('energy',   'Energy (delta-V)', 'Energy'),
        ('time',     'Flight Time (s)',   'Time (s)'),
        ('distance', 'Distance (m)',      'Distance (m)'),
        ('steps',    'Steps',             'Steps'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=150)
    fig.suptitle(
        f"Path Quality — {len(seeds)} seeds | {wind} wind · {reward} reward",
        fontsize=12, fontweight='bold', y=1.02
    )
    x = np.arange(len(algos))
    for ax, (mkey, ylabel, title) in zip(axes, metrics):
        means = [stats[k][mkey]['mean'] for k in algo_keys]
        stds  = [stats[k][mkey]['std']  for k in algo_keys]
        bars  = ax.bar(x, means, yerr=stds, capsize=5,
                       color=colors, edgecolor='white', linewidth=0.8, width=0.55,
                       error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
        for bar, mean, std in zip(bars, means, stds):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + std + max([m for m in means if not np.isnan(m)], default=1) * 0.01,
                        f"{mean:.2f}\n±{std:.2f}",
                        ha='center', va='bottom', fontsize=7.5, color='#2c3e50')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(algos, rotation=15, ha='right', fontsize=8.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
        valid = [m for m in means if not np.isnan(m)]
        if valid:
            ax.set_ylim(0, max(valid) * 1.35)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ 路径质量图已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 学习曲线（均值 ± std 阴影）
# ══════════════════════════════════════════════════════════════════════════════
def plot_learning_curves(all_results, path, seeds, wind, reward, smooth_window=30):
    """绘制 DQN-Basic 与 DQN-Improved 的多种子学习曲线（均值 ± std 阴影）。"""
    configs = [
        ('basic',    'DQN-Basic',    '#2980b9'),
        ('improved', 'DQN-Improved', '#e67e22'),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for algo_key, label, color in configs:
        reward_seqs = [np.array(r[algo_key]['rewards']) for r in all_results
                       if 'rewards' in r[algo_key]]
        if not reward_seqs:
            continue
        min_len = min(len(s) for s in reward_seqs)
        matrix  = np.stack([smooth(s[:min_len], smooth_window) for s in reward_seqs])
        min_len2 = min(m.shape[0] for m in matrix)
        matrix   = np.stack([m[:min_len2] for m in matrix])
        mean = matrix.mean(axis=0)
        std  = matrix.std(axis=0, ddof=1) if len(reward_seqs) > 1 else np.zeros_like(mean)
        eps  = np.arange(1, len(mean) + 1)
        ax.plot(eps, mean, color=color, lw=2, label=label)
        ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Cumulative Reward (smoothed)', fontsize=11)
    ax.set_title(
        f"Learning Curves — {len(seeds)} seeds · {wind} wind · {reward} reward\n"
        f"(shaded = ±1 std across seeds, window={smooth_window})",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ 学习曲线已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 学习效率指标图
# ══════════════════════════════════════════════════════════════════════════════
def plot_efficiency_metrics(stats, all_results, path, seeds, wind, reward):
    """
    四子图：AUC / 首达回合数 / 收敛回合数 / 训练总耗时。
    前三个为 Basic vs Improved，最后一个含 VI。
    """
    algo_keys  = ['vi', 'basic', 'improved']
    algo_names = ['VI', 'DQN-Basic', 'DQN-Improved']
    colors     = ['#7f8c8d', '#2980b9', '#e67e22']
    x3 = np.arange(3)
    x2 = np.arange(2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), dpi=150)
    fig.suptitle(
        f"Learning Efficiency Metrics — {len(seeds)} seeds | {wind} wind · {reward} reward",
        fontsize=12, fontweight='bold', y=1.02
    )

    # AUC
    ax = axes[0]
    auc_means = [stats[k]['auc']['mean'] for k in ['basic', 'improved']]
    auc_stds  = [stats[k]['auc']['std']  for k in ['basic', 'improved']]
    bars = ax.bar(x2, auc_means, yerr=auc_stds, capsize=5,
                  color=[colors[1], colors[2]], edgecolor='white', width=0.5,
                  error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
    for bar, m, s in zip(bars, auc_means, auc_stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(s) + abs(m) * 0.02,
                    f"{m:.2f}\n±{s:.2f}",
                    ha='center', va='bottom', fontsize=8, color='#2c3e50')
    ax.set_title('AUC\n(Normalized Reward Area)', fontsize=10, fontweight='bold')
    ax.set_ylabel('AUC (avg reward/ep)', fontsize=9)
    ax.set_xticks(x2); ax.set_xticklabels(['DQN-Basic', 'DQN-Improved'], fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)

    # 首达回合
    ax = axes[1]
    fr_means = [stats[k]['first_reach_ep']['mean'] for k in ['basic', 'improved']]
    fr_stds  = [stats[k]['first_reach_ep']['std']  for k in ['basic', 'improved']]
    bars = ax.bar(x2, fr_means, yerr=fr_stds, capsize=5,
                  color=[colors[1], colors[2]], edgecolor='white', width=0.5,
                  error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
    for bar, m, s in zip(bars, fr_means, fr_stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(s) + max(fr_means) * 0.02,
                    f"{m:.0f}\n±{s:.0f}",
                    ha='center', va='bottom', fontsize=8, color='#2c3e50')
    ax.set_title('First-Reach Episode\n(Convergence Speed)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Episode #', fontsize=9)
    ax.set_xticks(x2); ax.set_xticklabels(['DQN-Basic', 'DQN-Improved'], fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    valid_fr = [m for m in fr_means if not np.isnan(m)]
    if valid_fr:
        ax.set_ylim(0, max(valid_fr) * 1.4)
    ax.annotate('↓ better', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', fontsize=8, color='#7f8c8d', style='italic')

    # 收敛回合
    ax = axes[2]
    cv_means = [stats[k]['convergence_ep']['mean'] for k in ['basic', 'improved']]
    cv_stds  = [stats[k]['convergence_ep']['std']  for k in ['basic', 'improved']]
    bars = ax.bar(x2, cv_means, yerr=cv_stds, capsize=5,
                  color=[colors[1], colors[2]], edgecolor='white', width=0.5,
                  error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
    for bar, m, s in zip(bars, cv_means, cv_stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(s) + max(cv_means) * 0.02,
                    f"{m:.0f}\n±{s:.0f}",
                    ha='center', va='bottom', fontsize=8, color='#2c3e50')
    ax.set_title('Convergence Episode\n(Sample Efficiency)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Episode #', fontsize=9)
    ax.set_xticks(x2); ax.set_xticklabels(['DQN-Basic', 'DQN-Improved'], fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    valid_cv = [m for m in cv_means if not np.isnan(m)]
    if valid_cv:
        ax.set_ylim(0, max(valid_cv) * 1.4)
    ax.annotate('↓ better', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', fontsize=8, color='#7f8c8d', style='italic')

    # 训练耗时（含 VI）
    ax = axes[3]
    tt_means = [stats[k]['train_time']['mean'] for k in algo_keys]
    tt_stds  = [stats[k]['train_time']['std']  for k in algo_keys]
    bars = ax.bar(x3, tt_means, yerr=tt_stds, capsize=5,
                  color=colors, edgecolor='white', width=0.55,
                  error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
    for bar, m, s in zip(bars, tt_means, tt_stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(s) + max(tt_means) * 0.02,
                    f"{m:.1f}s\n±{s:.1f}",
                    ha='center', va='bottom', fontsize=8, color='#2c3e50')
    ax.set_title('Training Wall Time (s)\n(Total Train Duration)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Time (s)', fontsize=9)
    ax.set_xticks(x3); ax.set_xticklabels(algo_names, rotation=15, ha='right', fontsize=8.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    valid_tt = [m for m in tt_means if not np.isnan(m)]
    if valid_tt:
        ax.set_ylim(0, max(valid_tt) * 1.35)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ 学习效率图已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 训练稳定性指标图
# ══════════════════════════════════════════════════════════════════════════════
def plot_stability_metrics(stats, all_results, path, seeds, wind, reward):
    """
    三子图：
      1. 后期奖励滚动标准差（Late Rolling Std）   —— B vs I
      2. 多 seed 奖励方差（Multi-seed Variance）  —— B vs I
      3. 训练曲线负值峰次数（Downward Spikes）    —— B vs I
    """
    x2     = np.arange(2)
    colors = ['#2980b9', '#e67e22']
    labels = ['DQN-Basic', 'DQN-Improved']
    keys   = ['basic', 'improved']

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=150)
    fig.suptitle(
        f"Training Stability Metrics — {len(seeds)} seeds | {wind} wind · {reward} reward",
        fontsize=12, fontweight='bold', y=1.02
    )

    # Late Rolling Std
    ax = axes[0]
    m_list = [stats[k]['late_rolling_std']['mean'] for k in keys]
    s_list = [stats[k]['late_rolling_std']['std']  for k in keys]
    bars = ax.bar(x2, m_list, yerr=s_list, capsize=5, color=colors,
                  edgecolor='white', width=0.5,
                  error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
    for bar, m, s in zip(bars, m_list, s_list):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(s) + max([v for v in m_list if not np.isnan(v)], default=1) * 0.02,
                    f"{m:.2f}\n±{s:.2f}",
                    ha='center', va='bottom', fontsize=8.5, color='#2c3e50')
    ax.set_title('Late Rolling Std\n(LayerNorm Stabilization Effect)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Reward Std (late phase)', fontsize=9)
    ax.set_xticks(x2); ax.set_xticklabels(labels, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    valid = [v for v in m_list if not np.isnan(v)]
    if valid:
        ax.set_ylim(0, max(valid) * 1.45)
    ax.annotate('↓ better', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', fontsize=8, color='#7f8c8d', style='italic')

    # Multi-seed Reward Variance
    ax = axes[1]
    mv_list = [stats[k]['multiseed_reward_var'] for k in keys]
    bars = ax.bar(x2, mv_list, color=colors, edgecolor='white', width=0.5)
    for bar, m in zip(bars, mv_list):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max([v for v in mv_list if not np.isnan(v)], default=1) * 0.02,
                    f"{m:.2f}",
                    ha='center', va='bottom', fontsize=8.5, color='#2c3e50')
    ax.set_title('Multi-seed Reward Variance\n(Reproducibility across seeds)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Var(reward) averaged over episodes', fontsize=9)
    ax.set_xticks(x2); ax.set_xticklabels(labels, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    valid = [v for v in mv_list if not np.isnan(v)]
    if valid:
        ax.set_ylim(0, max(valid) * 1.45)
    ax.annotate('↓ better', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', fontsize=8, color='#7f8c8d', style='italic')

    # Downward Spikes
    ax = axes[2]
    sp_means = [stats[k]['negative_spikes']['mean'] for k in keys]
    sp_stds  = [stats[k]['negative_spikes']['std']  for k in keys]
    bars = ax.bar(x2, sp_means, yerr=sp_stds, capsize=5, color=colors,
                  edgecolor='white', width=0.5,
                  error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
    for bar, m, s in zip(bars, sp_means, sp_stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(s) + max([v for v in sp_means if not np.isnan(v)], default=1) * 0.02,
                    f"{m:.1f}\n±{s:.1f}",
                    ha='center', va='bottom', fontsize=8.5, color='#2c3e50')
    ax.set_title('Downward Spikes Count\n(Training Instability / Collapse Events)', fontsize=10, fontweight='bold')
    ax.set_ylabel('# of negative spikes (z > 2σ)', fontsize=9)
    ax.set_xticks(x2); ax.set_xticklabels(labels, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    valid = [v for v in sp_means if not np.isnan(v)]
    if valid:
        ax.set_ylim(0, max(valid) * 1.45)
    ax.annotate('↓ better', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', fontsize=8, color='#7f8c8d', style='italic')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ 稳定性指标图已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Q 值过估计比率图
# ══════════════════════════════════════════════════════════════════════════════
def plot_q_overestimation(stats, all_results, path, seeds, wind, reward):
    """
    单子图：Q̂ / MC_return 比率（B vs I）。
    ratio > 1 → 过估计；ratio ≈ 1 → Double DQN 的理论效果。
    同时绘制每个 seed 的散点（透明度低），直观展示分布。
    """
    keys   = ['basic', 'improved']
    labels = ['DQN-Basic', 'DQN-Improved']
    colors = ['#2980b9', '#e67e22']
    x2     = np.arange(2)

    q_means = [stats[k]['q_overestimation']['mean'] for k in keys]
    q_stds  = [stats[k]['q_overestimation']['std']  for k in keys]
    q_vals  = [stats[k]['q_overestimation']['vals'] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    bars = ax.bar(x2, q_means, yerr=q_stds, capsize=6,
                  color=colors, edgecolor='white', width=0.45, alpha=0.85,
                  error_kw={'elinewidth': 2, 'ecolor': '#2c3e50'})

    for xi, vals in enumerate(q_vals):
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(xi + jitter, vals, color=colors[xi],
                   s=30, zorder=5, alpha=0.6, edgecolors='white', linewidths=0.5)

    for bar, m, s in zip(bars, q_means, q_stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(s) + 0.02,
                    f"{m:.3f}\n±{s:.3f}",
                    ha='center', va='bottom', fontsize=9, color='#2c3e50')

    ax.axhline(1.0, color='#e74c3c', linestyle='--', lw=1.5, alpha=0.7,
               label='Ratio = 1.0 (no overestimation)')

    ax.set_title(
        f"Q-Value Overestimation Ratio (Q̂ / MC Return)\n"
        f"{len(seeds)} seeds · {wind} wind · {reward} reward",
        fontsize=11, fontweight='bold'
    )
    ax.set_ylabel('Q̂ / MC Return (ratio)', fontsize=10)
    ax.set_xticks(x2); ax.set_xticklabels(labels, fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    valid = [m for m in q_means if not np.isnan(m)]
    if valid:
        ax.set_ylim(0, max(max(valid) * 1.4, 1.2))

    ax.annotate('ratio ≈ 1 → Double DQN effective\nratio >> 1 → overestimation (Basic DQN issue)',
                xy=(0.02, 0.03), xycoords='axes fraction',
                fontsize=7.5, color='#7f8c8d', style='italic')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ Q过估计图已保存: {path}")
# plotting.py 新增函数
def plot_sample_efficiency(all_results, path, seeds, wind, reward):
    """
    X 轴：episode 数
    Y 轴：平均奖励的累计均值（running mean）
    直观显示 Improved 多快爬到 Basic 的最终水平
    """
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    configs = [('basic', 'DQN-Basic', '#2980b9'),
               ('improved', 'DQN-Improved', '#e67e22')]

    for algo_key, label, color in configs:
        seqs = [np.array(r[algo_key]['rewards']) for r in all_results]
        min_len = min(len(s) for s in seqs)
        # 用 expanding mean 代替 smoothing，更能体现累计学习效率
        matrix = np.stack([
            pd.Series(s[:min_len]).expanding().mean().values
            for s in seqs
        ])
        mean = matrix.mean(axis=0)
        std  = matrix.std(axis=0, ddof=1)
        eps  = np.arange(1, len(mean) + 1)
        ax.plot(eps, mean, color=color, lw=2, label=label)
        ax.fill_between(eps, mean-std, mean+std, color=color, alpha=0.15)

    # 标注 Basic 最终水平的水平线，看 Improved 何时穿越
    basic_final = np.mean([r['basic']['rewards'][-200:] for r in all_results])
    ax.axhline(basic_final, color='#2980b9', lw=1, ls='--', alpha=0.5,
               label=f'Basic final level ({basic_final:.1f})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Expanding mean reward')
    ax.set_title(f'Sample Efficiency — {wind} wind · {reward} reward')
    ax.legend(); plt.tight_layout(); plt.savefig(path); plt.close()