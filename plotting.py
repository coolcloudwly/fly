# plotting.py — 可视化函数（三算法对比版）

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from metrics import smooth

# 算法展示配置：key / 标签 / 颜色
ALGO_CONFIGS = [
    ('vi',       'VI',            '#7f8c8d'),
    ('basic',    'DQN-Basic',     '#2980b9'),
    ('eps_only', 'DQN-ε only',    '#27ae60'),
    ('eps_per',  'DQN-ε+PER',     '#e67e22'),
]
DQN_CONFIGS = ALGO_CONFIGS[1:]   # 不含 VI


def _label(key):
    return {c[0]: c[1] for c in ALGO_CONFIGS}[key]

def _color(key):
    return {c[0]: c[2] for c in ALGO_CONFIGS}[key]


# ══════════════════════════════════════════════════════════════════════════════
# 路径图（单种子，四算法并排）
# ══════════════════════════════════════════════════════════════════════════════
def plot_seed_paths(seed, result, out_dir, wind, reward):
    paths_info = result['_paths']
    fig, axes  = plt.subplots(1, 4, figsize=(20, 5.5), dpi=150)
    fig.suptitle(f"Paths — Seed {seed} | {wind} wind · {reward} reward",
                 fontsize=13, fontweight='bold')

    for ax, (key, label, color) in zip(axes, ALGO_CONFIGS):
        if key not in paths_info:
            ax.axis('off')
            continue
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
        goal_mark = "✅" if m['reached_goal'] else "❌"
        ax.set_title(f"{label}  {goal_mark}  ({m['steps']} steps)", fontsize=10)
        ax.set_xlim(0, G); ax.set_ylim(0, G)
        ax.set_xticks(np.arange(G) + 0.5); ax.set_xticklabels(range(G), fontsize=7)
        ax.set_yticks(np.arange(G) + 0.5); ax.set_yticklabels(range(G), fontsize=7)
        ax.set_aspect('equal')

    plt.tight_layout()
    out = os.path.join(out_dir, f"{wind}_wind_{reward}_reward_paths_seed_{seed}.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"✅ 路径图已保存: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 路径质量柱状图
# ══════════════════════════════════════════════════════════════════════════════
def plot_metrics_mean(stats, path, seeds, wind, reward):
    keys   = [c[0] for c in ALGO_CONFIGS]
    labels = [c[1] for c in ALGO_CONFIGS]
    colors = [c[2] for c in ALGO_CONFIGS]
    metrics = [
        ('energy',   'Energy (delta-V)'),
        ('time',     'Flight Time (s)'),
        ('distance', 'Distance (m)'),
        ('steps',    'Steps'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=150)
    fig.suptitle(
        f"Path Quality — {len(seeds)} seeds | {wind} wind · {reward} reward",
        fontsize=12, fontweight='bold', y=1.02
    )
    x = np.arange(len(keys))
    for ax, (mkey, ylabel) in zip(axes, metrics):
        means = [stats[k][mkey]['mean'] for k in keys]
        stds  = [stats[k][mkey]['std']  for k in keys]
        bars  = ax.bar(x, means, yerr=stds, capsize=4,
                       color=colors, edgecolor='white', linewidth=0.8, width=0.6,
                       error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
        valid = [m for m in means if not np.isnan(m)]
        for bar, mean, std in zip(bars, means, stds):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + std + (max(valid, default=1) * 0.01),
                        f"{mean:.2f}\n±{std:.2f}",
                        ha='center', va='bottom', fontsize=7, color='#2c3e50')
        ax.set_title(ylabel, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
        if valid:
            ax.set_ylim(0, max(valid) * 1.4)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ 路径质量图已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 学习曲线（三 DQN 变体）
# ══════════════════════════════════════════════════════════════════════════════
def plot_learning_curves(all_results, path, seeds, wind, reward, smooth_window=30):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for key, label, color in DQN_CONFIGS:
        seqs = [np.array(r[key]['rewards']) for r in all_results if 'rewards' in r[key]]
        if not seqs:
            continue
        min_len = min(len(s) for s in seqs)
        matrix  = np.stack([smooth(s[:min_len], smooth_window) for s in seqs])
        min_len2 = min(m.shape[0] for m in matrix)
        matrix   = np.stack([m[:min_len2] for m in matrix])
        mean = matrix.mean(axis=0)
        std  = matrix.std(axis=0, ddof=1) if len(seqs) > 1 else np.zeros_like(mean)
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
    dqn_keys   = [c[0] for c in DQN_CONFIGS]
    dqn_labels = [c[1] for c in DQN_CONFIGS]
    dqn_colors = [c[2] for c in DQN_CONFIGS]
    all_keys   = [c[0] for c in ALGO_CONFIGS]
    all_labels = [c[1] for c in ALGO_CONFIGS]
    all_colors = [c[2] for c in ALGO_CONFIGS]
    x3 = np.arange(len(dqn_keys))
    x4 = np.arange(len(all_keys))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
    fig.suptitle(
        f"Learning Efficiency — {len(seeds)} seeds | {wind} wind · {reward} reward",
        fontsize=12, fontweight='bold', y=1.02
    )

    def bar_plot(ax, xs, ks, ls, cs, mkey, ylabel, title, annotate='↓ better'):
        means = [stats[k][mkey]['mean'] for k in ks]
        stds  = [stats[k][mkey]['std']  for k in ks]
        bars  = ax.bar(xs, means, yerr=stds, capsize=4, color=cs,
                       edgecolor='white', width=0.55,
                       error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
        valid = [m for m in means if not np.isnan(m)]
        for bar, m, s in zip(bars, means, stds):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(s) + (max(valid, default=1) * 0.02),
                        f"{m:.1f}", ha='center', va='bottom', fontsize=8, color='#2c3e50')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(ls, rotation=20, ha='right', fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
        if valid:
            ax.set_ylim(0, max(valid) * 1.4)
        ax.annotate(annotate, xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', fontsize=8, color='#7f8c8d', style='italic')

    bar_plot(axes[0], x3, dqn_keys, dqn_labels, dqn_colors,
             'auc', 'AUC', 'AUC (↑ better)', '↑ better')
    bar_plot(axes[1], x3, dqn_keys, dqn_labels, dqn_colors,
             'first_reach_ep', 'Episode', 'First Reach Episode')
    bar_plot(axes[2], x3, dqn_keys, dqn_labels, dqn_colors,
             'convergence_ep', 'Episode', 'Convergence Episode')
    bar_plot(axes[3], x4, all_keys, all_labels, all_colors,
             'train_time', 'Time (s)', 'Train Time (s)')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ 学习效率图已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 训练稳定性指标图
# ══════════════════════════════════════════════════════════════════════════════
def plot_stability_metrics(stats, all_results, path, seeds, wind, reward):
    dqn_keys   = [c[0] for c in DQN_CONFIGS]
    dqn_labels = [c[1] for c in DQN_CONFIGS]
    dqn_colors = [c[2] for c in DQN_CONFIGS]
    x = np.arange(len(dqn_keys))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    fig.suptitle(
        f"Training Stability — {len(seeds)} seeds | {wind} wind · {reward} reward",
        fontsize=12, fontweight='bold', y=1.02
    )

    def bar_plot(ax, mkey, ylabel, title):
        means = [stats[k][mkey]['mean'] for k in dqn_keys]
        stds  = [stats[k][mkey]['std']  for k in dqn_keys]
        bars  = ax.bar(x, means, yerr=stds, capsize=4, color=dqn_colors,
                       edgecolor='white', width=0.55,
                       error_kw={'elinewidth': 1.5, 'ecolor': '#2c3e50'})
        valid = [m for m in means if not np.isnan(m)]
        for bar, m, s in zip(bars, means, stds):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(s) + (max(valid, default=1) * 0.02),
                        f"{m:.2f}", ha='center', va='bottom', fontsize=8.5, color='#2c3e50')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(dqn_labels, rotation=20, ha='right', fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
        if valid:
            ax.set_ylim(0, max(valid) * 1.45)
        ax.annotate('↓ better', xy=(0.98, 0.95), xycoords='axes fraction',
                    ha='right', fontsize=8, color='#7f8c8d', style='italic')

    bar_plot(axes[0], 'late_rolling_std', 'Reward Std (late phase)', 'Late Rolling Std')
    # Multi-seed variance
    mv = [stats[k]['multiseed_reward_var'] for k in dqn_keys]
    bars = axes[1].bar(x, mv, color=dqn_colors, edgecolor='white', width=0.55)
    valid = [v for v in mv if not np.isnan(v)]
    for bar, m in zip(bars, mv):
        if not np.isnan(m):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + (max(valid, default=1) * 0.02),
                         f"{m:.2f}", ha='center', va='bottom', fontsize=8.5, color='#2c3e50')
    axes[1].set_title('Multi-seed Reward Variance', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Var(reward)', fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(dqn_labels, rotation=20, ha='right', fontsize=8)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4); axes[1].set_axisbelow(True)
    if valid:
        axes[1].set_ylim(0, max(valid) * 1.45)
    axes[1].annotate('↓ better', xy=(0.98, 0.95), xycoords='axes fraction',
                     ha='right', fontsize=8, color='#7f8c8d', style='italic')

    bar_plot(axes[2], 'negative_spikes', '# negative spikes', 'Downward Spikes Count')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ 稳定性指标图已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Q 过估计图
# ══════════════════════════════════════════════════════════════════════════════
def plot_q_overestimation(stats, all_results, path, seeds, wind, reward):
    dqn_keys   = [c[0] for c in DQN_CONFIGS]
    dqn_labels = [c[1] for c in DQN_CONFIGS]
    dqn_colors = [c[2] for c in DQN_CONFIGS]
    x = np.arange(len(dqn_keys))

    q_means = [stats[k]['q_overestimation']['mean'] for k in dqn_keys]
    q_stds  = [stats[k]['q_overestimation']['std']  for k in dqn_keys]
    q_vals  = [stats[k]['q_overestimation']['vals'] for k in dqn_keys]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    bars = ax.bar(x, q_means, yerr=q_stds, capsize=6,
                  color=dqn_colors, edgecolor='white', width=0.5, alpha=0.85,
                  error_kw={'elinewidth': 2, 'ecolor': '#2c3e50'})
    for xi, vals in enumerate(q_vals):
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(xi + jitter, vals, color=dqn_colors[xi],
                   s=30, zorder=5, alpha=0.6, edgecolors='white', linewidths=0.5)
    for bar, m, s in zip(bars, q_means, q_stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    (bar.get_height() if m >= 0 else 0) + abs(s) + 0.05,
                    f"{m:+.3f}\n±{s:.3f}",
                    ha='center', va='bottom', fontsize=9, color='#2c3e50')
    ax.axhline(0.0, color='#e74c3c', linestyle='--', lw=1.5, alpha=0.7, label='Bias = 0')
    ax.set_title(
        f"Q-Value Estimation Bias (Q̂ − MC Return)\n"
        f"{len(seeds)} seeds · {wind} wind · {reward} reward",
        fontsize=11, fontweight='bold'
    )
    ax.set_ylabel('Q̂ − MC Return', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(dqn_labels, fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    all_flat = [v for vlist in q_vals for v in vlist if not np.isnan(v)]
    if all_flat:
        vmin, vmax = min(all_flat), max(all_flat)
        margin = max(abs(vmax - vmin) * 0.25, 0.5)
        ax.set_ylim(vmin - margin, vmax + margin)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ Q过估计图已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 样本效率图
# ══════════════════════════════════════════════════════════════════════════════
def plot_sample_efficiency(all_results, path, seeds, wind, reward):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for key, label, color in DQN_CONFIGS:
        seqs = [np.array(r[key]['rewards']) for r in all_results]
        min_len = min(len(s) for s in seqs)
        matrix = np.stack([
            pd.Series(s[:min_len]).expanding().mean().values for s in seqs
        ])
        mean = matrix.mean(axis=0)
        std  = matrix.std(axis=0, ddof=1)
        eps  = np.arange(1, len(mean) + 1)
        ax.plot(eps, mean, color=color, lw=2, label=label)
        ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.15)

    basic_final = np.mean([r['basic']['rewards'][-200:] for r in all_results])
    ax.axhline(basic_final, color='#2980b9', lw=1, ls='--', alpha=0.5,
               label=f'Basic final level ({basic_final:.1f})')
    ax.set_xlabel('Episode'); ax.set_ylabel('Expanding mean reward')
    ax.set_title(f'Sample Efficiency — {wind} wind · {reward} reward')
    ax.legend(); plt.tight_layout(); plt.savefig(path); plt.close()
    print(f"✅ 样本效率图已保存: {path}")
