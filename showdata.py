# showdata.py — 终端汇总打印（三算法对比版）

import numpy as np
from experiment import DQN_ALGOS

DQN_KEYS   = [k for k, _, _ in DQN_ALGOS]
DQN_LABELS = {k: l for k, l, _ in DQN_ALGOS}


def print_summary(stats, seeds, WIND, REWARD):
    print(f"\n{'='*85}")
    print(f"  Multi-seed Summary  |  {len(seeds)} seeds: {seeds}")
    print(f"  Wind={WIND}  Reward={REWARD}")
    print(f"{'='*85}")

    # 路径质量
    print(f"\n  ── 路径质量指标 ──────────────────────────────────────────────────────────")
    print(f"  {'Algorithm':18} | {'Energy':>14} | {'Time(s)':>14} | {'Steps':>10} | {'SuccRate':>8}")
    print("  " + "-" * 75)
    for key in ['vi'] + DQN_KEYS:
        s = stats[key]
        label = 'VI' if key == 'vi' else DQN_LABELS[key]
        print(f"  {label:18} | "
              f"{s['energy']['mean']:6.3f}±{s['energy']['std']:5.3f} | "
              f"{s['time']['mean']:7.1f}±{s['time']['std']:5.1f} | "
              f"{s['steps']['mean']:4.1f}±{s['steps']['std']:3.1f} | "
              f"{s['success_rate']*100:7.1f}%")

    # 学习效率
    print(f"\n  ── 学习效率指标 ──────────────────────────────────────────────────────────")
    print(f"  {'Algorithm':18} | {'AUC':>12} | {'FirstReach':>10} | {'ConvEp':>8} | {'TrainTime(s)':>12}")
    print("  " + "-" * 75)
    for key in DQN_KEYS:
        s = stats[key]
        print(f"  {DQN_LABELS[key]:18} | "
              f"{s['auc']['mean']:+8.2f}±{s['auc']['std']:5.2f} | "
              f"{s['first_reach_ep']['mean']:6.0f}±{s['first_reach_ep']['std']:4.0f} | "
              f"{s['convergence_ep']['mean']:4.0f}±{s['convergence_ep']['std']:3.0f} | "
              f"{s['train_time']['mean']:8.1f}±{s['train_time']['std']:4.1f}")

    # 训练稳定性
    print(f"\n  ── 训练稳定性指标 ────────────────────────────────────────────────────────")
    print(f"  {'Algorithm':18} | {'LateStd':>12} | {'MultiSeedVar':>14} | {'NegSpikes':>10}")
    print("  " + "-" * 75)
    for key in DQN_KEYS:
        s = stats[key]
        print(f"  {DQN_LABELS[key]:18} | "
              f"{s['late_rolling_std']['mean']:6.3f}±{s['late_rolling_std']['std']:5.3f} | "
              f"{s['multiseed_reward_var']:14.3f} | "
              f"{s['negative_spikes']['mean']:5.1f}±{s['negative_spikes']['std']:4.1f}")

    # Q估计偏差
    print(f"\n  ── Q估计偏差（Q̂−MC bias）────────────────────────────────────────────────")
    print(f"  {'Algorithm':18} | {'Bias':>14} |  说明")
    print("  " + "-" * 75)
    for key in DQN_KEYS:
        s = stats[key]
        bias = s['q_overestimation']['mean']
        if np.isnan(bias):      note = 'N/A'
        elif bias > 2.0:        note = '过估计明显 ↑'
        elif bias > 0.5:        note = '轻微过估计'
        elif bias > -0.5:       note = '基本无偏'
        else:                   note = '欠估计'
        print(f"  {DQN_LABELS[key]:18} | "
              f"{bias:+8.3f}±{s['q_overestimation']['std']:5.3f}  |  {note}")

    # 相对改进（以 Basic 为基准）
    print(f"\n  ── 相对改进（vs DQN-Basic）──────────────────────────────────────────────")
    metrics_labels = [
        ('auc',              'AUC ↑'),
        ('first_reach_ep',   'FirstReach ↓'),
        ('convergence_ep',   'ConvEp ↓'),
        ('late_rolling_std', 'LateStd ↓'),
        ('negative_spikes',  'Spikes ↓'),
        ('q_overestimation', 'Q̂/MC ↓'),
    ]
    bv_map = {m: stats['basic'][m]['mean'] for m, _ in metrics_labels}
    header = f"  {'Metric':20}"
    for key in DQN_KEYS[1:]:   # 跳过 basic 自身
        header += f"  {DQN_LABELS[key]:>14}"
    print(header)
    print("  " + "-" * 75)
    for mkey, mlabel in metrics_labels:
        bv = bv_map[mkey]
        row = f"  {mlabel:20}"
        for key in DQN_KEYS[1:]:
            iv = stats[key][mkey]['mean']
            if not np.isnan(bv) and not np.isnan(iv) and abs(bv) > 1e-8:
                pct = (iv - bv) / abs(bv) * 100
                row += f"  {pct:+13.1f}%"
            else:
                row += f"  {'N/A':>14}"
        print(row)

    print(f"\n{'='*85}")
