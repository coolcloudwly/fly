import numpy as np
# ══════════════════════════════════════════════════════════════════════════════
# 终端汇总打印（扩展版）
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(stats, seeds,WIND,REWARD):
    name_map = {'vi': 'VI', 'basic': 'DQN-Basic', 'improved': 'DQN-Improved'}
    print(f"\n{'='*80}")
    print(f"  Multi-seed Summary  |  {len(seeds)} seeds: {seeds}")
    print(f"  Wind={WIND}  Reward={REWARD}")
    print(f"{'='*80}")

    # 路径质量
    print(f"\n  ── 路径质量指标 ──────────────────────────────────────────────────────")
    print(f"  {'Algorithm':15} | {'Energy':>14} | {'Time(s)':>14} | {'Steps':>10} | {'SuccRate':>8}")
    print("  " + "-" * 70)
    for algo in ['vi', 'basic', 'improved']:
        s = stats[algo]
        print(f"  {name_map[algo]:15} | "
              f"{s['energy']['mean']:6.3f}±{s['energy']['std']:5.3f} | "
              f"{s['time']['mean']:7.1f}±{s['time']['std']:5.1f} | "
              f"{s['steps']['mean']:4.1f}±{s['steps']['std']:3.1f} | "
              f"{s['success_rate']*100:7.1f}%")

    # 学习效率
    print(f"\n  ── 学习效率指标（B vs I）──────────────────────────────────────────────")
    print(f"  {'Algorithm':15} | {'AUC':>12} | {'FirstReach':>10} | {'ConvEp':>8} | {'TrainTime(s)':>12}")
    print("  " + "-" * 70)
    for algo in ['basic', 'improved']:
        s = stats[algo]
        print(f"  {name_map[algo]:15} | "
              f"{s['auc']['mean']:+8.2f}±{s['auc']['std']:5.2f} | "
              f"{s['first_reach_ep']['mean']:6.0f}±{s['first_reach_ep']['std']:4.0f} | "
              f"{s['convergence_ep']['mean']:4.0f}±{s['convergence_ep']['std']:3.0f} | "
              f"{s['train_time']['mean']:8.1f}±{s['train_time']['std']:4.1f}")

    # 训练稳定性
    print(f"\n  ── 训练稳定性指标（B vs I）────────────────────────────────────────────")
    print(f"  {'Algorithm':15} | {'LateStd':>12} | {'MultiSeedVar':>14} | {'NegSpikes':>10}")
    print("  " + "-" * 70)
    for algo in ['basic', 'improved']:
        s = stats[algo]
        print(f"  {name_map[algo]:15} | "
              f"{s['late_rolling_std']['mean']:6.3f}±{s['late_rolling_std']['std']:5.3f} | "
              f"{s['multiseed_reward_var']:14.3f} | "
              f"{s['negative_spikes']['mean']:5.1f}±{s['negative_spikes']['std']:4.1f}")

    # 决策质量
    print(f"\n  ── 决策质量指标（B vs I）──────────────────────────────────────────────")
    # showdata.py
    print(f"  {'Algorithm':15} | {'Q̂−MC Bias':>14} |  理论含义")
    print("  " + "-" * 70)
    for algo in ['basic', 'improved']:
        s = stats[algo]
        # showdata.py — 决策质量那段改为：
        ratio = s['q_overestimation']['mean']
        if np.isnan(ratio):
            note = 'N/A'
        elif ratio > 2.0:
            note = '过估计明显 ↑'
        elif ratio > 0.5:
            note = '轻微过估计'
        elif ratio > -0.5:
            note = '基本无偏 ok'
        else:
            note = '欠估计（Double DQN 效果）'
        print(f"  {name_map[algo]:15} | "
              f"{ratio:+8.3f}±{s['q_overestimation']['std']:5.3f}  |  {note}")

    # 相对改进
    print(f"\n  ── 相对改进（Improved vs Basic）──────────────────────────────────────")
    for metric, label in [('auc', 'AUC'), ('first_reach_ep', 'FirstReach↓'),
                           ('convergence_ep', 'ConvEp↓'), ('late_rolling_std', 'LateStd↓'),
                           ('negative_spikes', 'Spikes↓'), ('q_overestimation', 'Q̂/MC↓')]:
        bv = stats['basic'][metric]['mean']
        iv = stats['improved'][metric]['mean']
        if not np.isnan(bv) and not np.isnan(iv) and abs(bv) > 1e-8:
            pct = (iv - bv) / abs(bv) * 100
            '''
            sign = '+' if pct > 0 else ''
            print(f"    {label:20s}  {sign}{pct:+.1f}%")
            '''
            # 方案 A：去掉 sign 变量，只用 f-string 格式符
            print(f"    {label:20s}  {pct:+.1f}%")

    print(f"\n{'='*80}")

