# experiment.py — 单种子实验流程、数据聚合与 CSV 保存
#
# 包含：
#   make_env()       — 环境构建（风场 + 障碍物配置）
#   get_path()       — 贪婪策略推理路径
#   vi_path_actions()— VI 路径动作名重建
#   run_one_seed()   — 单种子完整实验（VI + Basic + Improved）
#   save_seed_data() — 单种子数据持久化（.npz）
#   aggregate()      — 多种子统计聚合（均值 ± std）
#   save_csv()       — 汇总 CSV 输出

import os
import time
import numpy as np
import torch

from config import (
    N_EPISODES, MAX_STEPS, set_seed,
    OBSTACLE_DENSITY, OBSTACLE_SEED,
)
from mdp_env import UASWindEnv
from value_iteration import value_iteration, extract_path
from dqn_agent import DQNAgent
from trainer import train
from metrics import (
    smooth,
    compute_auc, compute_first_reach, compute_convergence_episode,
    compute_late_rolling_std, compute_negative_spike_count,
    compute_q_overestimation,
)

MAX_PATH_STEPS = 50

# ── 指标分组（供 aggregate / save_csv 使用）───────────────────────────────────
PATH_METRICS       = ['energy', 'time', 'distance', 'steps']
EFFICIENCY_METRICS = ['auc', 'first_reach_ep', 'convergence_ep', 'train_time']
STABILITY_METRICS  = ['late_rolling_std', 'negative_spikes']
QUALITY_METRICS    = ['q_overestimation']
ALL_SCALAR_METRICS = PATH_METRICS + EFFICIENCY_METRICS + STABILITY_METRICS + QUALITY_METRICS


# ══════════════════════════════════════════════════════════════════════════════
# 环境构建
# ══════════════════════════════════════════════════════════════════════════════
def make_env(reward_mode, wind, density=None, obs_seed=None):
    """
    构建 UASWindEnv。

    若 density 或 obs_seed 不为 None，则调用 set_random_obstacles()
    覆盖 config.OBSTACLE_CELLS 中的默认障碍物布局；
    否则使用 config.py 中 generate_random_obstacles() 预生成的默认布局。
    """
    env = UASWindEnv(reward_mode=reward_mode)
    if density is not None or obs_seed is not None:
        env.set_random_obstacles(
            density=density  if density  is not None else OBSTACLE_DENSITY,
            seed=obs_seed    if obs_seed is not None else OBSTACLE_SEED,
        )
    if wind == "circular":
        env.set_circular_wind()
    return env


# ══════════════════════════════════════════════════════════════════════════════
# 路径推理工具
# ══════════════════════════════════════════════════════════════════════════════
def get_path(agent, env):
    """贪婪策略推理路径（ε=0 等价）。"""
    env.reset()
    c, r    = env.start
    path    = [(c, r)]
    actions = []
    visit_count = {(c, r): 1}
    is_dae      = hasattr(agent, 'meta')
    policy_net  = agent.meta if is_dae else agent.q_net

    for _ in range(MAX_PATH_STEPS):
        if (c, r) == env.goal:
            break
        s_t = env.state_tensor(env.s2idx(c, r))
        with torch.no_grad():
            a = int(policy_net(s_t.unsqueeze(0)).argmax().item())
        a_name = env.ACTIONS[a]
        s_next, _, done = env.step(a)
        c_next, r_next  = env.idx2s(s_next)
        actions.append(a_name)
        path.append((c_next, r_next))
        visit_count[(c_next, r_next)] = visit_count.get((c_next, r_next), 0) + 1
        c, r = c_next, r_next
        if visit_count[(c, r)] > 3 or done:
            break
    return path, actions


def vi_path_actions(env, path):
    """从 VI 路径坐标序列还原动作名列表。"""
    delta_inv = {v: k for k, v in env.DELTAS.items()}
    actions   = []
    for (c0, r0), (c1, r1) in zip(path[:-1], path[1:]):
        a_name = delta_inv.get((c1 - c0, r1 - r0))
        if a_name is None:
            break
        actions.append(a_name)
    return actions


# ══════════════════════════════════════════════════════════════════════════════
# 单种子完整实验
# ══════════════════════════════════════════════════════════════════════════════
def run_one_seed(seed: int, wind: str, reward: str,
                 n_ep: int = N_EPISODES,
                 density: float = None, obs_seed: int = None) -> dict:
    """
    运行单个种子的完整实验：VI + DQN-Basic + DQN-Improved。

    返回结构化结果字典，包含路径指标、学习效率、稳定性、Q过估计等全部指标。
    """
    set_seed(seed)
    env = make_env(reward, wind, density, obs_seed)

    # ── 1. VI ─────────────────────────────────────────────────────────────────
    print(f"\n  [Seed={seed}] Value Iteration ...")
    t0_vi        = time.time()
    V, vi_policy = value_iteration(env)
    vi_path      = extract_path(env, vi_policy)
    vi_acts      = vi_path_actions(env, vi_path)
    vi_m         = env.path_metrics(vi_path, vi_acts)
    vi_m['train_time'] = time.time() - t0_vi
    print(f"train_time:{vi_m['train_time']}")

    # ── 2. DQN-Basic ──────────────────────────────────────────────────────────
    print(f"  [Seed={seed}] DQN-Basic ...")
    set_seed(seed)
    env_b   = make_env(reward, wind, density, obs_seed)
    agent_b = DQNAgent(eps_mode="basic")
    t0_b    = time.time()
    hist_b  = train(env_b, agent_b, n_ep, MAX_STEPS, verbose=False)
    train_time_b   = time.time() - t0_b
    path_b, acts_b = get_path(agent_b, env_b)
    m_b = env_b.path_metrics(path_b, acts_b)

    rews_b = hist_b['rewards']
    sm_b   = smooth(rews_b, 30)
    m_b['rewards']          = rews_b
    m_b['train_time']       = train_time_b
    m_b['auc']              = compute_auc(rews_b)
    m_b['first_reach_ep']   = compute_first_reach(rews_b)
    m_b['convergence_ep']   = compute_convergence_episode(rews_b)
    m_b['late_rolling_std'] = compute_late_rolling_std(rews_b)
    m_b['negative_spikes']  = compute_negative_spike_count(rews_b, sm_b)
    m_b['q_overestimation'] = compute_q_overestimation(agent_b, env_b, rews_b)
    m_b['degradation']      = float('nan')

    # ── 3. DQN-Improved ───────────────────────────────────────────────────────
    print(f"  [Seed={seed}] DQN-Improved ...")
    set_seed(seed)
    env_i   = make_env(reward, wind, density, obs_seed)
    agent_i = DQNAgent(eps_mode="improved")
    t0_i    = time.time()
    hist_i  = train(env_i, agent_i, n_ep, MAX_STEPS, verbose=False)
    train_time_i   = time.time() - t0_i
    path_i, acts_i = get_path(agent_i, env_i)
    m_i = env_i.path_metrics(path_i, acts_i)

    rews_i = hist_i['rewards']
    sm_i   = smooth(rews_i, 30)
    m_i['rewards']          = rews_i
    m_i['train_time']       = train_time_i
    m_i['auc']              = compute_auc(rews_i)
    m_i['first_reach_ep']   = compute_first_reach(rews_i)
    m_i['convergence_ep']   = compute_convergence_episode(rews_i)
    m_i['late_rolling_std'] = compute_late_rolling_std(rews_i)
    m_i['negative_spikes']  = compute_negative_spike_count(rews_i, sm_i)
    m_i['q_overestimation'] = compute_q_overestimation(agent_i, env_i, rews_i)
    m_i['degradation']      = float('nan')
    vi_m['degradation']     = float('nan')

    print(f"  [Seed={seed}] Done. "
          f"VI={vi_m['steps']}步 "
          f"Basic={'✅' if m_b['reached_goal'] else '❌'}{m_b['steps']}步 "
          f"Improved={'✅' if m_i['reached_goal'] else '❌'}{m_i['steps']}步")
    print(f"           AUC  Basic={m_b['auc']:+.2f}  Improved={m_i['auc']:+.2f}")
    print(f"           Q̂/MC Basic={m_b['q_overestimation']:.3f}  Improved={m_i['q_overestimation']:.3f}")

    return {
        'vi':       vi_m,
        'basic':    m_b,
        'improved': m_i,
        '_paths': {
            'vi':       (vi_path, env),
            'basic':    (path_b,  env_b),
            'improved': (path_i,  env_i),
        },
        '_agents': {
            'basic':    agent_b,
            'improved': agent_i,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 数据持久化
# ══════════════════════════════════════════════════════════════════════════════
def save_seed_data(seed: int, result: dict, out_root: str):
    """将单种子结果保存为 .npz 文件（不含路径/智能体对象）。"""
    seed_dir  = os.path.join(out_root, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    skip_keys = {'_paths', '_agents'}
    list_keys = {'rewards'}
    for algo, m in result.items():
        if algo in skip_keys:
            continue
        data = {}
        for k, v in m.items():
            if k in list_keys:
                data[k] = np.array(v)
            elif isinstance(v, (int, float, bool, np.integer, np.floating)):
                data[k] = np.array(v)
        if data:
            np.savez(os.path.join(seed_dir, f"{algo}.npz"), **data)


# ══════════════════════════════════════════════════════════════════════════════
# 统计聚合
# ══════════════════════════════════════════════════════════════════════════════
def aggregate(all_results: list) -> dict:
    """
    对所有种子结果做均值 ± 标准差。
    路径质量指标仅统计成功到达终点的种子。
    额外计算：跨 seed 奖励方差（multi-seed reward variance）。
    """
    algos = ['vi', 'basic', 'improved']
    stats = {}

    for algo in algos:
        stats[algo] = {}

        # 标量指标
        for metric in ALL_SCALAR_METRICS:
            vals = []
            for r in all_results:
                v = r[algo].get(metric, float('nan'))
                if metric in PATH_METRICS and not r[algo]['reached_goal']:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isnan(fv):
                    vals.append(fv)
            if vals:
                stats[algo][metric] = {
                    'mean': float(np.mean(vals)),
                    'std':  float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                    'vals': vals,
                }
            else:
                stats[algo][metric] = {'mean': float('nan'), 'std': float('nan'), 'vals': []}

        # 到达率
        stats[algo]['success_rate'] = float(
            np.mean([r[algo]['reached_goal'] for r in all_results])
        )

        # 多 seed 奖励方差
        if algo in ('basic', 'improved'):
            reward_seqs = [np.array(r[algo]['rewards']) for r in all_results
                           if 'rewards' in r[algo]]
            if len(reward_seqs) >= 2:
                min_len    = min(len(s) for s in reward_seqs)
                mat        = np.stack([s[:min_len] for s in reward_seqs])
                per_ep_var = mat.var(axis=0)
                stats[algo]['multiseed_reward_var'] = float(per_ep_var.mean())
            else:
                stats[algo]['multiseed_reward_var'] = float('nan')
        else:
            stats[algo]['multiseed_reward_var'] = float('nan')

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CSV 输出
# ══════════════════════════════════════════════════════════════════════════════
def save_csv(stats: dict, path: str):
    """将聚合统计写入 CSV 文件。"""
    header_cols = (
        ['Algorithm']
        + [f"{m}_mean,{m}_std" for m in PATH_METRICS]
        + ['SuccessRate']
        + [f"{m}_mean,{m}_std" for m in EFFICIENCY_METRICS]
        + ['multiseed_reward_var']
        + [f"{m}_mean,{m}_std" for m in STABILITY_METRICS]
        + [f"{m}_mean,{m}_std" for m in QUALITY_METRICS]
    )
    header   = ','.join(header_cols)
    name_map = {'vi': 'VI', 'basic': 'DQN-Basic', 'improved': 'DQN-Improved'}
    lines    = [header]

    for algo in ['vi', 'basic', 'improved']:
        s    = stats[algo]
        cols = [name_map[algo]]
        for m in PATH_METRICS:
            cols += [f"{s[m]['mean']:.4f}", f"{s[m]['std']:.4f}"]
        cols.append(f"{s['success_rate']:.2f}")
        for m in EFFICIENCY_METRICS:
            cols += [f"{s[m]['mean']:.4f}", f"{s[m]['std']:.4f}"]
        cols.append(f"{s['multiseed_reward_var']:.4f}")
        for m in STABILITY_METRICS:
            cols += [f"{s[m]['mean']:.4f}", f"{s[m]['std']:.4f}"]
        for m in QUALITY_METRICS:
            cols += [f"{s[m]['mean']:.4f}", f"{s[m]['std']:.4f}"]
        lines.append(','.join(cols))

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✅ CSV 已保存: {path}")
