# experiment.py — 单种子实验流程（三算法对比版）
#
# 算法：VI / DQN-Basic / DQN-eps_only / DQN-eps_per

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

PATH_METRICS       = ['energy', 'time', 'distance', 'steps']
EFFICIENCY_METRICS = ['auc', 'first_reach_ep', 'convergence_ep', 'train_time']
STABILITY_METRICS  = ['late_rolling_std', 'negative_spikes']
QUALITY_METRICS    = ['q_overestimation']
ALL_SCALAR_METRICS = PATH_METRICS + EFFICIENCY_METRICS + STABILITY_METRICS + QUALITY_METRICS

# 三个 DQN 算法的 key / label / eps_mode 映射
DQN_ALGOS = [
    ('basic',    'DQN-Basic',    'basic'),
    ('eps_only', 'DQN-eps_only', 'eps_only'),
    ('eps_per',  'DQN-eps_per',  'eps_per'),
]


def make_env(reward_mode, wind, density=None, obs_seed=None):
    env = UASWindEnv(reward_mode=reward_mode)
    if density is not None or obs_seed is not None:
        env.set_random_obstacles(
            density=density  if density  is not None else OBSTACLE_DENSITY,
            seed=obs_seed    if obs_seed is not None else OBSTACLE_SEED,
        )
    if wind == "circular":
        env.set_circular_wind()
    return env


def get_path(agent, env):
    env.reset()
    c, r    = env.start
    path    = [(c, r)]
    actions = []
    visit_count = {(c, r): 1}

    for _ in range(MAX_PATH_STEPS):
        if (c, r) == env.goal:
            break
        s_t = env.state_tensor(env.s2idx(c, r))
        with torch.no_grad():
            a = int(agent.q_net(s_t.unsqueeze(0)).argmax().item())
        a_name = env.ACTIONS[a]
        s_next, _, done = env.deterministic_step(a)
        c_next, r_next  = env.idx2s(s_next)
        actions.append(a_name)
        path.append((c_next, r_next))
        visit_count[(c_next, r_next)] = visit_count.get((c_next, r_next), 0) + 1
        c, r = c_next, r_next
        if visit_count[(c, r)] > 3 or done:
            break
    return path, actions


def vi_path_actions(env, path):
    delta_inv = {v: k for k, v in env.DELTAS.items()}
    actions   = []
    for (c0, r0), (c1, r1) in zip(path[:-1], path[1:]):
        a_name = delta_inv.get((c1 - c0, r1 - r0))
        if a_name is None:
            break
        actions.append(a_name)
    return actions


def _run_dqn(key, eps_mode, seed, wind, reward, n_ep, density, obs_seed):
    """训练单个 DQN 变体，返回指标 dict。"""
    set_seed(seed)
    env   = make_env(reward, wind, density, obs_seed)
    agent = DQNAgent(eps_mode=eps_mode)
    t0    = time.time()
    hist  = train(env, agent, n_ep, MAX_STEPS, verbose=False)
    train_time = time.time() - t0

    path, acts = get_path(agent, env)
    m = env.path_metrics(path, acts)

    rews = hist['rewards']
    sm   = smooth(rews, 30)
    m['rewards']          = rews
    m['train_time']       = train_time
    m['auc']              = compute_auc(rews)
    m['first_reach_ep']   = compute_first_reach(rews)
    m['convergence_ep']   = compute_convergence_episode(rews)
    m['late_rolling_std'] = compute_late_rolling_std(rews)
    m['negative_spikes']  = compute_negative_spike_count(rews, sm)
    m['q_overestimation'] = compute_q_overestimation(agent, env, rews)
    m['degradation']      = float('nan')

    return m, path, env, agent


def run_one_seed(seed, wind, reward,
                 n_ep=N_EPISODES,
                 density=None, obs_seed=None):
    set_seed(seed)
    env_vi = make_env(reward, wind, density, obs_seed)

    # ── VI ────────────────────────────────────────────────────
    print(f"\n  [Seed={seed}] Value Iteration ...")
    t0 = time.time()
    V, vi_policy = value_iteration(env_vi)
    vi_path      = extract_path(env_vi, vi_policy)
    vi_acts      = vi_path_actions(env_vi, vi_path)
    vi_m         = env_vi.path_metrics(vi_path, vi_acts)
    vi_m['train_time'] = time.time() - t0
    vi_m['degradation'] = float('nan')

    result = {'vi': vi_m, '_paths': {'vi': (vi_path, env_vi)}, '_agents': {}}

    # ── 三个 DQN 变体 ─────────────────────────────────────────
    for key, label, eps_mode in DQN_ALGOS:
        print(f"  [Seed={seed}] {label} ...")
        m, path, env, agent = _run_dqn(key, eps_mode, seed, wind, reward,
                                        n_ep, density, obs_seed)
        result[key]               = m
        result['_paths'][key]     = (path, env)
        result['_agents'][key]    = agent

        print(f"  [Seed={seed}] {label}: "
              f"{'✅' if m['reached_goal'] else '❌'}{m['steps']}步  "
              f"AUC={m['auc']:+.2f}  Q̂/MC={m['q_overestimation']:.3f}")

    return result


def save_seed_data(seed, result, out_root):
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


def aggregate(all_results):
    algos = ['vi'] + [k for k, _, _ in DQN_ALGOS]
    stats = {}

    for algo in algos:
        stats[algo] = {}
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

        stats[algo]['success_rate'] = float(
            np.mean([r[algo]['reached_goal'] for r in all_results])
        )

        if algo != 'vi':
            reward_seqs = [np.array(r[algo]['rewards']) for r in all_results
                           if 'rewards' in r[algo]]
            if len(reward_seqs) >= 2:
                min_len    = min(len(s) for s in reward_seqs)
                mat        = np.stack([s[:min_len] for s in reward_seqs])
                stats[algo]['multiseed_reward_var'] = float(mat.var(axis=0).mean())
            else:
                stats[algo]['multiseed_reward_var'] = float('nan')
        else:
            stats[algo]['multiseed_reward_var'] = float('nan')

    return stats


def save_csv(stats, path):
    algo_keys  = ['vi'] + [k for k, _, _ in DQN_ALGOS]
    name_map   = {'vi': 'VI', 'basic': 'DQN-Basic',
                  'eps_only': 'DQN-eps_only', 'eps_per': 'DQN-eps_per'}
    header_cols = (
        ['Algorithm']
        + [f"{m}_mean,{m}_std" for m in PATH_METRICS]
        + ['SuccessRate']
        + [f"{m}_mean,{m}_std" for m in EFFICIENCY_METRICS]
        + ['multiseed_reward_var']
        + [f"{m}_mean,{m}_std" for m in STABILITY_METRICS]
        + [f"{m}_mean,{m}_std" for m in QUALITY_METRICS]
    )
    lines = [','.join(header_cols)]

    for algo in algo_keys:
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
