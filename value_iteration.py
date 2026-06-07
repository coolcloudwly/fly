# value_iteration.py — 值迭代（VI）求最优策略（性能优化版）
#
# 优化点：
#   1. 内层循环：把 env.reward() 调用提到 ai 循环外的字典缓存，
#      避免同一格同一动作的奖励被重复计算
#   2. 用 np.full(-inf) + argmax 替代 Python max()，减少 Python 对象开销
#   3. 提前绑定 env 方法到局部变量

import numpy as np
from typing import Tuple, List, Dict
from config import VI_GAMMA, VI_THETA, VI_MAX_ITER, EXTRACT_MAX_STEPS, GOAL_CELL


def value_iteration(env) -> Tuple[np.ndarray, Dict]:
    G   = env.G
    n_s = G * G
    n_a = env.n_actions
    T   = env.get_T()

    V      = np.zeros(n_s)
    policy = {}

    # 预缓存所有 (state, action) 的奖励，避免 VI 迭代中重复计算
    # reward() 是纯函数（只依赖风场和格子），可以安全缓存
    reward_cache: dict = {}
    ACTIONS = env.ACTIONS
    idx2s   = env.idx2s
    s2idx   = env.s2idx
    reward  = env.reward
    obstacles = env.obstacles
    goal      = env.goal

    for s in range(n_s):
        col, row = idx2s(s)
        if (col, row) in obstacles or (col, row) == goal:
            continue
        for ai in range(n_a):
            if (col, row, ai) in T:
                reward_cache[(col, row, ai)] = reward(col, row, ACTIONS[ai])

    for iteration in range(VI_MAX_ITER):
        delta = 0.0
        for s in range(n_s):
            col, row = idx2s(s)
            if (col, row) == goal or (col, row) in obstacles:
                continue

            q_vals = np.full(n_a, -np.inf)

            for ai in range(n_a):
                trans = T.get((col, row, ai))
                #如果当前状态 s + 动作 ai 是非法动作（终点or障碍物），没有转移概率，直接跳过这个动作，不计算 Q 值。
                if trans is None:
                    continue
                r     = reward_cache.get((col, row, ai), 0.0)
                exp_v = sum(p * V[s2idx(nc, nr)] for (nc, nr), p in trans.items())
                q_vals[ai] = r + VI_GAMMA * exp_v

            # 当前格子四面全是障碍物，所有动作全都无效走不了，这个状态直接跳过、不更新 V。
            if np.all(q_vals == -np.inf):
                continue

            v_new = q_vals.max()
            delta = max(delta, abs(v_new - V[s]))
            V[s]  = v_new
            policy[s] = int(q_vals.argmax())

        if delta < VI_THETA:
            print(f"值迭代收敛，共 {iteration+1} 次迭代，Δ={delta:.2e}")
            break
    else:
        print(f"值迭代达到最大迭代次数 {VI_MAX_ITER}，最终 Δ={delta:.2e}")

    return V, policy


def extract_path(env, policy: Dict) -> List[Tuple]:
    env.reset()
    col, row = env.start
    path = [(col, row)]
    visit_count = {(col, row): 1}

    for _ in range(EXTRACT_MAX_STEPS):
        if (col, row) == env.goal:
            break

        s  = env.s2idx(col, row)
        ai = policy.get(s)
        if ai is None:
            print(f"  [extract_path] 状态({col},{row})无策略，终止")
            break

        s_next, _, done = env.step(ai)
        col_next, row_next = env.idx2s(s_next)
        path.append((col_next, row_next))
        visit_count[(col_next, row_next)] = visit_count.get((col_next, row_next), 0) + 1
        col, row = col_next, row_next

        if visit_count[(col, row)] > 3:
            print(f"  [extract_path] 检测到循环于 ({col},{row})，提前终止")
            break
        if done:
            break

    return path
