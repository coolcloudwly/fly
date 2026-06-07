# trainer.py — 训练主循环（性能优化版）
#
# 优化点：
#   1. 减少每步的 Python 属性查询：把热路径中的 agent.act/store/learn
#      提前绑定为局部变量（Python 局部变量查找比属性查找快 ~30%）
#   2. state_tensor() 已在 mdp_env 中缓存，trainer 无需额外优化
#   3. verbose 打印改为条件判断放在循环内顶部，减少函数调用开销

from __future__ import annotations
import numpy as np
from typing import Optional


def train(
    env,
    agent,
    n_episodes: int,
    max_steps: int,
    verbose: bool = True,
) -> dict:
    """
    通用训练循环，兼容 DQNAgent（basic/improved）与 DAEAgent。

    返回
    ----
    history : {
        'rewards'  : List[float]
        'eps'      : List[float]
        'losses'   : List[float]
        'steps'    : List[int]
    }
    """
    history = {'rewards': [], 'eps': [], 'losses': [], 'steps': []}
    global_step = 0
    rew_best: Optional[float] = None

    # ── 热路径函数绑定（减少属性查找开销）─────────────────────
    is_dqn     = hasattr(agent, 'eps_mode')
    act_fn     = agent.act
    store_fn   = agent.store
    learn_fn   = agent.learn
    update_eps = getattr(agent, 'update_eps', None)
    state_tensor_fn = env.state_tensor
    step_fn    = env.step

    # 预分配 loss 列表避免每回合 append 扩容
    losses_buf: list = []

    for ep in range(1, n_episodes + 1):
        s   = env.reset()
        s_t = state_tensor_fn(s)
        rew_ep = 0.0
        losses_buf.clear()

        for t in range(max_steps):
            global_step += 1

            # 动作选择
            if is_dqn:
                a = act_fn(s_t)
            else:
                a = act_fn(s_t, global_step)

            # 环境交互
            s_, r, done = step_fn(a)
            s_t_ = state_tensor_fn(s_)
            rew_ep += r

            # 存储 & 学习
            store_fn(s_t, a, r, s_t_, float(done))
            loss = learn_fn(global_step)
            if loss is not None:
                losses_buf.append(loss)

            s_t = s_t_
            if done:
                break

        # 回合结束
        if rew_best is None or rew_ep > rew_best:
            rew_best = rew_ep

        if update_eps is not None:
            update_eps(global_step, rew_curr=rew_ep)

        history['rewards'].append(rew_ep)
        history['eps'].append(getattr(agent, 'eps', 0.0))
        history['losses'].append(float(np.mean(losses_buf)) if losses_buf else 0.0)
        history['steps'].append(t + 1)

        if verbose and ep % 50 == 0:
            avg_rew = np.mean(history['rewards'][-50:])
            eps_now = getattr(agent, 'eps', 0.0)
            print(f"Ep {ep:4d}/{n_episodes}  "
                  f"avg_rew(50)={avg_rew:+.3f}  "
                  f"best={rew_best:+.3f}  "
                  f"ε={eps_now:.4f}  "
                  f"step={global_step}")

    return history
