# epsilon_strategy.py — ε-贪婪探索策略（简化版，去除风场复杂度EMA）
#
# improved 模式使用固定超参数，去掉 nIQR / EMA / get_adaptive_params 相关逻辑。
# 保留奖励滚动均值微调（好回合降ε / 差回合升ε）和保底衰减通道。

import random
import numpy as np
import torch
from typing import Optional
from config import EPS_INIT, EPS_MIN, TOTAL_STEPS

# ── improved 模式固定超参（原 get_adaptive_params 取 complexity=0.5 时的中性值）──
FIXED_DECAY_EXP = -3.0   # 原插值中点：(-3.5 + -2.5) / 2
FIXED_EPS_LO    =  0.70  # 原插值中点：(0.65 + 0.75) / 2
FIXED_PER_ALPHA =  0.45  # 原插值中点：(0.55 + 0.35) / 2
FIXED_TGT_FREQ  =  27    # 原插值中点：(20 + 35) / 2，取整
FIXED_GRAD_CLIP =  0.65  # 原插值中点：(0.80 + 0.50) / 2


# ── ε 更新 ────────────────────────────────────────────────────
def update_epsilon_basic(step: int) -> float:
    """基础版：指数衰减。"""
    progress = min(step / TOTAL_STEPS, 1.0)
    return float(max(EPS_INIT * np.exp(-2.0 * progress), EPS_MIN))


def update_epsilon_episode(
    episode: int,
    n_episodes: int,
    rew_curr: float,
    rew_rolling_avg: Optional[float],
) -> float:
    """
    改进版（简化）：固定衰减速度 + 奖励微调 + 保底衰减通道。

    去除了 complexity_ema 参数，所有超参改为固定值。
    保留奖励微调因子和 eps_floor 兜底，维持训练稳定性。
    """
    progress = min(episode / n_episodes, 1.0)

    # 主衰减（固定指数）
    eps_base = EPS_INIT * np.exp(FIXED_DECAY_EXP * progress)

    # 奖励微调因子：好回合降ε，差回合升ε，范围 [eps_lo, 1.05]
    if rew_rolling_avg is not None and not np.isnan(rew_rolling_avg):
        normalizer = abs(rew_rolling_avg) + 5.0
        ratio = 1.0 / (1.0 + np.exp(
            -(rew_curr - rew_rolling_avg) / (normalizer * 0.5 + 1e-8)
        ))
        eps_rew = float(np.clip(1.05 - (1.05 - FIXED_EPS_LO) * ratio, FIXED_EPS_LO, 1.05))
    else:
        eps_rew = 1.0

    # 保底衰减通道：确保 ε 随训练单调下降，防止锁死
    eps_floor = EPS_INIT * np.exp(-1.5 * progress)

    eps_combined = eps_base * eps_rew
    return float(np.clip(min(eps_combined, eps_floor * 1.5), EPS_MIN, EPS_INIT))


def select_action(state_tensor, eps: float, model) -> int:
    if random.random() < eps:
        return random.randint(0, 7)
    with torch.no_grad():
        return int(model(state_tensor.unsqueeze(0)).argmax().item())


# 兼容旧接口
def update_epsilon_improved(step, rew_curr, rew_rolling_avg):
    progress = min(step / TOTAL_STEPS, 1.0)
    episode_equiv = int(progress * 1000)
    return update_epsilon_episode(episode_equiv, 1000, rew_curr, rew_rolling_avg)


if __name__ == "__main__":
    print("=== 自检 ===")
    np.random.seed(42)
    for wind, rew_fn in [
        ("uniform",  lambda ep: -8  + ep*0.05 + np.random.randn()*2),
        ("circular", lambda ep: -20 + ep*0.03 + np.random.randn()*12),
    ]:
        eps_vals = []
        window = []
        for ep in range(1, 1001):
            r = rew_fn(ep)
            window.append(r)
            if len(window) > 30: window.pop(0)
            mu = np.mean(window)
            eps_vals.append(update_epsilon_episode(ep, 1000, r, mu))
        f = np.mean(eps_vals[:100])
        m = np.mean(eps_vals[450:550])
        l = np.mean(eps_vals[-100:])
        print(f"{wind}: ε前/中/后={f:.3f}/{m:.3f}/{l:.3f} 单调={f>m>l}")
        assert f > m > l, f"❌ {wind} ε未单调"
    print("✅ 自检通过")
