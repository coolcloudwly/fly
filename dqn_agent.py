# dqn_agent.py — DQN 智能体（v5 自适应风场版）

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import numpy as np
from config import (
    DQN_LR, DQN_GAMMA, DQN_BUF, DQN_BATCH, DQN_TGT_FREQ,
    EPS_INIT, EPS_MIN, N_EPISODES, TOTAL_STEPS,
)
from models import QNet, ImprovedDuelingQNet, ReplayBuffer, ImprovedPER
from epsilon_strategy import (
    update_epsilon_basic, update_epsilon_episode,
    update_complexity_ema, get_adaptive_params, select_action,
)

_ROLLING_WINDOW = 30


class DQNAgent:
    """
    DQN 智能体 v5（自适应风场版）。

    eps_mode='basic'    : 标准 QNet + 指数衰减 ε（不变）
    eps_mode='improved' : ImprovedDuelingQNet + Double DQN + ImprovedPER
                          + 在线风场复杂度检测，连续插值调整：
                            · ε 衰减速度（decay_exp）
                            · ε 微调下界（eps_lo）
                            · PER 优先级指数（per_alpha）
                            · 目标网络同步频率（tgt_freq）
                            · 梯度裁剪阈值（grad_clip）

    uniform 风场 → 激进模式（快收敛，强节能，α=0.55，tgt=20）
    circular风场 → 保守模式（稳健收敛，α=0.35，tgt=35）
    混合/未知    → 自动插值，无需人工指定
    """

    def __init__(self, eps_mode: str = 'basic'):
        assert eps_mode in ('basic', 'improved')
        self.eps_mode = eps_mode

        if eps_mode == 'improved':
            self.q_net   = ImprovedDuelingQNet()
            self.tgt_net = ImprovedDuelingQNet()
        else:
            self.q_net   = QNet()
            self.tgt_net = QNet()

        self.tgt_net.load_state_dict(self.q_net.state_dict())
        self.tgt_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=DQN_LR)
        self.loss_fn   = nn.MSELoss()

        self.eps      = EPS_INIT
        self.gamma    = DQN_GAMMA
        self.batch_sz = DQN_BATCH

        self._episode    = 0
        self._n_episodes = N_EPISODES

        self._rew_window: collections.deque = collections.deque(maxlen=_ROLLING_WINDOW)

        if eps_mode == 'improved':
            self.buffer = ImprovedPER(DQN_BUF)
            self._beta         = 0.4
            self._beta_anneal  = (1.0 - 0.4) / max(TOTAL_STEPS, 1)

            # 风场复杂度 EMA（初始中性 0.5，约50回合后收敛到真实风场）
            self._complexity_ema = 0.5
            # 当前自适应参数（初始中性）
            self._adaptive = get_adaptive_params(self._complexity_ema)
            # 目标网络同步频率由自适应参数决定
            self.tgt_freq = self._adaptive['tgt_freq']
        else:
            self.buffer   = ReplayBuffer(DQN_BUF)
            self.tgt_freq = DQN_TGT_FREQ

    # ── 动作选择 ──────────────────────────────────────────────
    def act(self, state_tensor) -> int:
        return select_action(state_tensor, self.eps, self.q_net)

    # ── ε 更新（每回合结束调用，传入完整回合奖励）────────────
    def update_eps(self, step: int, rew_curr: float = 0.0):
        self._episode += 1
        self._rew_window.append(rew_curr)
        rolling_avg = float(np.mean(self._rew_window)) if self._rew_window else None

        if self.eps_mode == 'basic':
            self.eps = update_epsilon_basic(step)
        else:
            # ① 更新风场复杂度 EMA
            self._complexity_ema = update_complexity_ema(
                list(self._rew_window), self._complexity_ema
            )
            # ② 重新计算自适应参数（EMA平滑，参数缓慢漂移）
            self._adaptive = get_adaptive_params(self._complexity_ema)
            # ③ tgt_freq 随复杂度动态调整
            self.tgt_freq = self._adaptive['tgt_freq']
            # ④ 更新 PER alpha（每回合重新传入 buffer）
            self.buffer.alpha = self._adaptive['per_alpha']
            # ⑤ 更新 ε
            self.eps = update_epsilon_episode(
                self._episode, self._n_episodes,
                rew_curr, rolling_avg,
                self._complexity_ema,
            )

    # ── 存储经验 ──────────────────────────────────────────────
    def store(self, s, a, r, s_, done):
        self.buffer.push(s, a, r, s_, done)

    # ── 学习 ──────────────────────────────────────────────────
    def learn(self, step: int):
        if len(self.buffer) < self.batch_sz:
            return None

        if self.eps_mode == 'improved':
            beta = min(1.0, self._beta + step * self._beta_anneal)
            states, actions, rewards, next_states, dones, indices, weights = \
                self.buffer.sample(self.batch_sz, beta=beta)

            with torch.no_grad():
                best_actions = self.q_net(next_states).argmax(1, keepdim=True)
                q_next = self.tgt_net(next_states).gather(1, best_actions).squeeze(1)
                q_tgt  = rewards + self.gamma * q_next * (1 - dones)

            q_curr    = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            td_errors = (q_curr - q_tgt).detach()
            loss = (weights * (q_curr - q_tgt) ** 2).mean()
            self.buffer.update_priorities(indices, td_errors.abs().numpy())

        else:
            states, actions, rewards, next_states, dones = \
                self.buffer.sample(self.batch_sz)
            with torch.no_grad():
                q_next = self.tgt_net(next_states).max(1)[0]
                q_tgt  = rewards + self.gamma * q_next * (1 - dones)
            q_curr = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss   = self.loss_fn(q_curr, q_tgt)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：改进版由自适应参数决定，basic版固定1.0
        clip_val = self._adaptive['grad_clip'] if self.eps_mode == 'improved' else 1.0
        nn.utils.clip_grad_norm_(self.q_net.parameters(), clip_val)
        self.optimizer.step()

        if step % self.tgt_freq == 0:
            self.tgt_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ── 当前风场诊断（供调试/日志使用）──────────────────────
    def wind_profile(self) -> str:
        if self.eps_mode != 'improved':
            return 'N/A (basic mode)'
        c = self._complexity_ema
        p = self._adaptive
        mode = 'uniform' if c < 0.35 else ('circular' if c > 0.6 else 'mixed')
        return (f"complexity={c:.3f}({mode}) "
                f"decay={p['decay_exp']:.2f} α={p['per_alpha']:.2f} "
                f"tgt={p['tgt_freq']} clip={p['grad_clip']:.2f}")

    # ── 持久化 ────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({'q_net': self.q_net.state_dict(),
                    'eps_mode': self.eps_mode}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.q_net.load_state_dict(ckpt['q_net'])
        self.tgt_net.load_state_dict(ckpt['q_net'])
