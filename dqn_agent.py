# dqn_agent.py — DQN 智能体（三模式：basic / eps_only / eps_per）

import torch
import torch.nn as nn
import torch.optim as optim
import collections
import numpy as np
from config import (
    DQN_LR, DQN_GAMMA, DQN_BUF, DQN_BATCH, DQN_TGT_FREQ,
    EPS_INIT, N_EPISODES, TOTAL_STEPS,
)
from models import QNet, ReplayBuffer, ImprovedPER
from epsilon_strategy import (
    update_epsilon_basic, update_epsilon_episode, select_action,
    FIXED_PER_ALPHA, FIXED_TGT_FREQ, FIXED_GRAD_CLIP,
)

_ROLLING_WINDOW = 30


class DQNAgent:
    """
    三种模式：
      'basic'    : 标准 QNet + 指数衰减 ε + 均匀经验回放
      'eps_only' : 标准 QNet + 奖励微调 ε + 均匀经验回放
      'eps_per'  : 标准 QNet + 奖励微调 ε + 优先经验回放(PER)
    网络结构三者相同（均为 QNet），便于单独对比探索策略和回放策略的贡献。
    """

    def __init__(self, eps_mode: str = 'basic'):
        assert eps_mode in ('basic', 'eps_only', 'eps_per')
        self.eps_mode = eps_mode

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

        if eps_mode == 'eps_per':
            self.buffer       = ImprovedPER(DQN_BUF, alpha=FIXED_PER_ALPHA)
            self._beta        = 0.4
            self._beta_anneal = (1.0 - 0.4) / max(TOTAL_STEPS, 1)
            self.tgt_freq     = FIXED_TGT_FREQ
            self._grad_clip   = FIXED_GRAD_CLIP
        else:
            self.buffer     = ReplayBuffer(DQN_BUF)
            self.tgt_freq   = DQN_TGT_FREQ
            self._grad_clip = 1.0

    # ── 动作选择 ──────────────────────────────────────────────
    def act(self, state_tensor) -> int:
        return select_action(state_tensor, self.eps, self.q_net)

    # ── ε 更新 ────────────────────────────────────────────────
    def update_eps(self, step: int, rew_curr: float = 0.0):
        self._episode += 1
        self._rew_window.append(rew_curr)
        rolling_avg = float(np.mean(self._rew_window)) if self._rew_window else None

        if self.eps_mode == 'basic':
            self.eps = update_epsilon_basic(step)
        else:
            # eps_only 和 eps_per 均使用奖励微调 ε
            self.eps = update_epsilon_episode(
                self._episode, self._n_episodes,
                rew_curr, rolling_avg,
            )

    # ── 存储经验 ──────────────────────────────────────────────
    def store(self, s, a, r, s_, done):
        self.buffer.push(s, a, r, s_, done)

    # ── 学习 ──────────────────────────────────────────────────
    def learn(self, step: int):
        if len(self.buffer) < self.batch_sz:
            return None

        if self.eps_mode == 'eps_per':
            beta = min(1.0, self._beta + step * self._beta_anneal)
            states, actions, rewards, next_states, dones, indices, weights = \
                self.buffer.sample(self.batch_sz, beta=beta)
            with torch.no_grad():
                q_next = self.tgt_net(next_states).max(1)[0]
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
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self._grad_clip)
        self.optimizer.step()

        if step % self.tgt_freq == 0:
            self.tgt_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ── 持久化 ────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({'q_net': self.q_net.state_dict(),
                    'eps_mode': self.eps_mode}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.q_net.load_state_dict(ckpt['q_net'])
        self.tgt_net.load_state_dict(ckpt['q_net'])
