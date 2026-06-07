# config.py — 全局超参数与实验配置
# 所有模块统一从此处导入，修改只需改动本文件

import random
import numpy as np
import torch

# ── 随机种子 ─────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ── 环境参数（论文1 Table I）─────────────────────────────────
GRID_SIZE       = 7
CELL_DIST       = 400.0       # 单元格边长 (m)
UAV_SPEED       = 10.0        # 无人机空速 (m/s)
WIND_SIGMA      = 0.3         # 高斯转移标准差 (rad)
GOAL_CELL       = (6, 0)
START_CELL      = (0, 6)
GOAL_REWARD     =  10
OBSTACLE_PENALTY =  10

# ── 障碍物随机生成参数 ────────────────────────────────────────
# density: 障碍物占总格数的比例（9×9=81格）
#   推荐范围 [0.03, 0.20]，超过 0.25 可能封堵路径
#   示例：0.06 ≈ 4格 | 0.09 ≈ 7格 | 0.12 ≈ 9格
# OBSTACLE_SEED: 仅控制障碍物布局，与训练随机种子独立
#   相同 seed+density → 完全相同的障碍物位置（可复现）
#   修改 seed 得到不同布局，密度不变
OBSTACLE_DENSITY = 0.15       # 默认 ~5–6 个障碍物（与原硬编码数量相当）
OBSTACLE_SEED    = 0           # 障碍物布局随机种子


def generate_random_obstacles(
    grid_size: int   = GRID_SIZE,
    density:   float = OBSTACLE_DENSITY,
    seed:      int   = OBSTACLE_SEED,
    goal:      tuple = GOAL_CELL,
    start:     tuple = START_CELL,
) -> list:
    """
    按密度在网格中随机放置固定障碍物。

    Parameters
    ----------
    grid_size : 网格边长（正方形）
    density   : 障碍物占 grid_size² 总格数的比例，范围 (0, 1)
    seed      : 随机种子；相同 seed+density 保证完全可复现
    goal      : 目标格坐标，强制排除在障碍物之外
    start     : 起点格坐标，强制排除在障碍物之外

    Returns
    -------
    list of (col, row) tuples，按 (col, row) 升序排列
    """
    rng = np.random.RandomState(seed)
    excluded = {goal, start}
    candidates = [
        (c, r)
        for c in range(grid_size)
        for r in range(grid_size)
        if (c, r) not in excluded
    ]
    n = max(0, int(round(density * grid_size * grid_size)))
    n = min(n, len(candidates))
    idx = rng.choice(len(candidates), n, replace=False)
    return sorted([candidates[i] for i in idx])


# 动态生成障碍物坐标列表（替换原来的硬编码版本）
# 修改 OBSTACLE_DENSITY 或 OBSTACLE_SEED 即可改变布局，无需手动编辑坐标
OBSTACLE_CELLS = generate_random_obstacles()

# ── 值迭代参数（论文1 Section II.B）─────────────────────────
VI_GAMMA    = 0.95
VI_THETA    = 1e-4
VI_MAX_ITER = 5000

# ── 神经网络结构 ──────────────────────────────────────────────
STATE_DIM   = 2
N_ACTIONS   = 8
QNET_HIDDEN1 = 28
QNET_HIDDEN2 = 49

# ── 训练参数 ─────────────────────────────────────────────────
N_EPISODES       = 3000        # 300 → 1000，给改进算法足够时间体现优势
MAX_STEPS        = 150         # 50  → 100，减少因步数上限截断导致的稀疏奖励
TOTAL_STEPS      = N_EPISODES * MAX_STEPS * 0.6      # 修复：与实际训练步数（~15-35k）对齐，确保ε正确衰减
EXTRACT_MAX_STEPS = 50

# ── ε-贪婪（基础版 / 改进版共用）────────────────────────────
EPS_INIT  = 1.0                # 0.9 → 1.0，改进版起点更高，衰减空间更大
EPS_MIN   = 0.05
BETA      = 0.15       # 0.05 → 0.15，扩大奖励因子调节范围               # 0.10 → 0.05，让奖励因子有更大调节范围

# ── DQN 基线超参数 ────────────────────────────────────────────
DQN_LR        = 5e-4
DQN_GAMMA     = 0.95
DQN_BUF       = 10_000        # 6000 → 10000，更大缓冲区减少样本相关性
DQN_BATCH     = 64
DQN_TGT_FREQ  = 20            # 40   → 20，目标网络更频繁同步，基线收敛更快


# ── 输出路径 ─────────────────────────────────────────────────
OUTPUT_DIR = "./outputs"