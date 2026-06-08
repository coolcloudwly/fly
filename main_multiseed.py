# main_multiseed.py — 多种子实验主程序（三算法对比版）
#
# 用法示例：
#   python main_multiseed.py --wind circular --reward energy
#   python main_multiseed.py --wind uniform  --reward time --seeds 0 1 2 3 4
#
# 对比算法：
#   VI / DQN-Basic / DQN-ε only / DQN-ε+PER

import os
import argparse
import random

from config import N_EPISODES, OUTPUT_DIR, OBSTACLE_DENSITY, OBSTACLE_SEED
from experiment import run_one_seed, save_seed_data, aggregate, save_csv
from plotting import (
    plot_seed_paths, plot_metrics_mean, plot_learning_curves,
    plot_efficiency_metrics, plot_stability_metrics,
    plot_q_overestimation, plot_sample_efficiency,
)
from showdata import print_summary

random.seed(2026)
_default_seeds = random.sample(range(1, 3001), 5)

parser = argparse.ArgumentParser(description="Multi-seed UAV ablation experiment")
parser.add_argument("--seeds",    type=int, nargs="+", default=_default_seeds)
parser.add_argument("--wind",     default="circular", choices=["uniform", "circular"])
parser.add_argument("--reward",   default="energy",   choices=["energy", "time"])
parser.add_argument("--ep",       type=int, default=N_EPISODES)
parser.add_argument("--density",  type=float, default=None)
parser.add_argument("--obs_seed", type=int,   default=None)
args = parser.parse_args()

SEEDS    = args.seeds
WIND     = args.wind
REWARD   = args.reward
N_EP     = args.ep
DENSITY  = args.density
OBS_SEED = args.obs_seed

_density_val  = DENSITY  if DENSITY  is not None else OBSTACLE_DENSITY
_obs_seed_val = OBS_SEED if OBS_SEED is not None else OBSTACLE_SEED

OUT_ROOT = os.path.join(OUTPUT_DIR, f"ablation_{WIND}_{REWARD}")
os.makedirs(OUT_ROOT, exist_ok=True)

print("=" * 65)
print(f"  Ablation Experiment: Basic vs ε-only vs ε+PER")
print(f"  Wind={WIND}  Reward={REWARD}  Episodes={N_EP}")
print(f"  Seeds={SEEDS}")
print(f"  Obstacles: density={_density_val:.3f}  obs_seed={_obs_seed_val}")
print("=" * 65)

all_results = []
path_images = []

for seed in SEEDS:
    print(f"\n{'─'*55}")
    print(f"  Running seed = {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
    print(f"{'─'*55}")
    result = run_one_seed(seed, WIND, REWARD, N_EP, DENSITY, OBS_SEED)
    save_seed_data(seed, result, OUT_ROOT)
    img = plot_seed_paths(seed, result, OUT_ROOT, WIND, REWARD)
    path_images.append(img)
    all_results.append(result)

stats = aggregate(all_results)

csv_path       = os.path.join(OUT_ROOT, "summary_table.csv")
metrics_png    = os.path.join(OUT_ROOT, "metrics_mean.png")
curves_png     = os.path.join(OUT_ROOT, "learning_curves.png")
efficiency_png = os.path.join(OUT_ROOT, "efficiency_metrics.png")
stability_png  = os.path.join(OUT_ROOT, "stability_metrics.png")
q_over_png     = os.path.join(OUT_ROOT, "q_overestimation.png")
sample_eff_png = os.path.join(OUT_ROOT, "sample_efficiency.png")

save_csv(stats, csv_path)
plot_metrics_mean(stats, metrics_png, SEEDS, WIND, REWARD)
plot_learning_curves(all_results, curves_png, SEEDS, WIND, REWARD)
plot_efficiency_metrics(stats, all_results, efficiency_png, SEEDS, WIND, REWARD)
plot_stability_metrics(stats, all_results, stability_png, SEEDS, WIND, REWARD)
plot_q_overestimation(stats, all_results, q_over_png, SEEDS, WIND, REWARD)
plot_sample_efficiency(all_results, sample_eff_png, SEEDS, WIND, REWARD)

print_summary(stats, SEEDS, WIND, REWARD)

print(f"\n✅ 全部完成，结果保存在: {OUT_ROOT}/")
print("\n  生成文件列表：")
for f in path_images + [metrics_png, curves_png, efficiency_png,
                         stability_png, q_over_png, sample_eff_png, csv_path]:
    print(f"    {f}")
