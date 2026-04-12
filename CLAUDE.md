# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DreamerV3 adapted for quadrotor drone control in MuJoCo. Compares DreamerV3 (world-model RL) against a GPU-parallel PPO baseline. Based on Hafner et al., Nature 2025.

Three tasks with increasing difficulty:
- **Hover** — stabilize at [0,0,2], 30-dim obs, 500 steps
- **Track** — follow lemniscate trajectory, 36-dim obs, 600 steps
- **Forest** — navigate 48m through randomized obstacles with 144-ray LiDAR, 167-dim obs, 800 steps

## Setup

Requires Python 3.11+, CUDA 12, conda.

```sh
conda env create -f environment.yml
conda activate dreamerv3-drone
```

## Commands

### DreamerV3 Training
```sh
python dreamerv3/main.py --configs mujocohover --jax.platform cuda
python dreamerv3/main.py --configs mujocotrack --jax.platform cuda
python dreamerv3/main.py --configs mujocoforest --jax.platform cuda
```

Config blocks can be combined: `--configs mujocoforest size50m` for a 50M param model. Use `--configs debug` for fast sanity checks.

### PPO Baseline Training (GPU-parallel, ~4096 envs via MJX)
```sh
python train_ppo.py --task hover --n_envs 4096
python train_ppo.py --task forest --n_envs 4096 --steps 100000000
```

### Visualization
```sh
python play_mujoco.py ~/logdir/dreamerv3/forest --task forest
python play_mujoco.py ~/logdir/ppo/forest --task forest --algo ppo
```

### Plotting
```sh
python plot_metrics.py ~/logdir/dreamerv3/forest
```

## Architecture

### Two-algorithm structure

1. **DreamerV3** (`dreamerv3/` + `embodied/`): World-model agent that learns from replay via imagined rollouts. Serial environment interaction through `embodied/core/driver.py`.
2. **PPO** (`train_ppo.py`): Self-contained GPU-parallel baseline using `mujoco.mjx` + `jax.vmap` for massively parallel simulation. Shares the same network backbone (3×1024 MLP + RMSNorm + SiLU) and value representation (symexp twohot) as DreamerV3.

### DreamerV3 internals (`dreamerv3/`)

- `main.py` — entry point; config parsing, environment/agent/logger/replay construction
- `agent.py` — world model + actor-critic agent definition
- `rssm.py` — Recurrent State Space Model (latent dynamics)
- `configs.yaml` — all hyperparameters; task-specific blocks override `defaults`

### Embodied framework (`embodied/`)

- `embodied/jax/` — JAX-based agent, neural net heads, optimizer, transforms (modified for newer JAX API: keyword args for `in_shardings`/`out_shardings`/`donate_argnums`)
- `embodied/run/train.py` — training loop (modified: episode termination logging, auto-plot on save)
- `embodied/core/` — driver, replay buffer, environment wrappers
- `embodied/envs/mujoco_drone.py` — all three drone environments; factory creates task variant based on config

### Drone environment (`embodied/envs/mujoco_drone.py`)

Uses Hummingbird quadrotor model (KF=8.54858e-6, KM=1.36777e-7, max RPM=838 rad/s). Actions are 4 normalized rotor throttles. MuJoCo XML models live in `embodied/envs/assets/`.

### Config flow

CLI args → `dreamerv3/configs.yaml` defaults → task-specific overrides (e.g., `mujocoforest`) → size overrides (e.g., `size50m`) → command-line flags. Logdir defaults to `~/logdir/dreamerv3/{task_name}`.

## Communication

The user is practicing English. When they make grammar or wording mistakes, gently correct them. When they use non-English words (e.g., Korean), state the English equivalent in this exact format: "The English equivalent of '한국어' is 'English word'." For example, if they say "Help me to 고치는거 this project", respond with: "The English equivalent of '고치는거' is 'fix/debug'." Then continue with the task.

## Troubleshooting

- CUDA errors: scroll up for root cause (often OOM). Try `--batch_size 1`.
- `Too many leaves for PyTreeDef`: checkpoint/config mismatch — don't reuse logdir across different configs.
- To resume a DreamerV3 run, reuse the same `--logdir`.
