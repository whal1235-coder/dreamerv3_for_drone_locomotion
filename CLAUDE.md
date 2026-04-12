# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DreamerV3 adapted for quadrotor drone control in MuJoCo. Compares DreamerV3 (world-model RL) against a GPU-parallel PPO baseline. Based on Hafner et al., Nature 2025.

## Architecture

```
.
├── dreamerv3/                 # DreamerV3 agent
│   ├── main.py                #   entry point, config parsing
│   ├── agent.py               #   world model + actor-critic
│   ├── rssm.py                #   Recurrent State Space Model
│   └── configs.yaml           #   all hyperparameters
├── embodied/                  # Embodied RL framework
│   ├── core/                  #   driver, replay buffer, wrappers
│   ├── envs/
│   │   ├── mujoco_drone.py    #   Hover / Track / Forest environments
│   │   └── assets/            #   MuJoCo XML models
│   ├── jax/                   #   JAX agent, optimizer, transforms
│   └── run/
│       └── train.py           #   training loop
├── train_ppo.py               # PPO baseline (GPU-parallel via MJX)
├── play_mujoco.py             # visualization / playback
├── plot_metrics.py            # training curve plotter
├── environment.yml            # conda environment spec
└── commands.txt               # command reference
```

## Tech Stack

- **Language**: Python 3.11.15
- **Deep Learning**: JAX 0.9.2, Flax 0.12.6, Optax 0.2.5, NinJAX 3.6.2
- **Physics Simulation**: MuJoCo 3.6.0, MuJoCo MJX (GPU-parallel)
- **RL Frameworks**: DreamerV3 (world-model), PPO (policy gradient)
- **Numerical**: NumPy 2.4.4, SciPy 1.17.1
- **Visualization**: MuJoCo Viewer, Matplotlib 3.10.8
- **Config/Logging**: elements 3.22.0, ruamel.yaml 0.19.1, Scope
- **Environment**: Conda, CUDA 12

## Commands

- `commands.txt` — read this before running any training, visualization, or plotting scripts.

## Troubleshooting

- CUDA errors: scroll up for root cause (often OOM). Try `--batch_size 1`.
- `Too many leaves for PyTreeDef`: checkpoint/config mismatch — don't reuse logdir across different configs.
- To resume a DreamerV3 run, reuse the same `--logdir`.
