# DreamerV3 for Drone Locomotion

DreamerV3 applied to quadrotor control tasks. The agent learns a world model from experience and trains an actor-critic policy on imagined rollouts — no task-specific tuning required.

Based on [Mastering Diverse Domains through World Models (Hafner et al., Nature 2025)][paper].

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

---

## Environments

### MuJoCo

| Config | Task | Description |
|--------|------|-------------|
| `mujocohover` | Hover | Stabilize at a fixed target position |
| `mujocotrack` | Track | Follow a lemniscate trajectory |
| `mujocoforest` | Forest | Navigate through a forest of obstacles (based on [OmniDrones][omnidrones] Forest) |

| Hover | Track | Forest |
|:---:|:---:|:---:|
| ![hover](assets/snapshot_hover.png) | ![track](assets/snapshot_track.png) | ![forest](assets/snapshot_forest.png) |

---

## Sample Efficiency (Track Task)

DreamerV3 trains a world model and optimizes the policy entirely inside imagined rollouts, requiring far fewer real environment interactions. PPO collects on-policy experience directly from the environment, which is less sample-efficient.

| | DreamerV3 | PPO |
|---|:---:|:---:|
| **Env steps** | 1M | 10M |

DreamerV3 achieves competitive scores using **10× fewer environment steps** than PPO.

| DreamerV3 | PPO |
|:---:|:---:|
| ![dreamerv3 track score](assets/score_dreamerv3_track.png) | ![ppo track score](assets/score_ppo_track.png) |

---

## Installation

Requires Python 3.11+ and CUDA 12.

```sh
conda env create -f environment.yml
conda activate dreamerv3-drone
```

---

## Training

### DreamerV3

```sh
python dreamerv3/main.py --configs mujocohover
python dreamerv3/main.py --configs mujocotrack
python dreamerv3/main.py --configs mujocoforest
```

To resume a stopped run, reuse the same `--logdir`.

### PPO

```sh
python train_ppo.py --task hover
python train_ppo.py --task track
python train_ppo.py --task forest
```

---

## Visualization

Play a trained policy with the MuJoCo viewer:

### DreamerV3

```sh
python play_mujoco.py ~/logdir/dreamerv3/hover --task hover
python play_mujoco.py ~/logdir/dreamerv3/track --task track
python play_mujoco.py ~/logdir/dreamerv3/forest --task forest
```

### PPO

```sh
python play_mujoco.py ~/logdir/ppo/hover --task hover --algo ppo
python play_mujoco.py ~/logdir/ppo/track --task track --algo ppo
python play_mujoco.py ~/logdir/ppo/forest --task forest --algo ppo
```

---

## Plotting

Generate training curves from a log directory:

```sh
python plot_metrics.py ~/logdir/dreamerv3/mujoco-forest
python plot_metrics.py ~/logdir/dreamerv3/mujoco-forest --out results.png --smooth 50
```

Metrics are also written as JSONL files (`metrics.jsonl`) and can be viewed with [Scope][scope]:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

---

## Tips

- All hyperparameters are in `dreamerv3/configs.yaml` and can be overridden from the command line.
- Use `--configs debug` for a fast sanity check (small model, frequent logs, no good learning).
- Combine config blocks: `--configs mujocoforest size50m` to use a 50M parameter model.
- Switch compute backend: `--jax.platform cpu` or `--jax.platform tpu`.
- CUDA error? Scroll up — the root cause is usually an earlier error (OOM, version mismatch). Try `--batch_size 1` to rule out OOM.
- `Too many leaves for PyTreeDef` means the checkpoint is incompatible with the current config — check you're not reusing a mismatched logdir.

---

## Acknowledgements

This repository was developed with [Claude Code](https://claude.ai/code).

[paper]: https://arxiv.org/pdf/2301.04104
[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[scope]: https://github.com/danijar/scope
[omnidrones]: https://github.com/btx0424/OmniDrones
