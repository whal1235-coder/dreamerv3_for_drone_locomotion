"""
Play a trained policy (DreamerV3 or PPO) on MuJoCo drone environments.

Usage:
  conda activate test2
  cd /home/psm/workspaces/rl/dreamerv3
  # DreamerV3
  python play_mujoco.py ~/logdir/dreamerv3/mujoco-forest-v1 --task forest
  # PPO
  python play_mujoco.py ~/logdir/ppo/mujoco-forest-v1 --task forest --algo ppo
"""

import argparse
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # play는 CPU inference로 충분

folder = pathlib.Path(__file__).parent / 'dreamerv3'
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))

import mujoco
import numpy as np
import elements
import embodied
import ruamel.yaml as yaml


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _draw_track_viz(viewer, env):
    """Draw lemniscate trajectory (orange) + current target (red) in viewer."""
    traj_pts, cur_target = env.traj_viz_data(n_pts=100)
    scn = viewer.user_scn
    scn.ngeom = 0
    eye = np.eye(3, dtype=np.float32).flatten()
    for pt in traj_pts:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.04, 0.04, 0.04]),
            pt.astype(np.float64),
            eye.astype(np.float64),
            np.array([1.0, 0.5, 0.0, 0.7], np.float32),
        )
        scn.ngeom += 1
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.12, 0.12, 0.12]),
            cur_target.astype(np.float64),
            eye.astype(np.float64),
            np.array([1.0, 0.0, 0.0, 1.0], np.float32),
        )
        scn.ngeom += 1


def _draw_grid(scn, x0=-16, x1=16, y0=-24, y1=24, step=1, z=0.01):
    """Draw a fixed 1m ground grid over the navigation area."""
    rgba = np.array([0.6, 0.6, 0.6, 0.3], np.float32)
    for y in range(y0, y1 + 1, step):
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 1.0,
            np.array([x0, y, z]), np.array([x1, y, z]),
        )
        scn.geoms[scn.ngeom].rgba[:] = rgba
        scn.ngeom += 1
    for x in range(x0, x1 + 1, step):
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 1.0,
            np.array([x, y0, z]), np.array([x, y1, z]),
        )
        scn.geoms[scn.ngeom].rgba[:] = rgba
        scn.ngeom += 1


def _draw_forest_viz(viewer, env):
    """Draw 1m grid, target (green), trees (orange), lidar rays, and direction arrow."""
    tree_xys, target_pos, tree_radii, tree_z, tree_half_h = env.traj_viz_data()
    scn = viewer.user_scn
    scn.ngeom = 0
    eye = np.eye(3, dtype=np.float32).flatten()

    # 1m ground grid (fixed over navigation area)
    _draw_grid(scn)

    # Target sphere
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.3, 0.3, 0.3]),
            target_pos.astype(np.float64),
            eye.astype(np.float64),
            np.array([0.0, 1.0, 0.0, 0.7], np.float32),
        )
        scn.ngeom += 1

    # Tree cylinders (skip invisible trees: r == 0)
    for i in range(len(tree_xys)):
        r = float(tree_radii[i])
        if r == 0.0:
            continue
        if scn.ngeom >= scn.maxgeom:
            break
        z_c = float(tree_z[i])
        h   = float(tree_half_h[i])
        tree_pos = np.array([tree_xys[i, 0], tree_xys[i, 1], z_c], dtype=np.float64)
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([r, h, 0.0]),
            tree_pos,
            eye.astype(np.float64),
            np.array([0.7, 0.4, 0.1, 0.6], np.float32),
        )
        scn.ngeom += 1

    # Lidar rays: only draw rays that actually hit an obstacle
    drone_pos = env.data.qpos[:3].astype(np.float64)
    R = env.data.xmat[env._drone_body_id].reshape(3, 3)  # body → world
    scan = env._last_lidar_scan
    for ray_dir_body, scan_val in zip(env._lidar_dirs, scan):
        if scan_val < 0.05:  # no obstacle detected
            continue
        if scn.ngeom >= scn.maxgeom:
            break
        ray_dir_world = R @ ray_dir_body
        actual_dist = float(env.LIDAR_RANGE - scan_val)
        end_pt = drone_pos + ray_dir_world * actual_dist
        t = actual_dist / env.LIDAR_RANGE  # 1=far, 0=touching
        rgba = np.array([1.0 - t, t * 0.7, 0.0, 0.5 + 0.5 * (1.0 - t)], np.float32)
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            1.5,
            drone_pos, end_pt,
        )
        scn.geoms[scn.ngeom].rgba[:] = rgba
        scn.ngeom += 1

    # Direction arrow: drone center → target (blue)
    if scn.ngeom < scn.maxgeom:
        direction = target_pos.astype(np.float64) - drone_pos
        dist = float(np.linalg.norm(direction))
        if dist > 0.01:
            arrow_len = min(dist, 2.5)
            end_pt = drone_pos + direction / dist * arrow_len
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.006,
                drone_pos, end_pt,
            )
            scn.geoms[scn.ngeom].rgba[:] = np.array([1.0, 0.0, 0.0, 1.0], np.float32)
            scn.ngeom += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_best_ckpt_from_log(logdir: pathlib.Path, algo: str = 'ppo') -> pathlib.Path:
    """Find the numbered checkpoint right before the best score.

    The best-score checkpoint itself may be one where the model is already
    breaking, so we select the checkpoint immediately preceding it for
    more stable behaviour.

    Works for both PPO (.pkl files) and DreamerV3 (directories).
    """
    import json as _json

    ckpt_dir = logdir / 'ckpt'

    # All numbered checkpoints, sorted by step
    if algo == 'ppo':
        ckpt_files = sorted(
            (p for p in ckpt_dir.glob('[0-9]*.pkl')),
            key=lambda p: int(p.stem),
        )
    else:
        ckpt_files = sorted(
            (p for p in ckpt_dir.iterdir()
             if p.is_dir() and p.name not in ('latest', 'best') and p.name.isdigit()),
            key=lambda p: int(p.name),
        )

    # Read metrics.jsonl to find the step at which the best score occurred
    metrics_path = logdir / 'metrics.jsonl'
    best_step, best_score = None, -float('inf')
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                    score = entry.get('episode/score', -float('inf'))
                    if score > best_score:
                        best_score = score
                        best_step = entry.get('step')
                except Exception:
                    continue

    def _ckpt_step(p):
        return int(p.stem) if p.suffix == '.pkl' else int(p.name)

    if best_step is not None and ckpt_files:
        print(f'Best episode/score={best_score:.2f} at step {best_step:,}')
        # Find the checkpoint right before the best-score step
        prev = None
        for p in ckpt_files:
            if _ckpt_step(p) >= best_step:
                break
            prev = p
        if prev is not None:
            print(f'Selecting checkpoint before best: {prev.name}')
            return prev
        # best_step is at or before the first checkpoint — use the first one
        print(f'No checkpoint before best step, using first: {ckpt_files[0].name}')
        return ckpt_files[0]

    # Fallback: latest numbered checkpoint
    if ckpt_files:
        print(f'[warn] Could not determine best step, using latest: {ckpt_files[-1].name}')
        return ckpt_files[-1]

    if algo == 'ppo':
        print('[warn] No checkpoints found, trying latest.pkl')
        return ckpt_dir / 'latest.pkl'
    else:
        print('[warn] No checkpoints found, trying latest')
        return ckpt_dir / 'latest'


def _ckpt_step_str(path: pathlib.Path) -> str:
    """Return 'step=N' if the path (or its symlink target) is a numbered checkpoint, else the filename."""
    resolved = path.resolve() if path.is_symlink() else path
    try:
        return f'step={int(resolved.stem):,}'
    except ValueError:
        return resolved.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    parser.add_argument('--task', type=str, default='track',
                        choices=['hover', 'track', 'forest'])
    parser.add_argument('--algo', type=str, default='dreamer',
                        choices=['dreamer', 'ppo'])
    parser.add_argument('--episodes', type=int, default=0,
                        help='Number of episodes to run (0 = infinite)')
    parser.add_argument('--speed', type=float, default=2.0,
                        help='Playback speed multiplier (e.g. 2.0 = 2x)')
    parser.add_argument('--no-best', action='store_true',
                        help='Load latest checkpoint instead of best')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to a specific checkpoint (overrides --no-best)')
    parser.add_argument('--no-gui', action='store_true',
                        help='Disable MuJoCo viewer')
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)

    # Build env (keep base reference for viewer access)
    from embodied.envs.mujoco_drone import MujocoDrone
    base_env = MujocoDrone(task=args.task)

    # ---------------------------------------------------------------------------
    # Load policy (algo-specific)
    # ---------------------------------------------------------------------------
    if args.algo == 'dreamer':
        logdir_el = elements.Path(str(logdir))
        config = elements.Config(yaml.YAML(typ='safe').load(
            (logdir_el / 'config.yaml').read()))

        env = base_env
        for name, space in env.act_space.items():
            if not space.discrete:
                env = embodied.wrappers.NormalizeAction(env, name)
        env = embodied.wrappers.UnifyDtypes(env)
        env = embodied.wrappers.CheckSpaces(env)
        for name, space in env.act_space.items():
            if not space.discrete:
                env = embodied.wrappers.ClipAction(env, name)

        obs_space = {k: v for k, v in env.obs_space.items()
                     if not k.startswith('log/')}
        act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}

        from dreamerv3.agent import Agent
        agent_config = elements.Config(
            **config.agent,
            logdir=str(logdir),
            seed=config.seed,
            jax=config.jax,
            batch_size=1,
            batch_length=2,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=0,
            replicas=1,
        )
        agent = Agent(obs_space, act_space, agent_config)

        if args.ckpt:
            ckpt_path = pathlib.Path(args.ckpt)
        elif not args.no_best:
            ckpt_path = _find_best_ckpt_from_log(logdir, algo='dreamer')
        else:
            ckpt_path = logdir / 'ckpt' / 'latest'
        print(f'Loading DreamerV3 checkpoint: {ckpt_path}')
        cp = elements.Checkpoint()
        cp.agent = agent
        cp.load(str(ckpt_path), keys=['agent'])

        def get_action(obs, state):
            obs_batch = {k: v[None] for k, v in obs.items() if not k.startswith('log/')}
            state, act_batch, _ = agent.policy(state, obs_batch, mode='eval')
            action = {k: v[0] for k, v in act_batch.items()}
            action['reset'] = False
            return action, state

        def init_state():
            return agent.init_policy(1)

        def make_step(obs, state):
            return get_action(obs, state)

    else:  # ppo
        import jax.numpy as jnp
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from train_ppo import ActorCritic, twohot_decode, _load_ppo_config

        if args.ckpt:
            ckpt_path = pathlib.Path(args.ckpt)
        elif not args.no_best:
            ckpt_path = _find_best_ckpt_from_log(logdir, algo='ppo')
        else:
            ckpt_path = logdir / 'ckpt' / 'latest.pkl'
        print(f'Loading PPO checkpoint: {ckpt_path.name} ({_ckpt_step_str(ckpt_path)})')
        with open(ckpt_path, 'rb') as f:
            params = pickle.load(f)

        enc_cfg, pol_cfg, val_cfg, *_ = _load_ppo_config()
        net = ActorCritic(
            enc_layers=enc_cfg['layers'], enc_units=enc_cfg['units'],
            pol_layers=pol_cfg['layers'], pol_units=pol_cfg['units'],
            val_bins=val_cfg['bins'],
        )

        env = base_env

        def init_state():
            return None

        def make_step(obs, state):
            obs_j = jnp.array(obs['state'], dtype=jnp.float32)[None]  # (1, 167)
            mean, _std, _ = net.apply(params, obs_j)
            action_arr = np.array(mean[0])
            return {'action': action_arr, 'reset': False}, state

    # ---------------------------------------------------------------------------
    # Run episodes
    # ---------------------------------------------------------------------------
    def run_episodes(viewer=None):
        ep = 0
        while args.episodes == 0 or ep < args.episodes:
            obs = env.step({'reset': True, 'action': np.zeros(4, np.float32)})
            state = init_state()
            ep_reward = 0.0
            step = 0
            while True:
                action, state = make_step(obs, state)
                obs = env.step(action)
                ep_reward += float(obs['reward'])
                step += 1

                if viewer is not None:
                    if args.task == 'track':
                        _draw_track_viz(viewer, base_env)
                    elif args.task == 'forest':
                        _draw_forest_viz(viewer, base_env)
                    viewer.sync()
                    time.sleep(0.016 / args.speed)

                if obs['is_last']:
                    break

            ep += 1
            print(f'Episode {ep}: steps={step}, reward={ep_reward:.2f}')

    if args.no_gui:
        run_episodes()
    else:
        import mujoco.viewer as mjviewer
        drone_id = mujoco.mj_name2id(base_env.model, mujoco.mjtObj.mjOBJ_BODY, 'drone')
        with mjviewer.launch_passive(base_env.model, base_env.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = drone_id
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -30.0
            run_episodes(viewer)

    env.close()


if __name__ == '__main__':
    main()
