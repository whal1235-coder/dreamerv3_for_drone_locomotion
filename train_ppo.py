"""
MJX GPU-parallel PPO for MuJoCo drone tasks (hover / track / forest).
Runs ~4096 envs in parallel on GPU via mujoco.mjx + JAX.
Saves checkpoints and plots to logdir.

Usage:
  conda activate dreamerv3-drone
  python train_ppo.py [--task hover|track|forest] [--logdir ~/logdir/ppo/forest] [--n_envs 4096] [--steps 100000000]
"""

import argparse, json, math, os, pathlib, pickle, sys, time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from typing import NamedTuple
import numpy as np

import jax
jax.config.update('jax_compilation_cache_dir', os.path.expanduser('~/.jax_cache'))
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
import jax.numpy as jnp
import flax.linen as nn
import optax
import mujoco
import mujoco.mjx as mjx

sys.path.insert(0, '.')

# ---------------------------------------------------------------------------
# Shared constants / model
# ---------------------------------------------------------------------------
_XML = os.path.join(os.path.dirname(__file__),
                    'embodied', 'envs', 'assets', 'quadrotor.xml')

_KF         = 8.54858e-6
_KM         = 1.36777e-7
_MAX_RPM    = 838.0
_DIRECTIONS = jnp.array([-1., 1., -1., 1.])
_ROTOR_POS  = jnp.array([[ 0.12,  0.12, 0.],
                           [-0.12,  0.12, 0.],
                           [-0.12, -0.12, 0.],
                           [ 0.12, -0.12, 0.]])
_ACT_DIM    = 4

_mj_model   = mujoco.MjModel.from_xml_path(_XML)
_mx_model   = mjx.put_model(_mj_model)
_DRONE_ID   = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_BODY, 'drone')

# ---------------------------------------------------------------------------
# Shared JAX physics helpers
# ---------------------------------------------------------------------------

def _quat_to_rot(q):
    """[qw,qx,qy,qz] → 3×3 rotation matrix."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ])


def _apply_forces(mx_data, action):
    throttle  = (action + 1.) / 2.
    rpm       = throttle * _MAX_RPM
    thrust    = _KF * rpm ** 2
    torque    = _KM * rpm ** 2

    R         = mx_data.xmat[_DRONE_ID]
    world_force = R @ jnp.array([0., 0., thrust.sum()])

    total_torque_z = jnp.sum(_DIRECTIONS * torque)
    rp_world  = (_ROTOR_POS @ R.T)
    rf_world  = jnp.outer(thrust, R[:, 2])
    moments   = jnp.cross(rp_world, rf_world).sum(0)
    world_torque = moments + R @ jnp.array([0., 0., total_torque_z])

    xfrc = mx_data.xfrc_applied.at[_DRONE_ID, :3].set(world_force)
    xfrc = xfrc.at[_DRONE_ID, 3:].set(world_torque)
    return mx_data.replace(xfrc_applied=xfrc)


def _make_quat(rpy):
    r, p, y = rpy[0], rpy[1], rpy[2]
    cy, sy = jnp.cos(y/2), jnp.sin(y/2)
    cp, sp = jnp.cos(p/2), jnp.sin(p/2)
    cr, sr = jnp.cos(r/2), jnp.sin(r/2)
    return jnp.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                       cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


# ===========================================================================
# HOVER task
# ===========================================================================

_HOVER_TARGET_POS = jnp.array([0., 0., 2.])
_HOVER_MAX_STEPS  = 500
_HOVER_MIN_Z      = 0.2
_HOVER_MAX_DIST   = 4.0
_HOVER_OBS_DIM    = 30


def _hover_get_obs(mx_data, step_cnt, last_throttle, target_heading):
    pos     = mx_data.qpos[:3]
    qw      = mx_data.qpos[3:7]
    lin_vel = mx_data.qvel[:3]
    ang_vel = mx_data.qvel[3:6]
    R       = _quat_to_rot(qw)
    heading = R[:, 0]
    up      = R[:, 2]
    qxyzw   = jnp.array([qw[1], qw[2], qw[3], qw[0]])
    rpos     = _HOVER_TARGET_POS - pos
    rheading = target_heading - heading
    throttle_obs = last_throttle * 2. - 1.
    time_enc = jnp.full(4, jnp.float32(step_cnt / _HOVER_MAX_STEPS))
    return jnp.concatenate([rpos, qxyzw, lin_vel, ang_vel, heading, up,
                             throttle_obs, rheading, time_enc]).astype(jnp.float32)


def _hover_reward_done(mx_data, last_throttle, target_heading):
    pos     = mx_data.qpos[:3]
    qw      = mx_data.qpos[3:7]
    ang_vel = mx_data.qvel[3:6]
    R       = _quat_to_rot(qw)
    heading = R[:, 0]
    up      = R[:, 2]
    rpos     = _HOVER_TARGET_POS - pos
    rheading = target_heading - heading
    dist    = jnp.linalg.norm(jnp.concatenate([rpos, rheading]))

    r_pose   = 1. / (1. + (1.2 * dist) ** 2)
    r_up     = ((up[2] + 1.) / 2.) ** 2
    r_spin   = 1. / (1. + ang_vel[2] ** 4)
    r_effort = 0.1 * jnp.exp(-last_throttle.sum())
    reward   = r_pose + r_pose * (r_up + r_spin) + r_effort

    terminated = (pos[2] < _HOVER_MIN_Z) | (dist > _HOVER_MAX_DIST)
    return reward.astype(jnp.float32), terminated


def _hover_reset_single(rng):
    rng, k1, k2, k3 = jax.random.split(rng, 4)
    pos  = jax.random.uniform(k1, (3,),
                               minval=jnp.array([-2.5, -2.5, 1.0]),
                               maxval=jnp.array([ 2.5,  2.5, 2.5]))
    rpy  = jax.random.uniform(k2, (3,),
                               minval=jnp.array([-0.2*math.pi, -0.2*math.pi, 0.]),
                               maxval=jnp.array([ 0.2*math.pi,  0.2*math.pi, 2*math.pi]))
    quat = _make_quat(rpy)
    dx   = mjx.make_data(_mx_model)
    dx   = dx.replace(qpos=jnp.concatenate([pos, quat]), qvel=jnp.zeros(6))
    dx   = mjx.forward(_mx_model, dx)
    tyaw = jax.random.uniform(k3, minval=0., maxval=2*math.pi)
    target_heading = jnp.array([jnp.cos(tyaw), jnp.sin(tyaw), 0.])
    return dx, target_heading.astype(jnp.float32)


class EnvStateHover(NamedTuple):
    mx_data:        object
    step_cnt:       jnp.ndarray   # (n,)
    last_throttle:  jnp.ndarray   # (n, 4)
    target_heading: jnp.ndarray   # (n, 3)


def hover_batch_reset(rng, n_envs):
    keys = jax.random.split(rng, n_envs)
    mx_data, target_heading = jax.vmap(_hover_reset_single)(keys)
    return EnvStateHover(
        mx_data        = mx_data,
        step_cnt       = jnp.zeros(n_envs, jnp.int32),
        last_throttle  = jnp.zeros((n_envs, 4), jnp.float32),
        target_heading = target_heading,
    )


def hover_batch_obs(state: EnvStateHover):
    return jax.vmap(_hover_get_obs)(
        state.mx_data, state.step_cnt,
        state.last_throttle, state.target_heading)


def hover_batch_step(state: EnvStateHover, action, rng):
    throttle = (action + 1.) / 2.
    mx_data  = jax.vmap(_apply_forces)(state.mx_data, action)
    mx_data  = jax.vmap(mjx.step, in_axes=(None, 0))(_mx_model, mx_data)

    step_cnt  = state.step_cnt + 1
    reward, terminated = jax.vmap(_hover_reward_done)(
        mx_data, throttle, state.target_heading)
    truncated = step_cnt >= _HOVER_MAX_STEPS
    done      = terminated | truncated

    keys    = jax.random.split(rng, action.shape[0])
    r_dx, r_th = jax.vmap(_hover_reset_single)(keys)

    def _where(reset_v, curr_v):
        mask = done.reshape(-1, *([1] * (curr_v.ndim - 1)))
        return jnp.where(mask, reset_v, curr_v)

    new_mx   = jax.tree.map(_where, r_dx, mx_data)
    new_th   = _where(r_th, state.target_heading)
    new_step = jnp.where(done, 0, step_cnt)
    new_last = jnp.where(done[:, None], jnp.zeros_like(throttle), throttle)

    new_state = EnvStateHover(new_mx, new_step, new_last, new_th)
    obs = jax.vmap(_hover_get_obs)(mx_data, step_cnt, throttle, state.target_heading)
    return new_state, obs, reward, done


# ===========================================================================
# TRACK task
# ===========================================================================

_TRACK_MAX_STEPS  = 600
_TRACK_RESET_THRES = 0.5
_TRACK_OBS_DIM    = 36
_TRACK_FUTURE     = 4
_TRACK_STEP_SIZE  = 5
_TRAJ_T0          = math.pi / 2.
_TRAJ_ORIGIN      = jnp.array([0., 0., 2.])


def _scale_time_jax(t):
    abs_t  = jnp.abs(t)
    safe_t = jnp.where(abs_t < 1e-6, 1.0, abs_t)
    return jnp.where(abs_t < 1e-6, 0., t / (1. + 1. / safe_t))


def _lemniscate_jax(t, c, scale, rot):
    sin_t = jnp.sin(t)
    cos_t = jnp.cos(t)
    denom = sin_t ** 2 + 1.
    raw   = jnp.array([cos_t / denom, sin_t * cos_t / denom, c * sin_t / denom])
    return _TRAJ_ORIGIN + (rot @ raw) * scale


def _track_ref_pos(step_cnt, step_offset, traj_w, traj_c, traj_scale, traj_rot):
    raw_t = traj_w * (step_cnt + step_offset) / 60.0
    t     = _TRAJ_T0 + _scale_time_jax(raw_t)
    return _lemniscate_jax(t, traj_c, traj_scale, traj_rot)


def _track_get_obs(mx_data, step_cnt, last_throttle,
                   traj_c, traj_scale, traj_w, traj_rot):
    pos     = mx_data.qpos[:3]
    qw      = mx_data.qpos[3:7]
    lin_vel = mx_data.qvel[:3]
    ang_vel = mx_data.qvel[3:6]
    R       = _quat_to_rot(qw)
    heading = R[:, 0]
    up      = R[:, 2]
    qxyzw   = jnp.array([qw[1], qw[2], qw[3], qw[0]])

    rpos = jnp.concatenate([
        _track_ref_pos(step_cnt, _TRACK_STEP_SIZE * k,
                       traj_w, traj_c, traj_scale, traj_rot) - pos
        for k in range(_TRACK_FUTURE)
    ])

    throttle_obs = last_throttle * 2. - 1.
    time_enc     = jnp.full(4, jnp.float32(step_cnt / _TRACK_MAX_STEPS))
    return jnp.concatenate([rpos, qxyzw, lin_vel, ang_vel,
                             heading, up, throttle_obs, time_enc]).astype(jnp.float32)


def _track_reward_done(mx_data, last_throttle,
                       traj_c, traj_scale, traj_w, traj_rot, step_cnt):
    pos     = mx_data.qpos[:3]
    qw      = mx_data.qpos[3:7]
    ang_vel = mx_data.qvel[3:6]
    R       = _quat_to_rot(qw)
    up      = R[:, 2]

    ref_pos = _track_ref_pos(step_cnt, 0, traj_w, traj_c, traj_scale, traj_rot)
    dist    = jnp.linalg.norm(ref_pos - pos)

    tiltage = jnp.abs(1. - up[2])
    spin    = ang_vel[2] ** 2

    r_pose   = jnp.exp(-1.6 * dist)
    r_up     = 0.5 / (1. + tiltage ** 2)
    r_spin   = 0.5 / (1. + spin ** 2)
    r_effort = 0.1 * jnp.exp(-last_throttle.sum())
    reward   = r_pose + r_pose * (r_up + r_spin) + r_effort

    terminated = (pos[2] < 0.1) | (dist > _TRACK_RESET_THRES)
    return reward.astype(jnp.float32), terminated


def _track_reset_single(rng):
    rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)

    c     = jax.random.uniform(k1, minval=-0.6, maxval=0.6)
    scale = jax.random.uniform(k2, (3,),
                                minval=jnp.array([1.8, 1.8, 1.0]),
                                maxval=jnp.array([3.2, 3.2, 1.5]))
    w_mag = jax.random.uniform(k3, minval=0.8, maxval=1.1)
    sign  = jax.random.choice(k4, jnp.array([-1., 1.]))
    w     = w_mag * sign
    yaw   = jax.random.uniform(k5, minval=0., maxval=2 * math.pi)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    rot   = jnp.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])

    init_pos = _lemniscate_jax(_TRAJ_T0, c, scale, rot)

    rng, k6 = jax.random.split(rng)
    rpy  = jax.random.uniform(k6, (3,),
                               minval=jnp.array([-0.2*math.pi, -0.2*math.pi, 0.]),
                               maxval=jnp.array([ 0.2*math.pi,  0.2*math.pi, 2*math.pi]))
    quat = _make_quat(rpy)

    dx = mjx.make_data(_mx_model)
    dx = dx.replace(qpos=jnp.concatenate([init_pos, quat]), qvel=jnp.zeros(6))
    dx = mjx.forward(_mx_model, dx)
    return dx, c.astype(jnp.float32), scale.astype(jnp.float32), w.astype(jnp.float32), rot.astype(jnp.float32)


class EnvStateTrack(NamedTuple):
    mx_data:       object
    step_cnt:      jnp.ndarray   # (n,)
    last_throttle: jnp.ndarray   # (n, 4)
    traj_c:        jnp.ndarray   # (n,)
    traj_scale:    jnp.ndarray   # (n, 3)
    traj_w:        jnp.ndarray   # (n,)
    traj_rot:      jnp.ndarray   # (n, 3, 3)


def track_batch_reset(rng, n_envs):
    keys = jax.random.split(rng, n_envs)
    mx_data, c, scale, w, rot = jax.vmap(_track_reset_single)(keys)
    return EnvStateTrack(
        mx_data       = mx_data,
        step_cnt      = jnp.zeros(n_envs, jnp.int32),
        last_throttle = jnp.zeros((n_envs, 4), jnp.float32),
        traj_c        = c,
        traj_scale    = scale,
        traj_w        = w,
        traj_rot      = rot,
    )


def track_batch_obs(state: EnvStateTrack):
    return jax.vmap(_track_get_obs)(
        state.mx_data, state.step_cnt, state.last_throttle,
        state.traj_c, state.traj_scale, state.traj_w, state.traj_rot)


def track_batch_step(state: EnvStateTrack, action, rng):
    throttle = (action + 1.) / 2.
    mx_data  = jax.vmap(_apply_forces)(state.mx_data, action)
    mx_data  = jax.vmap(mjx.step, in_axes=(None, 0))(_mx_model, mx_data)

    step_cnt = state.step_cnt + 1
    reward, terminated = jax.vmap(_track_reward_done)(
        mx_data, throttle,
        state.traj_c, state.traj_scale, state.traj_w, state.traj_rot, step_cnt)
    truncated = step_cnt >= _TRACK_MAX_STEPS
    done      = terminated | truncated

    keys = jax.random.split(rng, action.shape[0])
    r_dx, r_c, r_scale, r_w, r_rot = jax.vmap(_track_reset_single)(keys)

    def _where(reset_v, curr_v):
        mask = done.reshape(-1, *([1] * (curr_v.ndim - 1)))
        return jnp.where(mask, reset_v, curr_v)

    new_mx    = jax.tree.map(_where, r_dx, mx_data)
    new_c     = _where(r_c, state.traj_c)
    new_scale = _where(r_scale, state.traj_scale)
    new_w     = _where(r_w, state.traj_w)
    new_rot   = _where(r_rot, state.traj_rot)
    new_step  = jnp.where(done, 0, step_cnt)
    new_last  = jnp.where(done[:, None], jnp.zeros_like(throttle), throttle)

    new_state = EnvStateTrack(new_mx, new_step, new_last, new_c, new_scale, new_w, new_rot)
    obs = jax.vmap(_track_get_obs)(
        mx_data, step_cnt, throttle,
        state.traj_c, state.traj_scale, state.traj_w, state.traj_rot)
    return new_state, obs, reward, done


# ===========================================================================
# FOREST task
# ===========================================================================

_FOREST_MAX_STEPS      = 800
_FOREST_STATE_DIM      = 23    # rpos(3)+qxyzw(4)+lin_vel(3)+ang_vel(3)+heading(3)+up(3)+throttle(4)
_FOREST_OBS_DIM        = 167   # _FOREST_STATE_DIM + 144 lidar rays
_FOREST_N_TREES        = 1000
_FOREST_TREE_RAD       = 0.25
_FOREST_TREE_H         = 4.0
_FOREST_LIDAR_RANGE    = 4.0
_FOREST_MIN_Z          = 0.2
_FOREST_MAX_Z          = 4.0
_FOREST_MAX_SPEED      = 2.5
_FOREST_MIN_DIST       = 0.3
_FOREST_VEL_CLIP       = 2.0
_FOREST_SAFETY_WEIGHT  = 0.2
_FOREST_START_Y     = -24.0
_FOREST_TARGET_Y    = 24.0
_FOREST_START_X     = 16.0   # x ∈ [-16, 16]
_FOREST_EXCL_R      = 1.5
_FOREST_AREA_X      = 16.0
_FOREST_AREA_Y      = 20.0


def _build_lidar_dirs():
    rows = []
    for v_deg in [-10.0, 0.0, 10.0, 20.0]:
        v = math.radians(v_deg)
        cv, sv = math.cos(v), math.sin(v)
        for k in range(36):
            h = 2.0 * math.pi * k / 36.0
            rows.append([cv * math.cos(h), cv * math.sin(h), sv])
    return jnp.array(rows, dtype=jnp.float32)  # (144, 3)


_LIDAR_DIRS = _build_lidar_dirs()  # (144, 3), module-level constant


def _ray_cylinder_dist(ray_origin, ray_dir, cyl_xy):
    """Analytical ray–vertical-cylinder distance (JAX, handles no-hit with inf)."""
    ox = ray_origin[0] - cyl_xy[0]
    oy = ray_origin[1] - cyl_xy[1]
    dx, dy = ray_dir[0], ray_dir[1]
    a    = dx*dx + dy*dy
    b    = 2.0 * (ox*dx + oy*dy)
    c    = ox*ox + oy*oy - _FOREST_TREE_RAD**2
    disc = b*b - 4.0*a*c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t    = (-b - sqrt_disc) / (2.0 * a)
    z_hit = ray_origin[2] + t * ray_dir[2]
    valid = (disc > 0.0) & (t > 0.01) & (z_hit >= 0.0) & (z_hit <= _FOREST_TREE_H)
    return jnp.where(valid, t, jnp.inf)


def _scan_lidar_forest(drone_pos, tree_xys):
    """Return (144,) lidar_scan in range-dist convention (0=clear, range=touching)."""
    def single_ray(ray_dir):
        dists = jax.vmap(lambda cxy: _ray_cylinder_dist(drone_pos, ray_dir, cxy))(tree_xys)
        return jnp.minimum(jnp.min(dists), _FOREST_LIDAR_RANGE)

    raw_dists = jax.vmap(single_ray)(_LIDAR_DIRS)  # (144,)
    return (_FOREST_LIDAR_RANGE - raw_dists).astype(jnp.float32)


def _forest_get_obs(mx_data, step_cnt, last_throttle, target_pos, tree_xys):
    pos     = mx_data.qpos[:3]
    qw      = mx_data.qpos[3:7]
    lin_vel = mx_data.qvel[:3]
    ang_vel = mx_data.qvel[3:6]
    R       = _quat_to_rot(qw)
    heading = R[:, 0]
    up      = R[:, 2]
    qxyzw   = jnp.array([qw[1], qw[2], qw[3], qw[0]])
    rpos_raw     = target_pos - pos
    rpos         = rpos_raw / (jnp.linalg.norm(rpos_raw) + 1e-6)
    throttle_obs = last_throttle * 2.0 - 1.0
    state_23 = jnp.concatenate([
        rpos, qxyzw, lin_vel, ang_vel, heading, up, throttle_obs
    ]).astype(jnp.float32)
    lidar_scan = _scan_lidar_forest(pos, tree_xys)
    return jnp.concatenate([state_23, lidar_scan]).astype(jnp.float32)


# ===========================================================================
# FOREST2 task — Forest with goal bonus and no speed-limit termination
# ===========================================================================

_FOREST2_MAX_STEPS            = 1600
_FOREST2_GOAL_RADIUS          = 2.0
_FOREST2_GOAL_REWARD_PER_STEP = 1.0   # 남은 스텝 수만큼 보너스


def _forest_reward_done(mx_data, last_throttle, target_pos, tree_xys, step_cnt):
    pos     = mx_data.qpos[:3]
    qw      = mx_data.qpos[3:7]
    lin_vel = mx_data.qvel[:3]
    R       = _quat_to_rot(qw)
    up      = R[:, 2]
    lidar_scan = _scan_lidar_forest(pos, tree_xys)
    diff     = target_pos - pos
    dist     = jnp.linalg.norm(diff)
    vel_dir  = diff / jnp.maximum(dist, 1e-6)
    v        = jnp.dot(lin_vel, vel_dir)
    r_vel    = jnp.clip(v, a_min=None, a_max=_FOREST_VEL_CLIP)
    r_up     = ((up[2] + 1.0) / 2.0) ** 2
    actual   = jnp.maximum(_FOREST_LIDAR_RANGE - lidar_scan, 1e-6)
    r_safety = _FOREST_SAFETY_WEIGHT * jnp.mean(jnp.log(actual))

    # Goal bonus
    goal_reached = dist < _FOREST2_GOAL_RADIUS
    # remaining  = jnp.maximum(_FOREST2_MAX_STEPS - step_cnt, 0).astype(jnp.float32)
    # goal_bonus = jnp.where(goal_reached, remaining * _FOREST2_GOAL_REWARD_PER_STEP, 0.0)

    reward   = (r_vel + r_up + 1.0 + r_safety).astype(jnp.float32)
    min_obs_dist = _FOREST_LIDAR_RANGE - jnp.max(lidar_scan)
    terminated = (
        (pos[2] < _FOREST_MIN_Z)
        | (pos[2] > _FOREST_MAX_Z)
        | (min_obs_dist < _FOREST_MIN_DIST)
        | goal_reached
    )
    parts = jnp.array([r_vel, r_up, r_safety], dtype=jnp.float32)
    return reward, terminated, parts


def _forest_reset_single(rng):
    rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
    start_x   = jax.random.uniform(k1, minval=-_FOREST_START_X, maxval=_FOREST_START_X)
    start_pos = jnp.array([start_x, _FOREST_START_Y, 2.0])
    target    = jnp.array([start_x, _FOREST_TARGET_Y, 2.0])

    # Sample 200 candidates, pick first N_TREES valid ones (vectorized)
    N_CANDS = _FOREST_N_TREES * 10
    cand_x  = jax.random.uniform(k2, (N_CANDS,), minval=-_FOREST_AREA_X, maxval=_FOREST_AREA_X)
    cand_y  = jax.random.uniform(k3, (N_CANDS,), minval=-_FOREST_AREA_Y, maxval=_FOREST_AREA_Y)
    d_s = jnp.sqrt((cand_x - start_x)**2 + (cand_y - _FOREST_START_Y)**2)
    d_t = jnp.sqrt((cand_x - start_x)**2 + (cand_y - _FOREST_TARGET_Y)**2)
    valid  = (d_s >= _FOREST_EXCL_R) & (d_t >= _FOREST_EXCL_R)
    cumsum = jnp.cumsum(valid.astype(jnp.int32))
    slots  = jnp.arange(1, _FOREST_N_TREES + 1)
    idx    = jnp.clip(jnp.searchsorted(cumsum, slots, side='left'), 0, N_CANDS - 1)
    tree_xys = jnp.stack([cand_x[idx], cand_y[idx]], axis=1).astype(jnp.float32)

    # Drone at start with random orientation (matching OmniDrones)
    rpy  = jax.random.uniform(k4, (3,),
                               minval=jnp.array([-0.2*math.pi, -0.2*math.pi, 0.]),
                               maxval=jnp.array([ 0.2*math.pi,  0.2*math.pi, 2*math.pi]))
    quat = _make_quat(rpy)
    dx   = mjx.make_data(_mx_model)
    dx   = dx.replace(qpos=jnp.concatenate([start_pos, quat]), qvel=jnp.zeros(6))
    dx   = mjx.forward(_mx_model, dx)
    return dx, tree_xys, target.astype(jnp.float32)


class EnvStateForest(NamedTuple):
    mx_data:       object
    step_cnt:      jnp.ndarray   # (n,)
    last_throttle: jnp.ndarray   # (n, 4)
    obs_xy:        jnp.ndarray   # (n, N_TREES, 2)
    target_pos:    jnp.ndarray   # (n, 3)


def forest_batch_reset(rng, n_envs):
    keys = jax.random.split(rng, n_envs)
    mx_data, tree_xys, target = jax.vmap(_forest_reset_single)(keys)
    return EnvStateForest(
        mx_data       = mx_data,
        step_cnt      = jnp.zeros(n_envs, jnp.int32),
        last_throttle = jnp.zeros((n_envs, 4), jnp.float32),
        obs_xy        = tree_xys,
        target_pos    = target,
    )


def forest_batch_obs(state: EnvStateForest):
    return jax.vmap(_forest_get_obs)(
        state.mx_data, state.step_cnt, state.last_throttle,
        state.target_pos, state.obs_xy)


def forest_batch_step(state: EnvStateForest, action, rng):
    throttle = (action + 1.0) / 2.0
    mx_data  = jax.vmap(_apply_forces)(state.mx_data, action)
    mx_data  = jax.vmap(mjx.step, in_axes=(None, 0))(_mx_model, mx_data)
    step_cnt = state.step_cnt + 1
    reward, terminated, rew_parts = jax.vmap(_forest_reward_done)(
        mx_data, throttle, state.target_pos, state.obs_xy, step_cnt)
    truncated = step_cnt >= _FOREST2_MAX_STEPS
    done      = terminated | truncated

    keys = jax.random.split(rng, action.shape[0])
    r_dx, r_trees, r_target = jax.vmap(_forest_reset_single)(keys)

    def _where(reset_v, curr_v):
        mask = done.reshape(-1, *([1] * (curr_v.ndim - 1)))
        return jnp.where(mask, reset_v, curr_v)

    new_mx     = jax.tree.map(_where, r_dx, mx_data)
    new_trees  = _where(r_trees,  state.obs_xy)
    new_target = _where(r_target, state.target_pos)
    new_step   = jnp.where(done, 0, step_cnt)
    new_last   = jnp.where(done[:, None], jnp.zeros_like(throttle), throttle)

    new_state = EnvStateForest(new_mx, new_step, new_last, new_trees, new_target)
    obs = jax.vmap(_forest_get_obs)(
        mx_data, step_cnt, throttle, state.target_pos, state.obs_xy)
    return new_state, obs, reward, done, rew_parts


# ===========================================================================
# PPO network — same architecture as DreamerV3 actor/critic
#   MLP: 3 × Dense(1024) + RMSNorm + SiLU  (separate trunks for actor/critic)
#   Policy output  : bounded_normal  (tanh mean, sigmoid std ∈ [0.1, 1.0])
#   Value output   : symexp_twohot   (255 bins, symmetric-log spacing)
# ===========================================================================

_TWOHOT_BINS  = 255
_POLICY_MINSTD = 0.1
_POLICY_MAXSTD = 1.0
# trunc_normal_in initializer  (≈ LeCun normal)
_WINIT = nn.initializers.variance_scaling(1.0, mode='fan_in', distribution='truncated_normal')
# outscale=0.01 for actor heads, outscale=0.0 (zeros) for value head
_ACTOR_OUTINIT = nn.initializers.variance_scaling(1e-4, mode='fan_in', distribution='truncated_normal')


def _make_symexp_bins(n=_TWOHOT_BINS):
    half = jnp.linspace(-20.0, 0.0, (n - 1) // 2 + 1, dtype=jnp.float32)
    half = jnp.sign(half) * jnp.expm1(jnp.abs(half))   # symexp
    return jnp.concatenate([half, -half[:-1][::-1]])     # (255,)


_BINS = _make_symexp_bins()   # (255,)  — module-level constant


def twohot_decode(logits, bins=_BINS):
    """Decode twohot logits → scalar value."""
    probs = jax.nn.softmax(logits, axis=-1)
    return (probs * bins).sum(-1)


def twohot_loss(logits, target, bins=_BINS):
    """Cross-entropy loss with soft twohot target."""
    below = jnp.sum(bins <= target[..., None], axis=-1) - 1
    below = jnp.clip(below, 0, len(bins) - 2)
    above = below + 1
    w_upper = jnp.clip((target - bins[below]) /
                       jnp.maximum(bins[above] - bins[below], 1e-8), 0.0, 1.0)
    w_lower = 1.0 - w_upper
    # build soft target with scatter
    tgt = (jax.nn.one_hot(below, len(bins)) * w_lower[..., None]
         + jax.nn.one_hot(above, len(bins)) * w_upper[..., None])
    log_p = jax.nn.log_softmax(logits, axis=-1)
    return -(tgt * log_p).sum(-1)


class RMSNorm(nn.Module):
    """Identical to DreamerV3 Norm(impl='rms', eps=1e-4)."""
    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        mean2 = jnp.square(x).mean(-1, keepdims=True)
        return (x * jax.lax.rsqrt(mean2 + 1e-4) * scale).astype(x.dtype)


_TASK_CONFIG_NAME = {'hover': 'mujocohover', 'track': 'mujocotrack', 'forest': 'mujocoforest'}

def _load_ppo_config(task=None):
    """Read enc/policy/value/run settings from dreamerv3/configs.yaml."""
    import ruamel.yaml as yaml
    path = pathlib.Path(__file__).parent / 'dreamerv3' / 'configs.yaml'
    cfg = yaml.YAML(typ='safe').load(path.read_text())
    ag = cfg['defaults']['agent']
    run_steps = None
    log_every  = 1    # PPO logs every 1 second by default
    if task is not None:
        cfg_name = _TASK_CONFIG_NAME.get(task)
        if cfg_name and cfg_name in cfg:
            task_run = cfg[cfg_name].get('run', {})
            run_steps  = int(task_run.get('steps',      None) or 0) or None
            log_every  = int(task_run.get('log_every',  log_every))
    return ag['enc']['simple'], ag['policy'], ag['value'], run_steps, log_every


class ActorCritic(nn.Module):
    """
    DreamerV3-style encoder + separate actor/critic trunks.
    Config loaded from dreamerv3/configs.yaml via _load_ppo_config().
    Encoder: symlog → Dense(enc_units)+RMSNorm+SiLU × enc_layers → feat
    Actor:   feat → Dense(pol_units)+RMSNorm+SiLU × pol_layers → bounded_normal
    Critic:  feat → Dense(pol_units)+RMSNorm+SiLU × pol_layers → symexp_twohot
    """
    enc_layers: int
    enc_units:  int
    pol_layers: int
    pol_units:  int
    val_bins:   int

    @nn.compact
    def __call__(self, obs):
        # ---- encoder (symlog + MLP) ----
        x = jnp.sign(obs) * jnp.log1p(jnp.abs(obs))  # symlog
        for _ in range(self.enc_layers):
            x = nn.Dense(self.enc_units, kernel_init=_WINIT)(x)
            x = RMSNorm()(x)
            x = jax.nn.silu(x)
        feat = x  # (batch, enc_units)

        # ---- actor ----
        h = feat
        for _ in range(self.pol_layers):
            h = nn.Dense(self.pol_units, kernel_init=_WINIT)(h)
            h = RMSNorm()(h)
            h = jax.nn.silu(h)
        mean_raw = nn.Dense(_ACT_DIM, kernel_init=_ACTOR_OUTINIT)(h)
        std_raw  = nn.Dense(_ACT_DIM, kernel_init=_ACTOR_OUTINIT)(h)
        mean = jnp.tanh(mean_raw)
        std  = (_POLICY_MAXSTD - _POLICY_MINSTD) * jax.nn.sigmoid(std_raw + 2.0) + _POLICY_MINSTD

        # ---- critic ----
        v = feat
        for _ in range(self.pol_layers):
            v = nn.Dense(self.pol_units, kernel_init=_WINIT)(v)
            v = RMSNorm()(v)
            v = jax.nn.silu(v)
        value_logits = nn.Dense(self.val_bins, kernel_init=nn.initializers.zeros)(v)

        return mean, std, value_logits


def bounded_normal_log_prob(action, mean, std):
    """Log prob of bounded_normal: N(tanh_mean, std) evaluated at action."""
    return -0.5 * (((action - mean) / std) ** 2
                   + 2.0 * jnp.log(std)
                   + jnp.log(2.0 * math.pi)).sum(-1)


def sample_action(mean, std, rng):
    eps = jax.random.normal(rng, mean.shape)
    return jnp.clip(mean + std * eps, -1., 1.)


# ===========================================================================
# PPO training (task-agnostic)
# ===========================================================================

class PPOConfig(NamedTuple):
    n_envs:      int   = 1024
    n_steps:     int   = 512
    n_epochs:    int   = 10
    n_minibatch: int   = 32
    gamma:       float = 0.99
    lam:         float = 0.95
    clip_eps:    float = 0.2
    ent_coef:    float = 0.01
    vf_coef:     float = 0.5
    lr:          float = 3e-4
    max_grad:    float = 0.5


class Transition(NamedTuple):
    obs:       jnp.ndarray
    action:    jnp.ndarray
    reward:    jnp.ndarray
    done:      jnp.ndarray
    value:     jnp.ndarray
    log_prob:  jnp.ndarray
    rew_parts: jnp.ndarray  # (n_envs, 3): [r_vel, r_up, r_safety], zeros for non-forest tasks


def save_checkpoint(logdir: pathlib.Path, params, step: int):
    ckpt_dir = logdir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'{step:012d}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    # update latest symlink
    latest = ckpt_dir / 'latest.pkl'
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(path.name)
    return path


def load_checkpoint(logdir: pathlib.Path):
    ckpt_dir = logdir / 'ckpt'
    latest = ckpt_dir / 'latest.pkl'
    if latest.exists():
        with open(latest, 'rb') as f:
            return pickle.load(f)
    return None


def _smooth(vals, steps, w):
    import numpy as _np
    if len(vals) < w:
        return steps, vals
    sv = _np.convolve(vals, _np.ones(w) / w, mode='valid')
    return steps[w-1:], sv


def save_plot(logdir: pathlib.Path, metrics: list):
    """Delegate plotting to plot_metrics.py (same as DreamerV3)."""
    import subprocess, sys as _sys
    script = pathlib.Path(__file__).parent / 'plot_metrics.py'
    subprocess.Popen(
        [_sys.executable, str(script), str(logdir)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def make_train(cfg: PPOConfig, obs_dim, batch_reset_fn, batch_step_fn, batch_obs_fn,
               max_steps_per_ep, logdir: pathlib.Path = None,
               log_every_secs: int = 120):
    enc_cfg, pol_cfg, val_cfg, *_ = _load_ppo_config()
    net = ActorCritic(
        enc_layers=enc_cfg['layers'], enc_units=enc_cfg['units'],
        pol_layers=pol_cfg['layers'], pol_units=pol_cfg['units'],
        val_bins=val_cfg['bins'],
    )
    tx  = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad),
        optax.adam(cfg.lr),
    )

    @jax.jit
    def collect_rollout(params, state, rng):
        def _step(carry, _):
            st, rng = carry
            obs  = batch_obs_fn(st)
            rng, k1, k2 = jax.random.split(rng, 3)
            mean, std, value_logits = net.apply(params, obs)
            value    = twohot_decode(value_logits)
            action   = sample_action(mean, std, k1)
            log_prob = bounded_normal_log_prob(action, mean, std)
            step_out = batch_step_fn(st, action, k2)
            new_st, next_obs, reward, done = step_out[:4]
            rew_parts = step_out[4] if len(step_out) > 4 else jnp.zeros((obs.shape[0], 3))
            return (new_st, rng), Transition(obs, action, reward, done, value, log_prob, rew_parts)

        (state, rng), traj = jax.lax.scan(
            _step, (state, rng), None, length=cfg.n_steps)
        last_obs = batch_obs_fn(state)
        _, _, last_logits = net.apply(params, last_obs)
        last_value = twohot_decode(last_logits)
        return traj, last_value, state, rng

    def gae(traj: Transition, last_value):
        def _step(carry, rvd):
            gae_next, next_val = carry
            r, d, v = rvd
            delta  = r + cfg.gamma * next_val * (1 - d) - v
            gae_t  = delta + cfg.gamma * cfg.lam * (1 - d) * gae_next
            return (gae_t, v), gae_t
        _, advantages = jax.lax.scan(
            _step,
            (jnp.zeros_like(last_value), last_value),
            (traj.reward, traj.done.astype(jnp.float32), traj.value),
            reverse=True,
        )
        return advantages, advantages + traj.value

    @jax.jit
    def update(params, opt_state, traj: Transition, advantages, returns, rng):
        n       = cfg.n_envs * cfg.n_steps
        mb_size = n // cfg.n_minibatch
        adv_flat = advantages.reshape(n)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        def _epoch(carry, _):
            p, os, rng = carry
            rng, k = jax.random.split(rng)
            idx  = jax.random.permutation(k, n)
            obs_mb = traj.obs.reshape(n, obs_dim)[idx].reshape(cfg.n_minibatch, mb_size, obs_dim)
            act_mb = traj.action.reshape(n, _ACT_DIM)[idx].reshape(cfg.n_minibatch, mb_size, _ACT_DIM)
            lp_mb  = traj.log_prob.reshape(n)[idx].reshape(cfg.n_minibatch, mb_size)
            adv_mb = adv_flat[idx].reshape(cfg.n_minibatch, mb_size)
            ret_mb = returns.reshape(n)[idx].reshape(cfg.n_minibatch, mb_size)

            def _minibatch(carry, batch):
                p, os = carry
                obs_b, act_b, lp_b, adv_b, ret_b = batch
                def loss_fn(p):
                    mean, std, value_logits = net.apply(p, obs_b)
                    new_lp  = bounded_normal_log_prob(act_b, mean, std)
                    ratio   = jnp.exp(new_lp - lp_b)
                    pg_loss = -jnp.minimum(
                        ratio * adv_b,
                        jnp.clip(ratio, 1-cfg.clip_eps, 1+cfg.clip_eps) * adv_b
                    ).mean()
                    # twohot cross-entropy value loss (same as DreamerV3)
                    vf_loss = twohot_loss(value_logits, ret_b).mean()
                    entropy = -new_lp.mean()
                    return pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy
                g = jax.grad(loss_fn)(p)
                updates, new_os = tx.update(g, os, p)
                return (optax.apply_updates(p, updates), new_os), None

            (p, os), _ = jax.lax.scan(
                _minibatch, (p, os),
                (obs_mb, act_mb, lp_mb, adv_mb, ret_mb))
            return (p, os, rng), None

        (params, opt_state, _), _ = jax.lax.scan(
            _epoch, (params, opt_state, rng), None, length=cfg.n_epochs)
        return params, opt_state

    def train(total_env_steps, rng):
        rng, k1, k2 = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((1, obs_dim))
        params    = net.init(k1, dummy_obs)
        opt_state = tx.init(params)

        # resume from checkpoint if exists
        start_update = 0
        if logdir is not None:
            saved = load_checkpoint(logdir)
            if saved is not None:
                params = saved
                # count existing steps from latest ckpt filename
                ckpt_files = sorted((logdir / 'ckpt').glob('[0-9]*.pkl'))
                if ckpt_files:
                    start_update = int(ckpt_files[-1].stem) // (cfg.n_envs * cfg.n_steps)
                    print(f'Resumed from step {start_update * cfg.n_envs * cfg.n_steps:,}')

        state     = batch_reset_fn(k2, cfg.n_envs)
        n_updates = total_env_steps // (cfg.n_envs * cfg.n_steps)

        metrics_path = (logdir / 'metrics.jsonl') if logdir else None
        all_metrics  = []
        t0           = time.time()
        last_log_t   = t0
        best_score   = -float('inf')

        for i in range(start_update, n_updates):
            rng, k = jax.random.split(rng)
            traj, last_val, state, rng = collect_rollout(params, state, k)
            advantages, returns = gae(traj, last_val)
            rng, k = jax.random.split(rng)
            params, opt_state = update(params, opt_state, traj, advantages, returns, k)

            now = time.time()
            if now - last_log_t >= log_every_secs:
                last_log_t = now
                ep_steps    = (i + 1) * cfg.n_envs * cfg.n_steps
                mean_reward = float(traj.reward.mean())
                done_rate   = float(traj.done.mean())
                ep_len      = (1.0 / done_rate) if done_rate > 0 else max_steps_per_ep
                fps         = ep_steps / (now - t0)
                rew_np  = np.array(traj.reward)   # (n_steps, n_envs)
                done_np = np.array(traj.done)
                _ep_scores, _running = [], np.zeros(rew_np.shape[1])
                for _t in range(rew_np.shape[0]):
                    _running += rew_np[_t]
                    for _e in np.where(done_np[_t])[0]:
                        _ep_scores.append(_running[_e]); _running[_e] = 0.0
                score = float(np.mean(_ep_scores)) if _ep_scores else mean_reward * ep_len
                print(f'  step={ep_steps:>10,}  approx_score={score:6.2f}  ep_len={ep_len:5.0f}  fps={fps:,.0f}')

                entry = {
                    'step':                ep_steps,
                    'episode/score':       round(score, 4),
                    'episode/length':      round(ep_len, 1),
                    'episode/mean_reward': round(mean_reward, 4),
                    'fps/train':           int(fps),
                }
                parts = traj.rew_parts
                if parts is not None and float(parts.mean()) != 0.0:
                    entry['epstats/log/r_vel/avg']    = round(float(parts[..., 0].mean()), 4)
                    entry['epstats/log/r_up/avg']     = round(float(parts[..., 1].mean()), 4)
                    entry['epstats/log/r_safety/avg'] = round(float(parts[..., 2].mean()), 4)

                all_metrics.append(entry)
                if metrics_path is not None:
                    with open(metrics_path, 'a') as mf:
                        mf.write(json.dumps(entry) + '\n')

                if logdir is not None and score > best_score:
                    best_score = score
                    best_path = logdir / 'ckpt' / 'best.pkl'
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(best_path, 'wb') as f:
                        import pickle as _pickle
                        _pickle.dump(jax.device_get(params), f)

                if logdir is not None:
                    save_checkpoint(logdir, params, ep_steps)
                    if len(all_metrics) > 1:
                        save_plot(logdir, all_metrics)

        # final save — always write last metrics entry + checkpoint + plot
        if n_updates == 0:
            print('  [warn] 0 updates completed (total_steps too small); skipping final save.')
            return params, net
        ep_steps = n_updates * cfg.n_envs * cfg.n_steps
        now = time.time()
        mean_reward = float(traj.reward.mean())
        done_rate   = float(traj.done.mean())
        ep_len      = (1.0 / done_rate) if done_rate > 0 else max_steps_per_ep
        fps         = ep_steps / (now - t0)
        rew_np  = np.array(traj.reward)
        done_np = np.array(traj.done)
        _ep_scores, _running = [], np.zeros(rew_np.shape[1])
        for _t in range(rew_np.shape[0]):
            _running += rew_np[_t]
            for _e in np.where(done_np[_t])[0]:
                _ep_scores.append(_running[_e]); _running[_e] = 0.0
        score = float(np.mean(_ep_scores)) if _ep_scores else mean_reward * ep_len
        entry = {
            'step':                ep_steps,
            'episode/score':       round(score, 4),
            'episode/length':      round(ep_len, 1),
            'episode/mean_reward': round(mean_reward, 4),
            'fps/train':           int(fps),
        }
        parts = traj.rew_parts
        if parts is not None and float(parts.mean()) != 0.0:
            entry['epstats/log/r_vel/avg']    = round(float(parts[..., 0].mean()), 4)
            entry['epstats/log/r_up/avg']     = round(float(parts[..., 1].mean()), 4)
            entry['epstats/log/r_safety/avg'] = round(float(parts[..., 2].mean()), 4)
        all_metrics.append(entry)
        if metrics_path is not None:
            with open(metrics_path, 'a') as mf:
                mf.write(json.dumps(entry) + '\n')
        if logdir is not None:
            ckpt = save_checkpoint(logdir, params, ep_steps)
            print(f'  [saved] {ckpt}')
            if len(all_metrics) > 1:
                save_plot(logdir, all_metrics)
            print(f'  [done] logdir: {logdir}')

        return params, net

    return train



# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',       type=str,  default='hover',
                        choices=['hover', 'track', 'forest'])
    parser.add_argument('--logdir',     type=str,  default=None,
                        help='checkpoint/log directory (default: ~/logdir/ppo/<task>)')
    parser.add_argument('--n_envs',     type=int,  default=1024)
    parser.add_argument('--steps',      type=int,  default=10_000_000,
                        help='total env steps (default: 10M)')
    args = parser.parse_args()

    _, _, _, _, log_every_secs = _load_ppo_config(task=args.task)
    total_steps = args.steps

    # resolve logdir
    if args.logdir is None:
        args.logdir = os.path.expanduser(f'~/logdir/ppo/{args.task}')
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    cfg = PPOConfig(n_envs=args.n_envs)

    # save config
    config = {
        'task': args.task, 'n_envs': cfg.n_envs, 'n_steps': cfg.n_steps,
        'n_epochs': cfg.n_epochs, 'n_minibatch': cfg.n_minibatch,
        'gamma': cfg.gamma, 'lam': cfg.lam, 'clip_eps': cfg.clip_eps,
        'ent_coef': cfg.ent_coef, 'vf_coef': cfg.vf_coef,
        'lr': cfg.lr, 'total_steps': total_steps,
    }
    with open(logdir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    if args.task == 'hover':
        obs_dim       = _HOVER_OBS_DIM
        batch_reset   = hover_batch_reset
        batch_step    = hover_batch_step
        batch_obs     = hover_batch_obs
        max_steps_ep  = _HOVER_MAX_STEPS
    elif args.task == 'track':
        obs_dim       = _TRACK_OBS_DIM
        batch_reset   = track_batch_reset
        batch_step    = track_batch_step
        batch_obs     = track_batch_obs
        max_steps_ep  = _TRACK_MAX_STEPS
    else:  # forest
        obs_dim       = _FOREST_OBS_DIM
        batch_reset   = forest_batch_reset
        batch_step    = forest_batch_step
        batch_obs     = forest_batch_obs
        max_steps_ep  = _FOREST2_MAX_STEPS

    train = make_train(cfg, obs_dim, batch_reset, batch_step, batch_obs, max_steps_ep,
                       logdir=logdir, log_every_secs=log_every_secs)

    rng = jax.random.PRNGKey(0)
    print(f'Task: {args.task}  n_envs={cfg.n_envs}  total_steps={total_steps:,}')
    print(f'logdir: {logdir}')
    train(total_steps, rng)


if __name__ == '__main__':
    main()
