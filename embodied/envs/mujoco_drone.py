"""
MuJoCo drone environments for DreamerV3.

Tasks:
  hover  — fixed target [0,0,2], OmniDrones Hover-style  (30-dim obs)
  track  — lemniscate trajectory tracking, OmniDrones Track-style (36-dim obs)
  forest — obstacle avoidance navigation, OmniDrones Forest-style (167-dim obs)

Usage:
  conda activate test2
  cd /home/psm/workspaces/rl/dreamerv3
  python -c "from embodied.envs.mujoco_drone import MujocoDrone; ..."
"""

import functools
import math
import os
import numpy as np
import elements
import embodied

import mujoco

_ASSETS_DIR    = os.path.join(os.path.dirname(__file__), 'assets')
_QUADROTOR_XML = os.path.join(_ASSETS_DIR, 'quadrotor.xml')
_FOREST_XML    = os.path.join(_ASSETS_DIR, 'quadrotor_forest.xml')

# ---------------------------------------------------------------------------
# Hummingbird rotor parameters (from OmniDrones)
# ---------------------------------------------------------------------------
_KF         = 8.54858e-6     # thrust coefficient  [N / (rad/s)^2]
_KM         = 1.36777e-7     # torque coefficient  [N·m / (rad/s)^2]
_MAX_RPM    = 838.0           # max rotor speed     [rad/s]
_DIRECTIONS = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)  # CW/CCW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_wxyz_to_axes(quat_wxyz):
    """MuJoCo quaternion [qw,qx,qy,qz] → (heading, up) unit vectors."""
    qw, qx, qy, qz = (float(quat_wxyz[i]) for i in range(4))
    heading = np.array([
        1.0 - 2.0 * (qy*qy + qz*qz),
        2.0 * (qx*qy + qw*qz),
        2.0 * (qx*qz - qw*qy),
    ], dtype=np.float32)
    up = np.array([
        2.0 * (qx*qz + qw*qy),
        2.0 * (qy*qz - qw*qx),
        1.0 - 2.0 * (qx*qx + qy*qy),
    ], dtype=np.float32)
    return heading, up


def _euler_to_quat_wxyz(roll, pitch, yaw):
    """ZYX Euler → quaternion [qw, qx, qy, qz]."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,   # qw
        sr * cp * cy - cr * sp * sy,   # qx
        cr * sp * cy + sr * cp * sy,   # qy
        cr * cp * sy - sr * sp * cy,   # qz
    ], dtype=np.float64)


def _scale_time(t: float) -> float:
    """Nonlinear time scaling matching OmniDrones."""
    if abs(t) < 1e-6:
        return 0.0
    return t / (1.0 + 1.0 / abs(t))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class MujocoDrone(embodied.Env):
    """Factory: dispatches to the correct subclass based on task name."""

    _TASK_MAP = {}   # populated after subclass definitions

    def __new__(cls, task='hover', gui=False):
        if cls is MujocoDrone:
            subclass = cls._TASK_MAP.get(task)
            if subclass is None:
                raise ValueError(
                    f'Unknown MuJoCo drone task: {task!r}. '
                    f'Choose from {list(cls._TASK_MAP)}')
            obj = object.__new__(subclass)
            return obj
        return object.__new__(cls)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Shared physics mixin
# ---------------------------------------------------------------------------

class _DroneBase(MujocoDrone):
    """Shared rotor-force logic for all MuJoCo drone tasks."""

    def _init_mujoco(self):
        self.model = mujoco.MjModel.from_xml_path(_QUADROTOR_XML)
        self.data  = mujoco.MjData(self.model)
        self._drone_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'drone')
        self._rotor_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'rotor{i}')
            for i in range(4)
        ]

    def _apply_rotor_forces(self, throttle_01):
        rpm    = throttle_01 * _MAX_RPM
        thrust = _KF * rpm ** 2
        torque = _KM * rpm ** 2
        R      = self.data.xmat[self._drone_body_id].reshape(3, 3)

        world_force  = R @ np.array([0.0, 0.0, thrust.sum()])
        world_torque = R @ np.array([0.0, 0.0, np.sum(_DIRECTIONS * torque)])

        for i, site_id in enumerate(self._rotor_site_ids):
            r_world = (self.data.site_xpos[site_id]
                       - self.data.xpos[self._drone_body_id])
            world_torque += np.cross(r_world, R @ np.array([0.0, 0.0, thrust[i]]))

        self.data.xfrc_applied[self._drone_body_id, :3] = world_force
        self.data.xfrc_applied[self._drone_body_id, 3:] = world_torque

    @functools.cached_property
    def act_space(self):
        return {
            'reset':  elements.Space(bool),
            'action': elements.Space(np.float32, (4,), -1, 1),
        }

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Hover
# ---------------------------------------------------------------------------

class MuJoCoHoverDrone(_DroneBase):
    """
    OmniDrones Hover ported to MuJoCo.

    obs  : 30-dim
             [0:3]   rpos        (target_pos - drone_pos)
             [3:7]   quat        [qx, qy, qz, qw]
             [7:13]  vel         lin_vel(3) + ang_vel(3)
             [13:16] heading
             [16:19] up
             [19:23] throttle    last action, [-1,1]
             [23:26] rheading    target_heading - drone_heading
             [26:30] time_enc
    """

    OBS_DIM   = 30
    MAX_STEPS = 500
    TARGET_POS = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    _MIN_Z    = 0.2
    _MAX_DIST = 4.0

    def __init__(self, task='hover', gui=False):
        self._init_mujoco()
        self._gui           = gui
        self._step_cnt      = 0
        self._done          = True
        self._last_throttle = np.zeros(4, dtype=np.float32)
        self._target_heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    @functools.cached_property
    def obs_space(self):
        return {
            'state':       elements.Space(np.float32, (self.OBS_DIM,)),
            'reward':      elements.Space(np.float32),
            'is_first':    elements.Space(bool),
            'is_last':     elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        pos      = qpos[0:3].astype(np.float32)
        quat_w   = qpos[3:7]
        lin_vel  = qvel[0:3].astype(np.float32)
        ang_vel  = qvel[3:6].astype(np.float32)
        heading, up = _quat_wxyz_to_axes(quat_w)
        quat_xyzw = np.array([quat_w[1], quat_w[2], quat_w[3], quat_w[0]],
                              dtype=np.float32)
        rpos     = (self.TARGET_POS - pos).astype(np.float32)
        rheading = (self._target_heading - heading).astype(np.float32)
        t_norm   = np.float32(min(self._step_cnt / self.MAX_STEPS, 1.0))
        time_enc = np.full(4, t_norm, dtype=np.float32)
        throttle_obs = (self._last_throttle * 2.0 - 1.0).astype(np.float32)
        obs = np.concatenate([
            rpos, quat_xyzw, lin_vel, ang_vel, heading, up,
            throttle_obs, rheading, time_enc,
        ]).astype(np.float32)
        return obs, pos, heading, up, ang_vel

    def _compute_reward(self, pos, heading, up, ang_vel, throttle_01):
        rpos     = self.TARGET_POS - pos
        rheading = self._target_heading - heading
        distance = float(np.linalg.norm(np.concatenate([rpos, rheading])))
        r_pose   = 1.0 / (1.0 + (1.2 * distance) ** 2)
        r_up     = ((float(up[2]) + 1.0) / 2.0) ** 2
        r_spin   = 1.0 / (1.0 + float(ang_vel[2]) ** 4)
        r_effort = 0.1 * math.exp(-float(throttle_01.sum()))
        reward   = r_pose + r_pose * (r_up + r_spin) + r_effort
        return np.float32(reward), float(np.linalg.norm(rpos)), distance

    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()
        act         = np.clip(action['action'], -1.0, 1.0).astype(np.float32)
        throttle_01 = ((act + 1.0) / 2.0).astype(np.float64)
        self._apply_rotor_forces(throttle_01)
        mujoco.mj_step(self.model, self.data)
        self._step_cnt      += 1
        self._last_throttle  = throttle_01.astype(np.float32)
        obs_arr, pos, heading, up, ang_vel = self._get_obs()
        reward, pos_dist, full_dist = self._compute_reward(
            pos, heading, up, ang_vel, throttle_01)
        terminated = bool(pos[2] < self._MIN_Z or full_dist > self._MAX_DIST)
        truncated  = self._step_cnt >= self.MAX_STEPS
        done       = terminated or truncated
        self._done = done
        return {
            'state':       obs_arr,
            'reward':      reward,
            'is_first':    False,
            'is_last':     done,
            'is_terminal': terminated,
        }

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        px = np.random.uniform(-2.5, 2.5)
        py = np.random.uniform(-2.5, 2.5)
        pz = np.random.uniform(1.0, 2.5)
        roll  = np.random.uniform(-0.2, 0.2) * math.pi
        pitch = np.random.uniform(-0.2, 0.2) * math.pi
        yaw   = np.random.uniform(0.0, 2.0 * math.pi)
        quat  = _euler_to_quat_wxyz(roll, pitch, yaw)
        self.data.qpos[:3]  = [px, py, pz]
        self.data.qpos[3:7] = quat
        self.data.qvel[:]   = 0.0
        target_yaw = np.random.uniform(0.0, 2.0 * math.pi)
        self._target_heading = np.array([
            math.cos(target_yaw), math.sin(target_yaw), 0.0
        ], dtype=np.float32)
        self._step_cnt      = 0
        self._done          = False
        self._last_throttle = np.zeros(4, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)
        obs_arr, _, _, _, _ = self._get_obs()
        return {
            'state':       obs_arr,
            'reward':      np.float32(0.0),
            'is_first':    True,
            'is_last':     False,
            'is_terminal': False,
        }


# ---------------------------------------------------------------------------
# Track (lemniscate trajectory — OmniDrones Track-style)
# ---------------------------------------------------------------------------

class MuJoCoTrackDrone(_DroneBase):
    """
    OmniDrones Track ported to MuJoCo.

    obs  : 36-dim
             [0:12]  rpos to 4 future waypoints (step_size=5 apart)
             [12:16] quat [qx, qy, qz, qw]
             [16:19] lin_vel
             [19:22] ang_vel
             [22:25] heading
             [25:28] up
             [28:32] throttle (last action, [-1,1])
             [32:36] time_encoding
    """

    OBS_DIM      = 36
    FUTURE_STEPS = 4
    STEP_SIZE    = 5
    ORIGIN       = np.array([0., 0., 2.], dtype=np.float32)
    TRAJ_T0      = math.pi / 2
    MAX_STEPS    = 600    # 600 × 0.016s ≈ 9.6s
    RESET_THRES  = 0.5

    def __init__(self, task='track', gui=False):
        self._init_mujoco()
        self._gui           = gui
        self._step_cnt      = 0
        self._done          = True
        self._last_throttle = np.zeros(4, dtype=np.float32)
        self._traj_c        = 0.0
        self._traj_scale    = np.ones(3, dtype=np.float32)
        self._traj_w        = 1.0
        self._traj_rot      = np.eye(3, dtype=np.float32)

    @functools.cached_property
    def obs_space(self):
        return {
            'state':       elements.Space(np.float32, (self.OBS_DIM,)),
            'reward':      elements.Space(np.float32),
            'is_first':    elements.Space(bool),
            'is_last':     elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    def _make_traj_params(self):
        self._traj_c     = float(np.random.uniform(-0.6, 0.6))
        self._traj_scale = np.random.uniform(
            [1.8, 1.8, 1.0], [3.2, 3.2, 1.5]).astype(np.float32)
        self._traj_w     = float(
            np.random.uniform(0.8, 1.1) * np.random.choice([-1.0, 1.0]))
        yaw = float(np.random.uniform(0, 2 * math.pi))
        cy, sy = math.cos(yaw), math.sin(yaw)
        self._traj_rot = np.array(
            [[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]], dtype=np.float32)

    def _lemniscate_world(self, t: float) -> np.ndarray:
        sin_t = math.sin(t)
        cos_t = math.cos(t)
        denom = sin_t * sin_t + 1.0
        raw = np.array([cos_t / denom,
                        sin_t * cos_t / denom,
                        self._traj_c * sin_t / denom], dtype=np.float32)
        return self.ORIGIN + (self._traj_rot @ raw) * self._traj_scale

    def _get_ref_pos(self, step_offset: int = 0) -> np.ndarray:
        raw_t = self._traj_w * (self._step_cnt + step_offset) / 60.0
        t = self.TRAJ_T0 + _scale_time(raw_t)
        return self._lemniscate_world(t)

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        pos      = qpos[0:3].astype(np.float32)
        quat_w   = qpos[3:7]
        lin_vel  = qvel[0:3].astype(np.float32)
        ang_vel  = qvel[3:6].astype(np.float32)
        heading, up = _quat_wxyz_to_axes(quat_w)
        quat_xyzw = np.array([quat_w[1], quat_w[2], quat_w[3], quat_w[0]],
                              dtype=np.float32)
        rpos_list = []
        for k in range(self.FUTURE_STEPS):
            ref = self._get_ref_pos(self.STEP_SIZE * k)
            rpos_list.append((ref - pos).astype(np.float32))
        rpos = np.concatenate(rpos_list)
        throttle_obs = (self._last_throttle * 2.0 - 1.0).astype(np.float32)
        t_norm   = np.float32(min(self._step_cnt / self.MAX_STEPS, 1.0))
        time_enc = np.full(4, t_norm, dtype=np.float32)
        obs = np.concatenate([
            rpos, quat_xyzw, lin_vel, ang_vel, heading, up, throttle_obs, time_enc,
        ]).astype(np.float32)
        return obs, pos, ang_vel

    def _compute_reward(self, pos, ang_vel, throttle_01):
        ref_pos  = self._get_ref_pos(0)
        dist     = float(np.linalg.norm(ref_pos - pos))
        quat_w   = self.data.qpos[3:7]
        _, up    = _quat_wxyz_to_axes(quat_w)
        tiltage  = abs(1.0 - float(up[2]))
        spin     = float(ang_vel[2]) ** 2
        r_pose   = math.exp(-1.6 * dist)
        r_up     = 0.5 / (1.0 + tiltage ** 2)
        r_spin   = 0.5 / (1.0 + spin ** 2)
        r_effort = 0.1 * math.exp(-float(throttle_01.sum()))
        reward   = r_pose + r_pose * (r_up + r_spin) + r_effort
        return np.float32(reward), dist

    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()
        act         = np.clip(action['action'], -1.0, 1.0).astype(np.float32)
        throttle_01 = ((act + 1.0) / 2.0).astype(np.float64)
        self._apply_rotor_forces(throttle_01)
        mujoco.mj_step(self.model, self.data)
        self._step_cnt      += 1
        self._last_throttle  = throttle_01.astype(np.float32)
        obs_arr, pos, ang_vel = self._get_obs()
        reward, dist          = self._compute_reward(pos, ang_vel, throttle_01)
        terminated = bool(pos[2] < 0.1 or dist > self.RESET_THRES)
        truncated  = self._step_cnt >= self.MAX_STEPS
        done       = terminated or truncated
        self._done = done
        return {
            'state':       obs_arr,
            'reward':      reward,
            'is_first':    False,
            'is_last':     done,
            'is_terminal': terminated,
        }

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self._make_traj_params()
        self._step_cnt      = 0
        self._done          = False
        self._last_throttle = np.zeros(4, dtype=np.float32)
        init_pos = self._get_ref_pos(0)
        roll  = np.random.uniform(-0.2, 0.2) * math.pi
        pitch = np.random.uniform(-0.2, 0.2) * math.pi
        yaw   = np.random.uniform(0.0, 2.0 * math.pi)
        quat  = _euler_to_quat_wxyz(roll, pitch, yaw)
        self.data.qpos[:3]  = init_pos
        self.data.qpos[3:7] = quat
        self.data.qvel[:]   = 0.0
        mujoco.mj_forward(self.model, self.data)
        obs_arr, _, _ = self._get_obs()
        return {
            'state':       obs_arr,
            'reward':      np.float32(0.0),
            'is_first':    True,
            'is_last':     False,
            'is_terminal': False,
        }

    def traj_viz_data(self, n_pts=100):
        """Return (traj_points [n,3], current_target [3]) for visualization."""
        pts = []
        for i in range(n_pts):
            raw_t = self._traj_w * (i * self.MAX_STEPS / n_pts) / 60.0
            t = self.TRAJ_T0 + _scale_time(raw_t)
            pts.append(self._lemniscate_world(t))
        current_target = self._get_ref_pos(0)
        return np.array(pts, dtype=np.float32), current_target


# ---------------------------------------------------------------------------
# Forest (obstacle-avoidance navigation — OmniDrones Forest-style)
# ---------------------------------------------------------------------------

_DRONE_MASS         = 0.716   # kg (from XML inertial)
_HOVER_THROTTLE_01  = math.sqrt(_DRONE_MASS * 9.81 / (4.0 * _KF)) / _MAX_RPM


class MuJoCoForestDrone(_DroneBase):
    """
    OmniDrones Forest ported to MuJoCo.

    obs  : 167-dim
             [0:23]   state: rpos(3)+quat_xyzw(4)+lin_vel(3)+ang_vel(3)+heading(3)+up(3)+throttle(4)
             [23:167] lidar: 144 rays (36H×4V), range-dist convention (0=clear, 4=touching)
    act  : 4 normalized throttle commands in [-1, 1]
    Navigation: start y=-24 → target y=+24 (48m), random x per episode
    """

    STATE_DIM   = 23    # rpos(3)+quat_xyzw(4)+lin_vel(3)+ang_vel(3)+heading(3)+up(3)+throttle(4)
    OBS_DIM     = 167   # kept for PPO flat-obs compatibility
    MAX_STEPS   = 800
    N_TREES     = 1000
    TREE_H      = 4.0
    LIDAR_RANGE = 4.0
    N_H_RAYS    = 36
    N_V_RAYS    = 4
    N_RAYS      = 144     # 36 × 4
    MIN_DIST    = 0.3     # obstacle collision threshold
    MIN_Z       = 0.2
    MAX_Z       = 4.0
    MAX_SPEED   = 2.5

    START_Y     = -24.0
    TARGET_Y    = 24.0
    VEL_CLIP         = 2.0   # r_vel upper bound
    SAFETY_WEIGHT    = 0.2   # weight on mean log obstacle distance
    START_X_RANGE = 16.0  # x ∈ [-16, 16]
    EXCLUSION_R   = 1.5
    PLACE_X     = 16.0   # tree placement x ∈ [-16, 16]

    def __init__(self, task='forest', gui=False):
        self._init_mujoco_forest()
        self._gui            = gui
        self._step_cnt       = 0
        self._done           = True
        self._last_throttle  = np.full(4, _HOVER_THROTTLE_01, dtype=np.float32)
        self._target_pos     = np.array([0.0, self.TARGET_Y, 2.0], dtype=np.float32)
        self._tree_positions = np.zeros((self.N_TREES, 2), dtype=np.float32)
        self._tree_radii     = np.full(self.N_TREES, 0.3, dtype=np.float32)

    def _init_mujoco_forest(self):
        self.model = mujoco.MjModel.from_xml_path(_FOREST_XML)
        self.data  = mujoco.MjData(self.model)
        self._drone_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'drone')
        self._rotor_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'rotor{i}')
            for i in range(4)
        ]
        self._tree_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'tree{i}')
            for i in range(self.N_TREES)
        ]
        self._tree_geom_ids = [
            self.model.body_geomadr[bid]
            for bid in self._tree_body_ids
        ]
        # Precompute 144 LiDAR ray directions (float64 for mj_ray)
        v_degs = [-10.0, 0.0, 10.0, 20.0]
        h_angles = np.linspace(0.0, 2.0 * math.pi, self.N_H_RAYS, endpoint=False)
        rays = []
        for v in v_degs:
            v_rad = math.radians(v)
            cv, sv = math.cos(v_rad), math.sin(v_rad)
            for h in h_angles:
                rays.append([cv * math.cos(h), cv * math.sin(h), sv])
        self._lidar_dirs = np.array(rays, dtype=np.float64)  # (144, 3)
        self._geomid = np.array([-1], dtype=np.int32)
        self._last_lidar_scan = np.zeros(self.N_RAYS, dtype=np.float32)

    # ---- spaces ---------------------------------------------------------------

    @functools.cached_property
    def obs_space(self):
        return {
            'state':        elements.Space(np.float32, (self.STATE_DIM + self.N_RAYS,)),
            'reward':       elements.Space(np.float32),
            'is_first':     elements.Space(bool),
            'is_last':      elements.Space(bool),
            'is_terminal':  elements.Space(bool),
            'log/r_vel':    elements.Space(np.float32),
            'log/r_up':     elements.Space(np.float32),
            'log/r_safety': elements.Space(np.float32),
        }

    # ---- tree placement -------------------------------------------------------

    def _create_trees(self):
        """Randomly place N_TREES across the navigation area.

        Height mode (OmniDrones-style):
          full (25%):      z=2.0, half_h=2.0 → 4m tall, blocks drone
          half (25%):      z=1.0, half_h=1.0 → 2m tall, marginal
          invisible (50%): z=-10, half_h=0.1 → below ground, passable
        """
        _HEIGHT_MODES = [
            (2.0, 2.0),   # full
            (1.0, 1.0),   # half
            (-10., 0.1),  # invisible
            (-10., 0.1),  # invisible
        ]
        start_xy  = np.array([self.data.qpos[0], self.START_Y])
        target_xy = np.array([self.data.qpos[0], self.TARGET_Y])

        placed, tries = 0, 0
        while placed < self.N_TREES and tries < self.N_TREES * 20:
            tries += 1
            x = np.random.uniform(-self.PLACE_X, self.PLACE_X)
            y = np.random.uniform(self.START_Y,   self.TARGET_Y)
            if (math.hypot(x - start_xy[0],  y - start_xy[1])  < self.EXCLUSION_R or
                math.hypot(x - target_xy[0], y - target_xy[1]) < self.EXCLUSION_R):
                continue
            r       = np.random.uniform(0.2, 0.4)
            z_c, h  = _HEIGHT_MODES[np.random.randint(4)]
            bid = self._tree_body_ids[placed]
            gid = self._tree_geom_ids[placed]
            self.model.body_pos[bid]      = [x, y, z_c]
            self.model.geom_size[gid, 0]  = r
            self.model.geom_size[gid, 1]  = h
            self._tree_positions[placed]  = [x, y]
            self._tree_radii[placed]      = r if z_c > 0 else 0.0
            placed += 1
        mujoco.mj_forward(self.model, self.data)

    # ---- LiDAR ----------------------------------------------------------------

    def _scan_lidar(self) -> np.ndarray:
        """Fire 144 rays in body frame; return lidar_scan (range-dist convention)."""
        pos = self.data.qpos[:3].astype(np.float64)
        R   = self.data.xmat[self._drone_body_id].reshape(3, 3)  # body → world
        scan = np.empty(self.N_RAYS, dtype=np.float32)
        for i, ray_dir_body in enumerate(self._lidar_dirs):
            ray_dir_world = R @ ray_dir_body   # rotate to world frame
            dist = mujoco.mj_ray(
                self.model, self.data,
                pos, ray_dir_world,
                None,                       # geomgroup filter (None = all)
                1,                          # flg_static: include static geoms
                self._drone_body_id,        # bodyexclude: skip drone self-hits
                self._geomid,
            )
            if dist < 0.0 or dist > self.LIDAR_RANGE:
                dist = self.LIDAR_RANGE
            scan[i] = self.LIDAR_RANGE - dist  # 0=clear, LIDAR_RANGE=touching
        return scan

    # ---- obs ------------------------------------------------------------------

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        pos      = qpos[0:3].astype(np.float32)
        quat_w   = qpos[3:7]
        lin_vel  = qvel[0:3].astype(np.float32)
        ang_vel  = qvel[3:6].astype(np.float32)
        heading, up = _quat_wxyz_to_axes(quat_w)
        quat_xyzw   = np.array([quat_w[1], quat_w[2], quat_w[3], quat_w[0]],
                                dtype=np.float32)
        rpos_raw     = (self._target_pos - pos).astype(np.float32)
        rpos         = rpos_raw / (np.linalg.norm(rpos_raw) + 1e-6)
        throttle_obs = (self._last_throttle * 2.0 - 1.0).astype(np.float32)
        state_obs = np.concatenate([
            rpos, quat_xyzw, lin_vel, ang_vel, heading, up, throttle_obs
        ]).astype(np.float32)
        lidar_scan = self._scan_lidar()
        self._last_lidar_scan = lidar_scan
        full_obs = np.concatenate([state_obs, lidar_scan]).astype(np.float32)
        return full_obs, pos, lin_vel, up, ang_vel, lidar_scan

    # ---- reward ---------------------------------------------------------------

    def _compute_reward(self, lin_vel, up, lidar_scan):
        diff          = self._target_pos - self.data.qpos[:3].astype(np.float32)
        dist          = float(np.linalg.norm(diff))
        vel_direction = diff / max(dist, 1e-6)
        r_vel    = float(np.clip(np.dot(lin_vel, vel_direction), a_min=None, a_max=self.VEL_CLIP))
        r_up     = float(((up[2] + 1.0) / 2.0) ** 2)
        actual_dists = self.LIDAR_RANGE - lidar_scan
        r_safety = self.SAFETY_WEIGHT * float(np.mean(np.log(np.maximum(actual_dists, 1e-6))))
        reward   = r_vel + r_up + 1.0 + r_safety
        return np.float32(reward), np.float32(r_vel), np.float32(r_up), np.float32(r_safety)

    # ---- step / reset ---------------------------------------------------------

    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()
        act         = np.clip(action['action'], -1.0, 1.0).astype(np.float32)
        throttle_01 = ((act + 1.0) / 2.0).astype(np.float64)
        self._apply_rotor_forces(throttle_01)
        mujoco.mj_step(self.model, self.data)
        self._step_cnt      += 1
        self._last_throttle  = throttle_01.astype(np.float32)
        full_obs, pos, lin_vel, up, ang_vel, lidar_scan = self._get_obs()
        reward, r_vel, r_up, r_safety = self._compute_reward(lin_vel, up, lidar_scan)
        speed        = float(np.linalg.norm(lin_vel))
        min_obs_dist = float(self.LIDAR_RANGE - np.max(lidar_scan))
        term_reason = None
        if pos[2] < self.MIN_Z:            term_reason = f'too_low z={pos[2]:.2f}'
        elif pos[2] > self.MAX_Z:          term_reason = f'too_high z={pos[2]:.2f}'
        elif speed > self.MAX_SPEED:       term_reason = f'overspeed v={speed:.2f}'
        elif min_obs_dist < self.MIN_DIST: term_reason = f'collision dist={min_obs_dist:.2f}'
        terminated = term_reason is not None
        if terminated:
            print(f'[forest] terminated: {term_reason}  step={self._step_cnt}')
        truncated  = self._step_cnt >= self.MAX_STEPS
        done       = terminated or truncated
        self._done = done
        return {
            'state':        full_obs,
            'reward':       reward,
            'is_first':     False,
            'is_last':      done,
            'is_terminal':  terminated,
            'log/r_vel':    r_vel,
            'log/r_up':     r_up,
            'log/r_safety': r_safety,
        }

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        start_x = np.random.uniform(-self.START_X_RANGE, self.START_X_RANGE)
        self.data.qpos[:3]  = [start_x, self.START_Y, 2.0]
        roll  = np.random.uniform(-0.2, 0.2) * math.pi
        pitch = np.random.uniform(-0.2, 0.2) * math.pi
        yaw   = np.random.uniform(0.0, 2.0 * math.pi)
        self.data.qpos[3:7] = _euler_to_quat_wxyz(roll, pitch, yaw)
        self.data.qvel[:]   = 0.0
        self._target_pos    = np.array([start_x, self.TARGET_Y, 2.0], dtype=np.float32)
        self._step_cnt      = 0
        self._done          = False
        self._last_throttle = np.full(4, _HOVER_THROTTLE_01, dtype=np.float32)
        self._create_trees()   # updates model.body_pos + calls mj_forward
        full_obs, _, _, _, _, _ = self._get_obs()
        return {
            'state':        full_obs,
            'reward':       np.float32(0.0),
            'is_first':     True,
            'is_last':      False,
            'is_terminal':  False,
            'log/r_vel':    np.float32(0.0),
            'log/r_up':     np.float32(0.0),
            'log/r_safety': np.float32(0.0),
        }

    def traj_viz_data(self):
        """Return (tree_positions (N,2), target_pos (3,), tree_radii (N,), tree_z (N,), tree_half_h (N,)) for GUI visualization."""
        z_centers   = np.array([self.model.body_pos[bid][2] for bid in self._tree_body_ids], dtype=np.float32)
        half_heights = np.array([self.model.geom_size[self.model.body_geomadr[bid], 1] for bid in self._tree_body_ids], dtype=np.float32)
        return self._tree_positions.copy(), self._target_pos.copy(), self._tree_radii.copy(), z_centers, half_heights

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Forest2 — Forest with goal bonus and no speed-limit termination
# ---------------------------------------------------------------------------

class MuJoCoForest2Drone(MuJoCoForestDrone):
    """
    Forest variant (v2):
      - No speed-limit termination
      - Goal bonus: entering GOAL_RADIUS of target gives
        (remaining_steps × GOAL_REWARD_PER_STEP) as one-time bonus reward.
    """
    MAX_STEPS            = 800
    GOAL_RADIUS          = 2.0
    GOAL_REWARD_PER_STEP = 1.0   # 남은 스텝 수만큼 보너스

    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()
        act         = np.clip(action['action'], -1.0, 1.0).astype(np.float32)
        throttle_01 = ((act + 1.0) / 2.0).astype(np.float64)
        self._apply_rotor_forces(throttle_01)
        mujoco.mj_step(self.model, self.data)
        self._step_cnt      += 1
        self._last_throttle  = throttle_01.astype(np.float32)
        full_obs, pos, lin_vel, up, ang_vel, lidar_scan = self._get_obs()
        _, _r_vel_old, r_up, r_safety = self._compute_reward(lin_vel, up, lidar_scan)
        diff    = self._target_pos - self.data.qpos[:3].astype(np.float32)
        dist_3d = float(np.linalg.norm(diff))
        vel_dir = diff / max(dist_3d, 1e-6)
        v       = float(np.dot(lin_vel, vel_dir))
        r_vel   = np.float32(min(v, self.VEL_CLIP))
        reward  = np.float32(float(r_vel) + float(r_up) + 1.0 + float(r_safety))

        # Goal detection
        dist_to_goal = float(np.linalg.norm(
            self.data.qpos[:3].astype(np.float32) - self._target_pos))
        goal_reached = dist_to_goal < self.GOAL_RADIUS
        if goal_reached:
            # remaining = self.MAX_STEPS - self._step_cnt
            # bonus     = remaining * self.GOAL_REWARD_PER_STEP
            # reward    = np.float32(float(reward) + bonus)
            print(f'[forest] GOAL! step={self._step_cnt}')

        min_obs_dist = float(self.LIDAR_RANGE - np.max(lidar_scan))
        term_reason  = None
        if   pos[2] < self.MIN_Z:           term_reason = f'too_low z={pos[2]:.2f}'
        elif pos[2] > self.MAX_Z:           term_reason = f'too_high z={pos[2]:.2f}'
        elif min_obs_dist < self.MIN_DIST:  term_reason = f'collision dist={min_obs_dist:.2f}'
        elif goal_reached:                  term_reason = 'goal_reached'
        terminated = term_reason is not None
        if terminated and term_reason != 'goal_reached':
            print(f'[forest] terminated: {term_reason}  step={self._step_cnt}')
        truncated  = self._step_cnt >= self.MAX_STEPS
        done       = terminated or truncated
        self._done = done
        return {
            'state':        full_obs,
            'reward':       reward,
            'is_first':     False,
            'is_last':      done,
            'is_terminal':  terminated,
            'log/r_vel':    r_vel,
            'log/r_up':     r_up,
            'log/r_safety': r_safety,
        }


# ---------------------------------------------------------------------------
# Register tasks
# ---------------------------------------------------------------------------

MujocoDrone._TASK_MAP = {
    'hover':   MuJoCoHoverDrone,
    'track':   MuJoCoTrackDrone,
    'forest':  MuJoCoForest2Drone,
}
