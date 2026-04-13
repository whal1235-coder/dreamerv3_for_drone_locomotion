"""
Render environment snapshots to assets/ for README.

Usage:
  python snapshot.py
"""

import os
import pathlib
import sys
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import mujoco
from PIL import Image
from embodied.envs.mujoco_drone import MuJoCoHoverDrone, MuJoCoTrackDrone, MuJoCoForestDrone

_OUT_DIR = pathlib.Path(__file__).parent / 'assets'
_OUT_DIR.mkdir(exist_ok=True)

W, H = 640, 480


def render_snapshot(env, out_path, camera='overview'):
    # trigger proper reset (places drone, trees, etc.)
    env.step({'reset': True, 'action': np.zeros(4, dtype=np.float32)})

    renderer = mujoco.Renderer(env.model, height=H, width=W)
    renderer.update_scene(env.data, camera=camera)
    pixels = renderer.render()
    renderer.close()

    Image.fromarray(pixels).save(out_path)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    render_snapshot(MuJoCoHoverDrone(),  _OUT_DIR / 'snapshot_hover.png')
    render_snapshot(MuJoCoTrackDrone(),  _OUT_DIR / 'snapshot_track.png')
    render_snapshot(MuJoCoForestDrone(), _OUT_DIR / 'snapshot_forest.png')
