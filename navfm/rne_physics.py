"""Run navfm episodes on Robot Native Engine physics (Rapier, joint-driven differential drive).

The same layout is exported as a `.rne.scene.toml`: walls, furniture and target objects become
fixed box obstacles, and a small differential-drive robot (wheel radius 0.06 m, track 0.28 m)
spawns at the episode start. Each 0.25 s control step turns the agent's (v, omega) into wheel
velocity targets and advances RNE by 15 physics ticks (60 Hz), so wheel motors, inertia, slip
and contacts are RNE's. Observations are the same panoramas as in the Python simulator,
taken at RNE's robot pose.

Requires the `rne_py` wheel built from RobotNativeEngine (`maturin build -m crates/rne_py/Cargo.toml`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import world as W
from .rne_scene import _rects

WHEEL_R_M = 0.06
TRACK_M = 0.28
BODY_HALF = (0.15, 0.08, 0.12)  # x (forward), y (up), z (side)
SPAWN_H_M = 0.14
HEIGHT_M = {W.WALL: 1.0, W.LOW: 0.42}
OBJECT_H_M = 0.8

ROBOT_TOML = f"""kind = "diff_drive"
model_name = "navfm_bot"

[diff_drive]
wheel_radius_m = {WHEEL_R_M}
track_width_m = {TRACK_M}
base_half_extents_m = [{BODY_HALF[0]}, {BODY_HALF[1]}, {BODY_HALF[2]}]
max_wheel_velocity_rad_s = 20.0
initial_translation_m = [{{x}}, {SPAWN_H_M}, {{z}}]
drive_mode = "joint_driven"
"""


def scene_toml(lay: W.Layout, seed: int) -> str:
    out = [
        "[world]\ngravity_m_s2 = [0.0, -9.81, 0.0]",
        f"seed = {seed}\n",
        "[ground]\nenabled = true\n",
        '[[robots]]\npath = "robot.rne.robot.toml"\n',
    ]
    res = W.RES_M
    n = 0
    for lab in np.unique(lay.grid):
        if lab == W.FREE:
            continue
        h = HEIGHT_M.get(int(lab), OBJECT_H_M)
        for i0, j0, i1, j1 in _rects(lay.grid == lab):
            sx, sy = (i1 - i0 + 1) * res, (j1 - j0 + 1) * res
            cx, cy = i0 * res + sx / 2, j0 * res + sy / 2
            out.append(
                f'[[obstacles]]\nname = "o{n}"\n'
                f"translation_m = [{cx - W.WIDTH_M / 2:.4f}, {h / 2:.3f}, {W.HEIGHT_M / 2 - cy:.4f}]\n"
                f"half_extents_m = [{sx / 2:.4f}, {h / 2:.3f}, {sy / 2:.4f}]\n"
                'body_type = "fixed"\n'
            )
            n += 1
    return "\n".join(out)


class RneNavEnv(W.NavEnv):
    """NavEnv whose robot motion is simulated by RNE (Rapier) instead of the unicycle model."""

    TICKS = 15  # 0.25 s at 60 Hz
    SPIN_TICKS_MAX = 600

    def __init__(self, seed: int, kind: str = "apartment"):
        import rne_py

        super().__init__(seed, kind)
        self._dir = Path(tempfile.mkdtemp(prefix="rne_nav_"))
        x0, y0, th0 = self.pose
        (self._dir / "robot.rne.robot.toml").write_text(
            ROBOT_TOML.format(x=f"{x0 - W.WIDTH_M / 2:.4f}", z=f"{W.HEIGHT_M / 2 - y0:.4f}")
        )
        (self._dir / "scene.rne.scene.toml").write_text(scene_toml(self.layout, seed))
        self.sim = rne_py.DiffDriveSim.from_scene(str(self._dir / "scene.rne.scene.toml"))
        self._obs = self.sim.reset()
        assert abs(self.sim.fixed_delta_s * self.TICKS - W.DT_S) < 1e-3
        # the asset format has no initial yaw: spin in place to the start heading (not counted)
        for _ in range(self.SPIN_TICKS_MAX):
            err = (th0 - self._obs.base_yaw + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < np.deg2rad(1.0):
                break
            w = float(np.clip(3.0 * err, -1.6, 1.6))
            self._obs = self.sim.step(*self._wheels(0.0, w))
        for _ in range(30):  # settle
            self._obs = self.sim.step(0.0, 0.0)
        self.pose = self._pose()
        self.path = [self.pose.copy()]

    @staticmethod
    def _wheels(v, w):
        return (v - w * TRACK_M / 2) / WHEEL_R_M, (v + w * TRACK_M / 2) / WHEEL_R_M

    def _pose(self):
        o = self._obs
        return np.array([o.base_x + W.WIDTH_M / 2, W.HEIGHT_M / 2 - o.base_z, (o.base_yaw + np.pi) % (2 * np.pi) - np.pi])

    def step(self, action):
        v = float(np.clip(action[0], *W.V_RANGE))
        w = float(np.clip(action[1], *W.W_RANGE))
        before = self.pose.copy()
        wl, wr = self._wheels(v, w)
        for _ in range(self.TICKS):
            self._obs = self.sim.step(wl, wr)
        self.pose = self._pose()
        moved = float(np.hypot(*(self.pose[:2] - before[:2])))
        collided = abs(v) > 0.1 and moved < 0.3 * abs(v) * W.DT_S
        self.collisions += int(collided)
        self.t += 1
        self.path.append(self.pose.copy())
        if self.target_distance() <= W.SUCCESS_DIST_M:
            self.done = self.success = True
        elif self.t >= self.max_steps:
            self.done = True
        return self.observe(), collided
