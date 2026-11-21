"""A small, deterministic 2D apartment simulator with a ray-cast egocentric camera.

The world is an occupancy grid with semantic labels.  Tall cells (walls and the
target objects) block line of sight; low cells (tables, shelves) only block
motion, so the robot can see objects across a cluttered room.  Observations are
a 360-degree panoramic strip (colour + two depth layers) and a pseudo-3D
first-person view used for the hero GIFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

RES_M = 0.05  # metres per grid cell
WIDTH_M, HEIGHT_M = 8.0, 6.0
NX, NY = int(WIDTH_M / RES_M), int(HEIGHT_M / RES_M)
ROBOT_RADIUS_M = 0.18
DT_S = 0.25
V_RANGE = (-0.2, 0.6)
W_RANGE = (-1.6, 1.6)
SUCCESS_DIST_M = 0.5
MAX_STEPS = 160
MAX_RANGE_M = 8.0
PANO_RAYS = 128

FREE, WALL, LOW = 0, 1, 2
OBJ0 = 3  # first object label

# name, colour word, RGB, footprint (w, h) metres
OBJECTS = [
    ("chair", "red", (220, 50, 47), (0.45, 0.45)),
    ("sofa", "blue", (38, 110, 220), (1.1, 0.5)),
    ("plant", "green", (60, 170, 70), (0.35, 0.35)),
    ("fridge", "yellow", (240, 200, 40), (0.6, 0.55)),
    ("bed", "purple", (150, 80, 200), (0.9, 0.9)),
    ("tv", "cyan", (40, 200, 210), (0.8, 0.25)),
]
N_OBJ = len(OBJECTS)
WALL_RGB = (175, 175, 185)
LOW_RGB = (140, 100, 70)
LABEL_RGB = np.array([(0, 0, 0), WALL_RGB, LOW_RGB] + [o[2] for o in OBJECTS], dtype=np.float32)

TEMPLATES = [
    "go to the {c} {o}",
    "find the {o}",
    "navigate to the {c} {o}",
    "head over to the {o}",
    "take me to the {c} {o}",
    "where is the {c} {o}",
]


def instruction_text(obj: int, template: int) -> str:
    name, colour, _, _ = OBJECTS[obj]
    return TEMPLATES[template].format(c=colour, o=name)


@dataclass
class Layout:
    """Static scene of one episode."""

    grid: np.ndarray  # (NX, NY) uint8 labels
    obj_boxes: list  # [(x0, y0, x1, y1)] metres, index = object id
    cspace: np.ndarray = field(init=False)  # bool: robot centre may NOT be here (collision)
    plan_cspace: np.ndarray = field(init=False)  # conservative inflation used for planning
    tall: np.ndarray = field(init=False)
    blocked: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.blocked = self.grid != FREE
        self.tall = (self.grid == WALL) | (self.grid >= OBJ0)
        dist = ndimage.distance_transform_edt(~self.blocked) * RES_M
        self.clearance = dist
        self.cspace = dist < ROBOT_RADIUS_M
        self.plan_cspace = dist < ROBOT_RADIUS_M + 0.1


def to_cell(x: np.ndarray | float, y: np.ndarray | float):
    return (np.asarray(x) / RES_M).astype(int), (np.asarray(y) / RES_M).astype(int)


def _fill(grid, x0, y0, x1, y1, label):
    i0, j0 = to_cell(x0, y0)
    i1, j1 = to_cell(x1, y1)
    grid[max(i0, 0) : min(i1 + 1, NX), max(j0, 0) : min(j1 + 1, NY)] = label


def make_layout(rng: np.random.Generator) -> Layout:
    """Random open-plan apartment: outer walls, a partial divider, low furniture, six objects."""
    for _ in range(200):
        grid = np.zeros((NX, NY), np.uint8)
        t = 0.1
        _fill(grid, 0, 0, WIDTH_M, t, WALL)
        _fill(grid, 0, HEIGHT_M - t, WIDTH_M, HEIGHT_M, WALL)
        _fill(grid, 0, 0, t, HEIGHT_M, WALL)
        _fill(grid, WIDTH_M - t, 0, WIDTH_M, HEIGHT_M, WALL)
        # partial divider walls (tall, occlude)
        dx = rng.uniform(3.2, 4.8)
        if rng.random() < 0.5:
            _fill(grid, dx, 0, dx + 0.1, rng.uniform(2.6, 3.6), WALL)
        else:
            _fill(grid, dx, rng.uniform(2.4, 3.4), dx + 0.1, HEIGHT_M, WALL)
        if rng.random() < 0.6:
            wy = rng.uniform(2.0, 4.0)
            x0 = rng.uniform(0.0, 1.0) if rng.random() < 0.5 else rng.uniform(5.5, 6.5)
            _fill(grid, x0, wy, x0 + rng.uniform(1.0, 1.6), wy + 0.1, WALL)
        # low furniture
        boxes = []
        for _k in range(rng.integers(4, 7)):
            w, h = rng.uniform(0.4, 1.2), rng.uniform(0.4, 0.9)
            x, y = rng.uniform(0.6, WIDTH_M - 0.6 - w), rng.uniform(0.6, HEIGHT_M - 0.6 - h)
            if _clear(grid, x - 0.5, y - 0.5, x + w + 0.5, y + h + 0.5):
                _fill(grid, x, y, x + w, y + h, LOW)
        # objects (each mostly against free space, keep 0.7 m apart)
        obj_boxes = []
        ok = True
        for k, (_n, _c, _rgb, (w, h)) in enumerate(OBJECTS):
            if rng.random() < 0.5:
                w, h = h, w
            for _try in range(100):
                x, y = rng.uniform(0.25, WIDTH_M - 0.25 - w), rng.uniform(0.25, HEIGHT_M - 0.25 - h)
                if _clear(grid, x - 0.6, y - 0.6, x + w + 0.6, y + h + 0.6, allow_wall=True) and _clear(
                    grid, x - 0.15, y - 0.15, x + w + 0.15, y + h + 0.15
                ):
                    _fill(grid, x, y, x + w, y + h, OBJ0 + k)
                    obj_boxes.append((x, y, x + w, y + h))
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        lay = Layout(grid, obj_boxes)
        # all objects must be reachable from one connected free component
        lab, n = ndimage.label(~lay.plan_cspace)
        if n == 0:
            continue
        main = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        reach = lab == main
        if all(_goal_region(lay, k, reach).any() for k in range(N_OBJ)):
            lay.reach = reach
            return lay
    raise RuntimeError("could not sample a layout")


def _clear(grid, x0, y0, x1, y1, allow_wall=False):
    i0, j0 = to_cell(max(x0, 0), max(y0, 0))
    i1, j1 = to_cell(min(x1, WIDTH_M - 1e-6), min(y1, HEIGHT_M - 1e-6))
    sub = grid[i0 : i1 + 1, j0 : j1 + 1]
    if allow_wall:
        return not ((sub != FREE) & (sub != WALL)).any()
    return not (sub != FREE).any()


_XS = (np.arange(NX) + 0.5) * RES_M
_YS = (np.arange(NY) + 0.5) * RES_M


def box_distance(box, x, y):
    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - x, 0), x - x1)
    dy = np.maximum(np.maximum(y0 - y, 0), y - y1)
    return np.hypot(dx, dy)


def _goal_region(lay: Layout, k: int, reach: np.ndarray) -> np.ndarray:
    d = box_distance(lay.obj_boxes[k], _XS[:, None], _YS[None, :])
    return (d <= SUCCESS_DIST_M - 0.08) & reach


def raycast(lay: Layout, x: float, y: float, angles: np.ndarray, step_m: float = 0.025):
    """March rays; returns tall depth, tall label, low depth (first hit of anything)."""
    n_steps = int(MAX_RANGE_M / step_m)
    ts = (np.arange(1, n_steps + 1) * step_m)[None, :]
    px = x + np.cos(angles)[:, None] * ts
    py = y + np.sin(angles)[:, None] * ts
    i = np.clip((px / RES_M).astype(int), 0, NX - 1)
    j = np.clip((py / RES_M).astype(int), 0, NY - 1)
    labels = lay.grid[i, j]
    tall = (labels == WALL) | (labels >= OBJ0)
    anyhit = labels != FREE
    ti = np.where(tall.any(1), tall.argmax(1), n_steps - 1)
    ai = np.where(anyhit.any(1), anyhit.argmax(1), n_steps - 1)
    rows = np.arange(len(angles))
    return ts[0, ti], labels[rows, ti], ts[0, ai]


@dataclass
class Obs:
    pano_rgb: np.ndarray  # (PANO_RAYS, 3) float 0..1, column 0 = behind-left, centre = forward
    pano_depth: np.ndarray  # (PANO_RAYS,) metres, tall layer
    pano_low: np.ndarray  # (PANO_RAYS,) metres, first hit of anything
    pano_label: np.ndarray  # (PANO_RAYS,) int (privileged; used only for supervision)
    pose: np.ndarray  # (3,) x, y, theta (odometry)
    instruction: str
    target: int  # privileged


PANO_ANGLES = np.linspace(-np.pi, np.pi, PANO_RAYS, endpoint=False) + np.pi / PANO_RAYS


def pano_tensor(obs: Obs) -> np.ndarray:
    """(5, PANO_RAYS) float32 network input: shaded RGB + two inverse-ish depths."""
    shade = 1.0 / (1.0 + 0.12 * obs.pano_depth)
    rgb = obs.pano_rgb.T * shade[None]
    return np.concatenate(
        [rgb, (obs.pano_depth / MAX_RANGE_M)[None], np.minimum(obs.pano_low, 3.0)[None] / 3.0], 0
    ).astype(np.float32)


class NavEnv:
    """Object-goal navigation episode: reach within 0.5 m of the instructed object.

    A blocked move slides along the obstacle (Habitat-style) and counts as a collision.
    """

    def __init__(self, seed: int):
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.layout = make_layout(rng)
        self.target = int(rng.integers(N_OBJ))
        self.template = int(rng.integers(len(TEMPLATES)))
        self.instruction = instruction_text(self.target, self.template)
        from .expert import geodesic_field

        self.field = geodesic_field(self.layout, self.target)
        # start: reachable, at least 3 m geodesic from goal
        cand = np.argwhere(self.layout.reach & (self.field > 3.0) & np.isfinite(self.field))
        if len(cand) == 0:
            cand = np.argwhere(self.layout.reach & np.isfinite(self.field) & (self.field > 1.0))
        c = cand[rng.integers(len(cand))]
        self.pose = np.array([(c[0] + 0.5) * RES_M, (c[1] + 0.5) * RES_M, rng.uniform(-np.pi, np.pi)])
        self.t = 0
        self.collisions = 0
        self.path = [self.pose.copy()]
        self.done = False
        self.success = False

    def geodesic(self, x: float, y: float) -> float:
        i, j = to_cell(np.clip(x, 0, WIDTH_M - 1e-6), np.clip(y, 0, HEIGHT_M - 1e-6))
        return float(self.field[i, j])

    def target_distance(self) -> float:
        return float(box_distance(self.layout.obj_boxes[self.target], self.pose[0], self.pose[1]))

    def observe(self) -> Obs:
        x, y, th = self.pose
        d, lab, low = raycast(self.layout, x, y, th + PANO_ANGLES)
        return Obs(LABEL_RGB[lab] / 255.0, d, low, lab, self.pose.copy(), self.instruction, self.target)

    def free(self, x: float, y: float) -> bool:
        if not (0 < x < WIDTH_M and 0 < y < HEIGHT_M):
            return False
        i, j = to_cell(x, y)
        return not self.layout.cspace[i, j]

    def step(self, action) -> tuple[Obs, bool]:
        v = float(np.clip(action[0], *V_RANGE))
        w = float(np.clip(action[1], *W_RANGE))
        x, y, th = self.pose
        nx, ny, nth = unicycle(x, y, th, v, w)
        collided = False
        if self.free(nx, ny):
            self.pose = np.array([nx, ny, nth])
        else:
            # slide along the obstacle (like Habitat's allow_sliding): keep the larger free axis component
            collided = True
            self.collisions += 1
            cands = [(abs(nx - x), nx, y), (abs(ny - y), x, ny)]
            cands = [c for c in sorted(cands, reverse=True) if self.free(c[1], c[2])]
            sx, sy = (cands[0][1], cands[0][2]) if cands else (x, y)
            self.pose = np.array([sx, sy, nth])
        self.t += 1
        self.path.append(self.pose.copy())
        if self.target_distance() <= SUCCESS_DIST_M:
            self.done = self.success = True
        elif self.t >= MAX_STEPS:
            self.done = True
        return self.observe(), collided


def unicycle(x, y, th, v, w, dt=DT_S):
    if abs(w) < 1e-6:
        return x + v * dt * np.cos(th), y + v * dt * np.sin(th), th
    nth = th + w * dt
    return (
        x + v / w * (np.sin(nth) - np.sin(th)),
        y - v / w * (np.cos(nth) - np.cos(th)),
        (nth + np.pi) % (2 * np.pi) - np.pi,
    )
