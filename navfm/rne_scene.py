"""Export episodes to Robot Native Engine (RNE) replay scripts for wgpu-rendered GIF frames.

The simulator stays in Python; RNE's `150_scene_replay_render` example renders the scene. The
world is mapped to RNE's Y-up frame as X = x - 4, Y = height, Z = 3 - y (metres), so a
simulator heading theta becomes a rotation of theta about +Y.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from . import models as M
from . import world as W

RNE_BIN_ENV = "RNE_REPLAY_BIN"
DEFAULT_BIN = Path(__file__).resolve().parents[2] / "RobotNativeEngine/target/release/examples/150_scene_replay_render"

WALL_H = 1.0
LOW_H = 0.42
CAM_H = 0.55
OVERVIEW = dict(width=352, height=264)
FPV = dict(width=272, height=150)


def rne_binary() -> Path | None:
    p = Path(os.environ.get(RNE_BIN_ENV, DEFAULT_BIN))
    return p if p.exists() else None


# ---------------------------------------------------------------- coordinates
def P(x, y, h=0.0):
    return [float(x - W.WIDTH_M / 2), float(h), float(W.HEIGHT_M / 2 - y)]


def yaw_q(theta):
    return [0.0, float(np.sin(theta / 2)), 0.0, float(np.cos(theta / 2))]


Q_UP = [-float(np.sin(np.pi / 4)), 0.0, 0.0, float(np.cos(np.pi / 4))]  # cylinder Z -> Y


def look_rot(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.stack([r, u, -f], 1)  # columns: camera X, Y, Z in world
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = s / 4
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    return [float(v) for v in q]


def lin(rgb, a=1.0):
    """sRGB 0..255 -> linear RGBA."""
    c = (np.asarray(rgb, float) / 255.0) ** 2.2
    return [float(c[0]), float(c[1]), float(c[2]), a]


def box(cx, cy, h0, sx, sy, sh, rgb, theta=0.0, rough=0.7, emissive=None):
    it = dict(shape="box", size_m=[sx, sh, sy], pos_m=P(cx, cy, h0 + sh / 2), rot=yaw_q(theta),
              color_rgba=lin(rgb), roughness=rough)
    if emissive is not None:
        it["emissive_rgb"] = emissive
    return it


def sphere(x, y, h, r, rgb, emissive=None, rough=0.5):
    it = dict(shape="sphere", radius_m=r, pos_m=P(x, y, h), color_rgba=lin(rgb), roughness=rough)
    if emissive is not None:
        it["emissive_rgb"] = emissive
    return it


def cyl(x, y, h0, r, length, rgb, rough=0.6, emissive=None):
    it = dict(shape="cylinder", radius_m=r, length_m=length, pos_m=P(x, y, h0 + length / 2), rot=Q_UP,
              color_rgba=lin(rgb), roughness=rough)
    if emissive is not None:
        it["emissive_rgb"] = emissive
    return it


# ---------------------------------------------------------------- static scene
def _rects(mask):
    """Greedy cover of a boolean grid with axis-aligned rectangles (cells)."""
    m = mask.copy()
    out = []
    for i in range(m.shape[0]):
        j = 0
        while j < m.shape[1]:
            if not m[i, j]:
                j += 1
                continue
            j1 = j
            while j1 + 1 < m.shape[1] and m[i, j1 + 1]:
                j1 += 1
            i1 = i
            while i1 + 1 < m.shape[0] and m[i1 + 1, j : j1 + 1].all():
                i1 += 1
            m[i : i1 + 1, j : j1 + 1] = False
            out.append((i, j, i1, j1))
            j = j1 + 1
    return out


def _object_items(k, box_m):
    x0, y0, x1, y1 = box_m
    cx, cy, sx, sy = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    long_x = sx >= sy
    rgb = W.OBJECTS[k][2]
    dark = tuple(int(c * 0.6) for c in rgb)
    name = W.OBJECTS[k][0]
    it = []
    if name == "chair":
        it += [box(cx, cy, 0.0, sx, sy, 0.45, rgb)]
        bx, by = (cx, y1 - 0.04) if True else (cx, cy)
        it += [box(bx, by, 0.45, sx, 0.08, 0.45, dark)]
    elif name == "sofa":
        it += [box(cx, cy, 0.0, sx, sy, 0.38, rgb, rough=0.9)]
        if long_x:
            it += [box(cx, y1 - 0.07, 0.38, sx, 0.14, 0.32, dark, rough=0.9)]
            it += [box(x0 + 0.06, cy, 0.38, 0.12, sy, 0.16, dark), box(x1 - 0.06, cy, 0.38, 0.12, sy, 0.16, dark)]
        else:
            it += [box(x1 - 0.07, cy, 0.38, 0.14, sy, 0.32, dark, rough=0.9)]
            it += [box(cx, y0 + 0.06, 0.38, sx, 0.12, 0.16, dark), box(cx, y1 - 0.06, 0.38, sx, 0.12, 0.16, dark)]
    elif name == "plant":
        r = min(sx, sy) / 2
        it += [cyl(cx, cy, 0.0, r * 0.7, 0.35, (150, 90, 60))]
        it += [sphere(cx, cy, 0.62, r * 1.05, rgb, rough=0.9), sphere(cx + 0.05, cy - 0.04, 0.85, r * 0.75, rgb, rough=0.9)]
    elif name == "fridge":
        it += [box(cx, cy, 0.0, sx, sy, 1.25, rgb, rough=0.35)]
        it += [box(cx, cy, 0.78, sx + 0.01, sy + 0.01, 0.015, (60, 60, 60))]
    elif name == "bed":
        it += [box(cx, cy, 0.0, sx, sy, 0.3, (235, 230, 220), rough=0.9)]
        it += [box(cx, cy - sy * 0.1 if not long_x else cy, 0.3, sx * (0.75 if long_x else 1.0), sy * (1.0 if long_x else 0.75), 0.06, rgb, rough=0.95)]
        px, py = (x1 - 0.12, cy) if long_x else (cx, y1 - 0.12)
        it += [box(px, py, 0.3, 0.2 if long_x else sx * 0.6, sy * 0.6 if long_x else 0.2, 0.1, (250, 250, 250))]
    elif name == "tv":
        it += [box(cx, cy, 0.0, sx, sy, 0.45, (60, 55, 50))]
        it += [box(cx, cy, 0.45, sx * 0.95 if long_x else 0.05, 0.05 if long_x else sy * 0.95, 0.5, rgb, rough=0.2,
                   emissive=[0.05, 0.25, 0.28])]
    return it


def static_items(lay: W.Layout):
    items = [box(W.WIDTH_M / 2, W.HEIGHT_M / 2, -0.02, W.WIDTH_M + 0.4, W.HEIGHT_M + 0.4, 0.02, (120, 100, 85), rough=0.8)]
    # floor tiles for depth cues
    for i in range(16):
        for j in range(12):
            if (i + j) % 2 == 0:
                items.append(box(0.25 + 0.5 * i, 0.25 + 0.5 * j, -0.015, 0.5, 0.5, 0.017, (138, 116, 98), rough=0.8))
    res = W.RES_M
    for lab, h, rgb in ((W.WALL, WALL_H, (205, 205, 212)), (W.LOW, LOW_H, (140, 100, 70))):
        for i0, j0, i1, j1 in _rects(lay.grid == lab):
            sx, sy = (i1 - i0 + 1) * res, (j1 - j0 + 1) * res
            items.append(box(i0 * res + sx / 2, j0 * res + sy / 2, 0.0, sx, sy, h, rgb))
    for k, b in enumerate(lay.obj_boxes):
        items += _object_items(k, b)
    return items


# ---------------------------------------------------------------- dynamic items
def robot_items(pose, accent):
    x, y, th = pose
    it = [cyl(x, y, 0.03, W.ROBOT_RADIUS_M, 0.22, (240, 240, 245), rough=0.35)]
    it += [cyl(x, y, 0.25, W.ROBOT_RADIUS_M * 0.92, 0.03, accent, emissive=[c / 255 * 0.6 for c in accent])]
    it += [cyl(x, y, 0.28, 0.035, 0.24, (70, 70, 80))]
    cx, cy = x + np.cos(th) * 0.03, y + np.sin(th) * 0.03
    it += [box(cx, cy, 0.5, 0.1, 0.16, 0.08, (40, 40, 48), theta=th, rough=0.3)]
    lx, ly = x + np.cos(th) * 0.09, y + np.sin(th) * 0.09
    it += [sphere(lx, ly, 0.54, 0.025, (120, 220, 255), emissive=[0.4, 0.9, 1.2])]
    return it


def dots(points, h, r, rgb, glow, every=1):
    e = [c / 255 * glow for c in rgb]
    return [sphere(p[0], p[1], h, r, rgb, emissive=e) for p in points[::every]]


def integrate(pose, acts):
    x, y, th = pose
    pts = []
    for v, w in acts:
        x, y, th = W.unicycle(x, y, th, float(v), float(w))
        pts.append((x, y))
    return pts


def overlay_items(kind, viz, pose, path, target_box, accent):
    it = []
    # target halo on the floor
    x0, y0, x1, y1 = target_box
    it.append(cyl((x0 + x1) / 2, (y0 + y1) / 2, 0.0, max(x1 - x0, y1 - y0) / 2 + 0.3, 0.01, (200, 205, 215),
                  emissive=[0.08, 0.08, 0.09]))
    # travelled trajectory
    if len(path) > 1:
        it += dots([(p[0], p[1]) for p in path[:-1]], 0.03, 0.035, (255, 255, 255), 0.5)
    if kind == "vlm":
        pts = viz.get("path") or []
        it += dots(pts[::3], 0.06, 0.03, accent, 1.2)
        g = viz.get("goal")
        if g is not None:
            it.append(cyl(g[0], g[1], 0.0, 0.06, 0.9, accent, emissive=[c / 255 * 1.5 for c in accent]))
    elif kind == "wm" and "samples" in viz:
        cost = viz["sample_cost"]
        order = np.argsort(cost)[:14]
        for idx in order[1:]:
            it += dots(integrate(pose, viz["samples"][idx]), 0.05, 0.018, (150, 140, 170), 0.15, every=2)
        it += dots(integrate(pose, viz["best"]), 0.07, 0.035, accent, 1.4)
    elif kind == "wam" and "chunk" in viz:
        pts = integrate(viz["chunk_pose"], viz["chunk"])
        it += dots(pts, 0.06, 0.03, accent, 1.0)
        it += dots([pts[k - 1] for k in M.WAM.FUTURE], 0.06, 0.055, accent, 1.6)
    return it


# ---------------------------------------------------------------- views
def overview_view(out):
    eye = P(W.WIDTH_M / 2, -1.9, 7.2)
    target = P(W.WIDTH_M / 2, W.HEIGHT_M / 2 - 0.15, 0.0)
    return dict(out=out, fov_y_rad=0.78, eye_m=eye, rot=look_rot(eye, target), **OVERVIEW)


def fpv_view(pose, out):
    x, y, th = pose
    eye = P(x + np.cos(th) * 0.12, y + np.sin(th) * 0.12, CAM_H)
    target = P(x + np.cos(th) * 2.0, y + np.sin(th) * 2.0, CAM_H - 0.12)
    vfov = 2 * np.arctan(np.tan(np.deg2rad(50)) * FPV["height"] / FPV["width"])
    return dict(out=out, fov_y_rad=float(vfov), eye_m=eye, rot=look_rot(eye, target), **FPV)


def render_episode(lay, target_box, kind, accent, records, out_dir: Path):
    """records: list of dicts with pose, viz, path.  Writes ov_XXX.png / fpv_XXX.png in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, r in enumerate(records):
        robot = robot_items(r["pose"], accent)
        overlay = overlay_items(kind, r["viz"], r["pose"], r["path"], target_box, accent)
        frames.append(dict(items=robot + overlay, views=[overview_view(f"ov_{i:04d}.png")]))
        # the egocentric camera sees the world only (plans are drawn in the overview)
        frames.append(dict(items=robot, views=[fpv_view(r["pose"], f"fpv_{i:04d}.png")]))
    script = dict(clear_rgba=[0.07, 0.08, 0.1, 1.0], static_items=static_items(lay), frames=frames)
    path = out_dir / "script.json"
    path.write_text(json.dumps(script))
    subprocess.run([str(rne_binary()), str(path), str(out_dir)], check=True, stdout=subprocess.DEVNULL)
    return out_dir
