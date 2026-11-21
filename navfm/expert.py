"""Privileged expert: geodesic distance field + pure pursuit (used only to make training data)."""

from __future__ import annotations

import heapq

import numpy as np

from . import world as W

_NEI = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)] + [
    (a, b, 2**0.5) for a in (-1, 1) for b in (-1, 1)
]


def dijkstra(free: np.ndarray, seeds: np.ndarray, cost: np.ndarray | None = None) -> np.ndarray:
    """8-connected Dijkstra from seed cells over `free` cells; returns metres (inf if unreachable)."""
    dist = np.full(free.shape, np.inf)
    pq = []
    for i, j in np.argwhere(seeds & free):
        dist[i, j] = 0.0
        pq.append((0.0, int(i), int(j)))
    heapq.heapify(pq)
    nx, ny = free.shape
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, c in _NEI:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                nd = d + c * W.RES_M * (1.0 if cost is None else cost[a, b])
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    heapq.heappush(pq, (nd, a, b))
    return dist


def geodesic_field(lay: W.Layout, target: int) -> np.ndarray:
    goal = W._goal_region(lay, target, lay.reach)
    # mild cost to keep clear of obstacles so paths look natural
    clear = lay.clearance
    cost = 1.0 + 1.5 * np.clip(0.45 - clear, 0, None) / 0.45
    field = dijkstra(lay.reach, goal, cost)
    # cells outside the planning region (e.g. hugging a wall) inherit the nearest finite value + offset
    from scipy import ndimage

    bad = ~np.isfinite(field)
    dist, (ii, jj) = ndimage.distance_transform_edt(bad, return_indices=True)
    filled = field[ii, jj] + dist * W.RES_M * 1.5
    filled[lay.blocked] = np.inf
    return np.where(bad, filled, field)


def follow_field(field: np.ndarray, x: float, y: float, lookahead_cells: int = 12):
    """Greedy descent of a distance field; returns list of (x, y) waypoints."""
    i, j = W.to_cell(x, y)
    i, j = int(np.clip(i, 0, W.NX - 1)), int(np.clip(j, 0, W.NY - 1))
    pts = []
    for _ in range(lookahead_cells):
        best = (field[i, j], i, j)
        for di, dj, _c in _NEI:
            a, b = i + di, j + dj
            if 0 <= a < W.NX and 0 <= b < W.NY and field[a, b] < best[0]:
                best = (field[a, b], a, b)
        if best[1:] == (i, j):
            break
        i, j = best[1], best[2]
        pts.append(((i + 0.5) * W.RES_M, (j + 0.5) * W.RES_M))
    return pts


def pursue(pose, point, v_max=0.55):
    """Pure-pursuit style (v, w) toward a world point."""
    x, y, th = pose
    err = np.arctan2(point[1] - y, point[0] - x) - th
    err = (err + np.pi) % (2 * np.pi) - np.pi
    w = float(np.clip(2.2 * err, *W.W_RANGE))
    v = float(v_max * max(0.0, np.cos(err)) ** 3) if abs(err) < 1.2 else 0.0
    return np.array([v, w], np.float32)


def line_free(blocked: np.ndarray, x0, y0, x1, y1, step=0.025) -> bool:
    """True if the straight segment stays out of `blocked` (a configuration-space grid)."""
    n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    i, j = W.to_cell(np.clip(xs, 0, W.WIDTH_M - 1e-6), np.clip(ys, 0, W.HEIGHT_M - 1e-6))
    return not blocked[i, j].any()


def safe_pursue(pose, pts, blocked, v_max=0.55):
    """Pursue the farthest waypoint in line of sight; turn in place if the next step would collide."""
    x, y, th = pose
    target = pts[0]
    for p in pts[1:]:
        if not line_free(blocked, x, y, p[0], p[1]):
            break
        target = p
    a = pursue(pose, target, v_max)
    nx, ny, _ = W.unicycle(x, y, th, float(a[0]), float(a[1]))
    if a[0] > 0 and not line_free(blocked, x, y, nx, ny, 0.01):
        a = np.array([0.0, a[1] if abs(a[1]) > 0.3 else np.sign(a[1] + 1e-6) * 1.2], np.float32)
    return a


def expert_action(env: W.NavEnv) -> np.ndarray:
    x, y, th = env.pose
    pts = follow_field(env.field, x, y)
    if not pts:
        # inside the goal region or stuck: face the object centre
        b = env.layout.obj_boxes[env.target]
        return pursue(env.pose, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), 0.25)
    return safe_pursue(env.pose, pts, env.layout.cspace)
