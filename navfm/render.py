"""Frame renderer for hero GIFs: top-down map, pseudo-3D first-person view, panorama and a per-paradigm panel."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import models as M
from . import world as W

PW, PH = 640, 360  # panel size
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = (18, 20, 26)
FG = (230, 232, 238)
DIM = (140, 145, 160)
ACCENT = {
    "VLM + planning": (255, 170, 60),
    "VLA": (90, 200, 255),
    "WM + planning": (190, 120, 255),
    "WAM": (90, 230, 150),
}
SUBTITLE = {
    "VLM + planning": "grounded goal → online map → A* planner",
    "VLA": "pixels + words → action tokens",
    "WM + planning": "CEM search in a learned world model",
    "WAM": "joint future frames + action chunk",
}


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


F10, F11, F12, F13, F15B = font(10), font(11), font(12), font(13), font(15, True)
F11B = font(11, True)

# ---------------------------------------------------------------- top-down map
MAP_X, MAP_Y, PPM = 4, 30, 44.0  # 8 m x 6 m -> 352 x 264 px
MAP_W, MAP_H = int(W.WIDTH_M * PPM), int(W.HEIGHT_M * PPM)


def map_base(lay: W.Layout) -> Image.Image:
    g = lay.grid.T[::-1]  # rows = -y
    rgb = np.zeros(g.shape + (3,), np.uint8)
    rgb[:] = (42, 46, 56)
    ii, jj = np.indices(g.shape)
    rgb[((ii // 10 + jj // 10) % 2 == 0) & (g == W.FREE)] = (48, 52, 63)
    rgb[g == W.WALL] = (165, 168, 180)
    rgb[g == W.LOW] = (120, 86, 60)
    for k in range(W.N_OBJ):
        rgb[g == W.OBJ0 + k] = W.OBJECTS[k][2]
    img = Image.fromarray(rgb).resize((MAP_W, MAP_H), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for k, b in enumerate(lay.obj_boxes):
        cx, cy = w2m((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0, 0)
        d.text((cx, cy), W.OBJECTS[k][0], fill=(15, 15, 15), font=F10, anchor="mm")
    return img


def w2m(x, y, ox=MAP_X, oy=MAP_Y):
    return ox + x * PPM, oy + (W.HEIGHT_M - y) * PPM


def integrate(pose, acts):
    x, y, th = pose
    pts = [(x, y)]
    for v, w in acts:
        x, y, th = W.unicycle(x, y, th, float(v), float(w))
        pts.append((x, y))
    return pts


# ---------------------------------------------------------------- first-person view
FPV_X, FPV_Y, FPV_W, FPV_H = 364, 30, 272, 150
FOV = np.deg2rad(100)
OBJ_HEIGHT = {0: 0.55, 1: 0.5, 2: 0.95, 3: 1.25, 4: 0.45, 5: 0.85}
WALL_H, LOW_H, CAM_H = 1.5, 0.42, 0.55


def _march(lay, x, y, angles, step=0.02):
    n = int(W.MAX_RANGE_M / step)
    ts = np.arange(1, n + 1) * step
    px = x + np.cos(angles)[:, None] * ts[None]
    py = y + np.sin(angles)[:, None] * ts[None]
    lab = lay.grid[np.clip((px / W.RES_M).astype(int), 0, W.NX - 1), np.clip((py / W.RES_M).astype(int), 0, W.NY - 1)]

    def first(m):
        has = m.any(1)
        return np.where(has, ts[m.argmax(1)], np.inf), m.argmax(1)

    dw, _ = first(lab == W.WALL)
    do, io = first(lab >= W.OBJ0)
    dl, _ = first(lab == W.LOW)
    lo = lab[np.arange(len(angles)), io]
    return dw, do, lo, dl


def fpv(lay: W.Layout, pose, highlight_cols=None) -> np.ndarray:
    x, y, th = pose
    H, Wd = FPV_H, FPV_W
    rel = np.linspace(FOV / 2, -FOV / 2, Wd)
    ang = th + rel
    dw, do, lo, dl = _march(lay, x, y, ang)
    f = (Wd / 2) / np.tan(FOV / 2)
    hor = H * 0.5
    rows = np.arange(H)[:, None].astype(np.float32)
    img = np.zeros((H, Wd, 3), np.float32)
    # ceiling + floor casting with tiles
    img[:] = np.array([34, 37, 46], np.float32)
    below = rows > hor + 0.5
    dist = CAM_H * f / np.maximum(rows - hor, 0.5)
    corr = dist / np.cos(rel)[None, :]
    fx = x + np.cos(ang)[None] * corr
    fy = y + np.sin(ang)[None] * corr
    tile = ((np.floor(fx / 0.5) + np.floor(fy / 0.5)) % 2)[..., None]
    floor = (np.array([92, 84, 76]) + tile * np.array([18, 16, 14])) / (1 + 0.18 * corr[..., None])
    img = np.where(below[..., None], floor, img)

    def slab(depth, height, colour):
        z = depth * np.cos(rel)
        top = hor - (height - CAM_H) * f / z
        bot = hor + CAM_H * f / z
        m = (rows >= top[None]) & (rows <= bot[None]) & np.isfinite(depth)[None]
        shade = 1.0 / (1.0 + 0.16 * depth)
        # subtle vertical gradient for depth cues
        grad = 0.85 + 0.15 * np.clip((rows - top[None]) / np.maximum(bot - top, 1)[None], 0, 1)
        col = colour[None] * (shade[None, :, None] * grad[..., None])
        return m, col

    layers = [
        (dw, np.full(Wd, WALL_H), np.broadcast_to(np.array(W.WALL_RGB, np.float32), (Wd, 3))),
        (do, np.array([OBJ_HEIGHT.get(int(l) - W.OBJ0, 0.6) for l in lo]), W.LABEL_RGB[lo]),
        (dl, np.full(Wd, LOW_H), np.broadcast_to(np.array(W.LOW_RGB, np.float32), (Wd, 3))),
    ]
    # painter's algorithm per column: far to near
    order = np.argsort(-np.nan_to_num(np.stack([l[0] for l in layers]), posinf=1e9), axis=0)
    for rank in range(3):
        for li, (dep, hgt, colr) in enumerate(layers):
            sel = order[rank] == li
            if not sel.any():
                continue
            d = np.where(sel, dep, np.inf)
            z = d * np.cos(rel)
            top = hor - (hgt - CAM_H) * f / z
            bot = hor + CAM_H * f / z
            m = (rows >= top[None]) & (rows <= bot[None]) & np.isfinite(d)[None]
            shade = 1.0 / (1.0 + 0.16 * d)
            grad = 0.82 + 0.18 * np.clip((rows - top[None]) / np.maximum(bot - top, 1)[None], 0, 1)
            col = colr[None] * (np.where(np.isfinite(shade), shade, 0)[None, :, None] * grad[..., None])
            # dark outline at slab tops for readability
            edge = m & (rows < top[None] + 1.5)
            img = np.where(m[..., None], col, img)
            img = np.where(edge[..., None], img * 0.55, img)
    return np.clip(img, 0, 255).astype(np.uint8)


def pano_strip(pano: np.ndarray, w: int, h: int) -> Image.Image:
    """pano: (5, 128) network tensor -> colour strip, forward at centre."""
    rgb = np.clip(pano[:3].T, 0, 1)
    img = (rgb[None] * 255).astype(np.uint8)
    return Image.fromarray(img).resize((w, h), Image.NEAREST)


def depth_strip(pano: np.ndarray, w: int, h: int) -> Image.Image:
    """Depth/colour visual for predicted panoramas: colour, with brightness from predicted depth."""
    rgb = np.clip(pano[:3].T, 0, 1)
    depth_m = np.clip(pano[3], 0.01, 1) * W.MAX_RANGE_M
    rows = np.linspace(0, 1, h)[:, None]
    hgt = np.clip(0.55 / depth_m, 0.06, 0.5)[None]  # near -> tall (like a tiny first-person render)
    m = np.abs(rows - 0.5) <= hgt
    out = np.where(m[..., None], rgb[None] * 255, np.array([28, 30, 38]))
    return Image.fromarray(out.astype(np.uint8)).resize((w, h), Image.NEAREST)


# ---------------------------------------------------------------- full panel
def draw_robot(d, pose, colour, r_px=None):
    x, y, th = pose
    cx, cy = w2m(x, y)
    r = r_px or W.ROBOT_RADIUS_M * PPM
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(245, 245, 250), outline=colour, width=2)
    d.line([cx, cy, cx + np.cos(th) * r * 1.6, cy - np.sin(th) * r * 1.6], fill=colour, width=3)
    # FOV wedge
    for s in (-1, 1):
        a = th + s * FOV / 2
        d.line([cx, cy, cx + np.cos(a) * 34, cy - np.sin(a) * 34], fill=colour + (110,), width=1)


def render_panel(env: W.NavEnv, agent, obs, pose, base: Image.Image, status: str = "") -> Image.Image:
    name = agent.name
    acc = ACCENT[name]
    img = Image.new("RGB", (PW, PH), BG)
    img.paste(base, (MAP_X, MAP_Y))
    d = ImageDraw.Draw(img, "RGBA")
    # header
    d.rectangle([0, 0, PW, 25], fill=(28, 31, 40))
    d.rectangle([0, 0, 5, 25], fill=acc)
    d.text((12, 4), name, fill=acc, font=F15B)
    tw = d.textlength(name, font=F15B)
    d.text((20 + tw, 7), SUBTITLE[name], fill=DIM, font=F11)
    d.text((PW - 8, 7), f'"{env.instruction}"', fill=FG, font=F11B, anchor="ra")
    viz = agent.viz

    # target highlight
    b = env.layout.obj_boxes[env.target]
    x0, y1 = w2m(b[0], b[1])
    x1, y0 = w2m(b[2], b[3])
    pad = W.SUCCESS_DIST_M * PPM
    d.rounded_rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], radius=pad, outline=(255, 255, 255, 90), width=1)

    if name == "VLM + planning":
        occ = viz.get("occ")
        if occ is not None:
            from scipy import ndimage

            known = ndimage.binary_closing(occ > 0, iterations=3)
            o = np.where(known, np.maximum(occ, 1), 0).T[::-1]
            ov = np.zeros(o.shape + (4,), np.uint8)
            ov[o == 0] = (0, 0, 0, 120)  # unknown: darken
            ov[o == 2] = (255, 170, 60, 150)
            ovi = Image.fromarray(ov, "RGBA").resize((MAP_W, MAP_H), Image.NEAREST)
            img.paste(ovi, (MAP_X, MAP_Y), ovi)
            d = ImageDraw.Draw(img, "RGBA")
        path = viz.get("path") or []
        if len(path) > 1:
            d.line([w2m(*p) for p in path], fill=acc + (255,), width=3)
        g = viz.get("goal")
        if g is not None:
            gx, gy = w2m(*g)
            d.line([gx - 7, gy - 7, gx + 7, gy + 7], fill=acc, width=3)
            d.line([gx - 7, gy + 7, gx + 7, gy - 7], fill=acc, width=3)
    elif name == "WM + planning" and "samples" in viz:
        cost = viz["sample_cost"]
        cn = (cost - cost.min()) / (np.ptp(cost) + 1e-6)
        for acts, c in sorted(zip(viz["samples"], cn), key=lambda t: -t[1]):
            pts = integrate(pose, acts)
            col = tuple(int(v) for v in (np.array(acc) * (1 - c) + np.array((90, 90, 100)) * c)) + (int(200 - 140 * c),)
            d.line([w2m(*p) for p in pts], fill=col, width=1)
        d.line([w2m(*p) for p in integrate(pose, viz["best"])], fill=(255, 255, 255, 255), width=3)
    elif name == "WAM" and "chunk" in viz:
        pts = integrate(viz["chunk_pose"], viz["chunk"])
        d.line([w2m(*p) for p in pts], fill=acc + (255,), width=3)
        for p in [pts[k] for k in M.WAM.FUTURE]:
            px, py = w2m(*p)
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=acc)
    elif name == "VLA":
        pass

    # trajectory
    trail = [w2m(p[0], p[1]) for p in env.path[: max(1, len(env.path))]]
    if len(trail) > 1:
        d.line(trail, fill=(255, 255, 255, 150), width=2)
    draw_robot(d, pose, acc)

    # status line under map
    sy = MAP_Y + MAP_H + 6
    d.text((MAP_X + 2, sy), status, fill=FG, font=F12)
    d.text((MAP_X + 2, sy + 17), f"t = {env.t * W.DT_S:4.1f} s   collisions {env.collisions}", fill=DIM, font=F11)
    if env.success:
        d.rounded_rectangle([MAP_X + MAP_W - 92, sy, MAP_X + MAP_W - 2, sy + 22], 6, fill=(40, 170, 90))
        d.text((MAP_X + MAP_W - 47, sy + 11), "REACHED", fill=(255, 255, 255), font=F11B, anchor="mm")

    # first-person view
    fp = fpv(env.layout, pose)
    img.paste(Image.fromarray(fp), (FPV_X, FPV_Y))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([FPV_X, FPV_Y, FPV_X + FPV_W - 1, FPV_Y + FPV_H - 1], outline=(70, 75, 90))
    d.text((FPV_X + 5, FPV_Y + 3), "egocentric camera (rendered for you)", fill=(230, 230, 235, 200), font=F10)

    # panorama input strip
    py0 = FPV_Y + FPV_H + 16
    pano = W.pano_tensor(obs)
    d.text((FPV_X, py0 - 13), "360° input panorama (what the model sees)", fill=DIM, font=F10)
    img.paste(pano_strip(pano, FPV_W, 16), (FPV_X, py0))
    d = ImageDraw.Draw(img, "RGBA")
    cxp = FPV_X + FPV_W / 2
    d.line([cxp, py0 - 2, cxp, py0 + 18], fill=(255, 255, 255, 160))

    # per-paradigm panel
    y0 = py0 + 24
    if name == "VLM + planning" and "prob" in viz:
        p = viz["prob"]
        d.text((FPV_X, y0), "VLM grounding  p(target | column, text)", fill=acc, font=F11B)
        bw = FPV_W / len(p)
        hb = 46
        for i, v in enumerate(p):
            h = v * hb
            d.rectangle([FPV_X + i * bw, y0 + 16 + hb - h, FPV_X + (i + 1) * bw, y0 + 16 + hb], fill=acc + (int(80 + 175 * v),))
        d.line([FPV_X, y0 + 16 + hb * 0.5, FPV_X + FPV_W, y0 + 16 + hb * 0.5], fill=(255, 255, 255, 60))
        d.text((FPV_X, y0 + 66), "goal = back-project(argmax, depth) -> A*", fill=DIM, font=F10)
        # mark detection on the input strip
        cols = viz.get("det_cols")
        if cols is not None and len(cols):
            for c in cols:
                d.rectangle([FPV_X + c * bw, py0 - 1, FPV_X + (c + 1) * bw, py0 + 16], outline=acc)
    elif name == "VLA" and "token_prob" in viz:
        p = viz["token_prob"].reshape(len(M.V_BINS), len(M.W_BINS))
        d.text((FPV_X, y0), "action-token distribution  (v x ω)", fill=acc, font=F11B)
        cw, ch = 30, 14
        gx0 = FPV_X + 56
        for vi in range(len(M.V_BINS)):
            d.text((FPV_X, y0 + 18 + (len(M.V_BINS) - 1 - vi) * (ch + 2)), f"v={M.V_BINS[vi]:.2f}", fill=DIM, font=F10)
            for wi in range(len(M.W_BINS)):
                v = float(p[vi, wi])
                yy = y0 + 17 + (len(M.V_BINS) - 1 - vi) * (ch + 2)
                d.rectangle([gx0 + wi * cw, yy, gx0 + wi * cw + cw - 3, yy + ch], fill=acc + (int(25 + 230 * min(1, v * 1.5)),))
                if vi * len(M.W_BINS) + wi == viz["token"]:
                    d.rectangle([gx0 + wi * cw - 1, yy - 1, gx0 + wi * cw + cw - 2, yy + ch + 1], outline=(255, 255, 255), width=2)
        d.text((gx0, y0 + 70), "← left        ω        right →", fill=DIM, font=F10)
    elif name == "WM + planning" and "imagined" in viz:
        d.text((FPV_X, y0), f"imagined futures of best plan   ĝ={viz['pred_dist']:.1f} m", fill=acc, font=F11B)
        for i, (fr, k) in enumerate(zip(viz["imagined"], (2, 4, 6, 8))):
            yy = y0 + 16 + i * 29
            d.text((FPV_X, yy + 8), f"t+{k}", fill=DIM, font=F10)
            img.paste(depth_strip(fr, FPV_W - 30, 27), (FPV_X + 30, yy))
        d = ImageDraw.Draw(img, "RGBA")
    elif name == "WAM" and "future" in viz:
        d.text((FPV_X, y0), "predicted future  +  8-step action chunk", fill=acc, font=F11B)
        for i, k in enumerate(M.WAM.FUTURE):
            yy = y0 + 16 + i * 29
            d.text((FPV_X, yy + 8), f"t+{k}", fill=DIM, font=F10)
            img.paste(depth_strip(viz["future"][i], FPV_W - 30, 27), (FPV_X + 30, yy))
        d = ImageDraw.Draw(img, "RGBA")
    return img
