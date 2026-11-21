"""Render hero GIFs: one per paradigm + a 2x2 comparison on the same episode.

Usage: python scripts/make_gifs.py [SEED]
"""

import json
import sys

sys.path.insert(0, ".")
import numpy as np
import torch
from PIL import Image, ImageDraw

from navfm import render as R
from navfm import world as W
from navfm.agents import load_agent

KINDS = ["vlm", "vla", "wm", "wam"]
OUT = "docs/media"
SUB = 2  # rendered sub-frames per control step (pose interpolation)
FRAME_MS = 70


def lerp_pose(a, b, s):
    dth = (b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi
    return np.array([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + dth * s])


def status_line(kind, agent, env):
    v = agent.viz
    if kind == "vlm":
        if v.get("mode") == "goal":
            return f"goal grounded (p={v.get('conf', 0):.2f})  -> following A* path"
        return "target not visible  -> frontier exploration"
    if kind == "vla":
        return f"token #{v.get('token', 0)}  p={v.get('token_prob', np.zeros(1)).max():.2f}"
    if kind == "wm":
        return f"CEM {agent.N} samples x {agent.ITERS} iters (+ policy prior), H={agent.H}"
    return f"chunk of {len(v.get('chunk', []))} actions, execute {agent.EXEC}, re-plan"


def episode_frames(kind, seed):
    torch.set_num_threads(4)
    agent = load_agent(kind)
    env = W.NavEnv(seed)
    agent.reset(env)
    base = R.map_base(env.layout)
    obs = env.observe()
    frames = []
    while not env.done:
        a = agent.act(obs)
        prev_pose = env.pose.copy()
        st = status_line(kind, agent, env)
        frames.append(R.render_panel(env, agent, obs, prev_pose, base, st))
        nobs, _ = env.step(a)
        for s in range(1, SUB):
            frames.append(R.render_panel(env, agent, obs, lerp_pose(prev_pose, env.pose, s / SUB), base, st))
        obs = nobs
    frames.append(R.render_panel(env, agent, obs, env.pose, base, "reached the target" if env.success else "time out"))
    return frames, env


def save_gif(frames, path, hold=20):
    frames = frames + [frames[-1]] * hold
    pal_src = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for i, f in enumerate([frames[0], frames[len(frames) // 2], frames[-1]]):
        pal_src.paste(f, (0, i * frames[0].height))
    pal = pal_src.quantize(colors=200, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=FRAME_MS, loop=0, optimize=True, disposal=1)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    runs = {}
    for k in KINDS:
        fr, env = episode_frames(k, seed)
        runs[k] = (fr, env)
        print(k, "success", env.success, "steps", env.t, "frames", len(fr), flush=True)
        save_gif(fr, f"{OUT}/{k}.gif")
        fr[min(len(fr) - 1, len(fr) // 2)].save(f"{OUT}/{k}.png")
    n = max(len(r[0]) for r in runs.values())
    grid = []
    for i in range(n):
        g = Image.new("RGB", (R.PW * 2, R.PH * 2), R.BG)
        for j, k in enumerate(KINDS):
            fr = runs[k][0]
            g.paste(fr[min(i, len(fr) - 1)], ((j % 2) * R.PW, (j // 2) * R.PH))
        d = ImageDraw.Draw(g)
        d.line([R.PW, 0, R.PW, 2 * R.PH], fill=(60, 64, 78), width=2)
        d.line([0, R.PH, 2 * R.PW, R.PH], fill=(60, 64, 78), width=2)
        grid.append(g)
    save_gif(grid, f"{OUT}/hero.gif", hold=25)
    grid[len(grid) // 2].save(f"{OUT}/hero.png")
    meta = {k: dict(seed=seed, success=bool(e.success), steps=e.t, collisions=e.collisions, instruction=e.instruction)
            for k, (_f, e) in runs.items()}
    json.dump(meta, open(f"{OUT}/hero.json", "w"), indent=1)


if __name__ == "__main__":
    main()
