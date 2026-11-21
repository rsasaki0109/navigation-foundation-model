"""Render hero GIFs: one per paradigm + a 2x2 comparison on the same episode.

Usage: python scripts/make_gifs.py [SEED] [--raycast]

By default the 3D views are rendered by Robot Native Engine's wgpu renderer
(`examples/150_scene_replay_render`; set RNE_REPLAY_BIN to the built binary).  With
--raycast, or when the binary is missing, a built-in pseudo-3D ray caster is used.
"""

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")
import numpy as np
import torch
from PIL import Image, ImageDraw

from navfm import render as R
from navfm import rne_scene as S
from navfm import world as W
from navfm.agents import load_agent

KINDS = ["vlm", "vla", "wm", "wam"]
OUT = "docs/media"
SUB = 2  # rendered sub-frames per control step (pose interpolation)
FRAME_MS = 70


def lerp_pose(a, b, s):
    dth = (b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi
    return np.array([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + dth * s])


def status_line(kind, agent):
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


def snapshot(env, pose, obs, agent, status):
    e = SimpleNamespace(layout=env.layout, target=env.target, instruction=env.instruction, path=[p.copy() for p in env.path],
                        t=env.t, collisions=env.collisions, success=env.success)
    a = SimpleNamespace(name=agent.name, viz=copy.deepcopy(agent.viz))
    return dict(env=e, agent=a, obs=obs, pose=np.array(pose), status=status)


def run_episode(kind, seed):
    torch.set_num_threads(4)
    agent = load_agent(kind)
    env = W.NavEnv(seed)
    agent.reset(env)
    obs = env.observe()
    recs = []
    while not env.done:
        a = agent.act(obs)
        prev_pose = env.pose.copy()
        st = status_line(kind, agent)
        recs.append(snapshot(env, prev_pose, obs, agent, st))
        nobs, _ = env.step(a)
        for s in range(1, SUB):
            recs.append(snapshot(env, lerp_pose(prev_pose, env.pose, s / SUB), obs, agent, st))
        obs = nobs
    recs.append(snapshot(env, env.pose, obs, agent, "reached the target" if env.success else "time out"))
    return recs, env


def compose(kind, recs, env, use_rne):
    base = R.map_base(env.layout)
    views = None
    if use_rne:
        tmp = Path(tempfile.mkdtemp(prefix=f"rne_{kind}_"))
        rne_recs = [dict(pose=r["pose"], viz=r["agent"].viz, path=r["env"].path) for r in recs]
        S.render_episode(env.layout, env.layout.obj_boxes[env.target], kind, R.ACCENT[recs[0]["agent"].name], rne_recs, tmp)
        views = tmp
    frames = []
    for i, r in enumerate(recs):
        ov = fp = None
        if views is not None:
            ov = Image.open(views / f"ov_{i:04d}.png").convert("RGB")
            fp = Image.open(views / f"fpv_{i:04d}.png").convert("RGB")
        frames.append(R.render_panel(r["env"], r["agent"], r["obs"], r["pose"], base, r["status"], ov, fp))
    return frames


def save_gif(frames, path, hold=20):
    frames = frames + [frames[-1]] * hold
    pal_src = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for i, f in enumerate([frames[0], frames[len(frames) // 2], frames[-1]]):
        pal_src.paste(f, (0, i * frames[0].height))
    pal = pal_src.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=FRAME_MS, loop=0, optimize=True, disposal=1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seed = int(args[0]) if args else 35
    use_rne = "--raycast" not in sys.argv and S.rne_binary() is not None
    print("3D renderer:", "RNE wgpu" if use_rne else "built-in ray caster", flush=True)
    runs = {}
    for k in KINDS:
        recs, env = run_episode(k, seed)
        fr = compose(k, recs, env, use_rne)
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
    meta = {k: dict(seed=seed, success=bool(e.success), steps=e.t, collisions=e.collisions, instruction=e.instruction,
                    renderer="RNE wgpu (examples/150_scene_replay_render)" if use_rne else "built-in ray caster")
            for k, (_f, e) in runs.items()}
    json.dump(meta, open(f"{OUT}/hero.json", "w"), indent=1)


if __name__ == "__main__":
    main()
