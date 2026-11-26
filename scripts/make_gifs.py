"""Render hero GIFs: one per paradigm + a 2x2 comparison on the same episode.

Usage: python scripts/make_gifs.py [SEED] [--layout house] [--physics rne] [--raycast | --3dgs]

By default the 3D views are rendered by Robot Native Engine's wgpu renderer
(`examples/150_scene_replay_render`; set RNE_REPLAY_BIN to the built binary).  With
--raycast, or when the binary is missing, a built-in pseudo-3D ray caster is used.
"""

import argparse
import copy
import json
import os
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


def run_episode(kind, seed, layout="apartment", physics="python"):
    torch.set_num_threads(1)
    agent = load_agent(kind, os.environ.get("NAVFM_CKPT", "checkpoints"))
    if physics == "rne":
        from navfm.rne_physics import RneNavEnv

        env = RneNavEnv(seed, layout)
    else:
        env = W.NavEnv(seed, layout)
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


def compose(kind, recs, env, use_rne, *, wood_floor=False):
    base = R.map_base(env.layout)
    if not use_rne:
        return compose_frames(recs, base)
    with tempfile.TemporaryDirectory(prefix=f"rne_{kind}_") as directory:
        tmp = Path(directory)
        rne_recs = [dict(pose=r["pose"], viz=r["agent"].viz, path=r["env"].path) for r in recs]
        S.render_episode(env.layout, env.layout.obj_boxes[env.target], kind,
                         R.ACCENT[recs[0]["agent"].name], rne_recs, tmp, wood_floor=wood_floor)
        return compose_frames(recs, base, tmp)


def compose_frames(recs, base, views=None, *, renderer_label="rendered by RNE (wgpu)"):
    frames = []
    for i, r in enumerate(recs):
        ov = fp = None
        if views is not None:
            ov = Image.open(views / f"ov_{i:04d}.png").convert("RGB")
            fp = Image.open(views / f"fpv_{i:04d}.png").convert("RGB")
        frames.append(R.render_panel(r["env"], r["agent"], r["obs"], r["pose"], base, r["status"], ov, fp,
                                     renderer_label=renderer_label))
    return frames


def save_gif(frames, path, hold=20, frame_ms=FRAME_MS):
    frames = frames + [frames[-1]] * hold
    pal_src = Image.new("RGB", (frames[0].width, frames[0].height * 3))
    for i, f in enumerate([frames[0], frames[len(frames) // 2], frames[-1]]):
        pal_src.paste(f, (0, i * frames[0].height))
    pal = pal_src.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=frame_ms, loop=0, optimize=True, disposal=1)


def save_small_gif(frames, path, *, frame_ms=FRAME_MS):
    # Preserve the full episode, including the last observation.
    for stride, width in ((2, 960), (2, 800), (4, 800), (4, 640), (8, 640)):
        sampled = frames[:-1:stride] + [frames[-1]]
        height = frames[0].height * width // frames[0].width
        sampled = [f.resize((width, height), Image.Resampling.LANCZOS) for f in sampled]
        save_gif(sampled, path, hold=max(1, 25 // stride), frame_ms=frame_ms * stride)
        if path.stat().st_size <= 3_000_000:
            return
    path.unlink()
    raise RuntimeError(f"{path} exceeds the 3 MB limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", nargs="?", type=int, default=35)
    parser.add_argument("--layout", choices=("apartment", "house"), default="apartment")
    parser.add_argument("--physics", choices=("python", "rne"), default="python")
    renderer = parser.add_mutually_exclusive_group()
    renderer.add_argument("--raycast", action="store_true")
    renderer.add_argument("--3dgs", dest="splat", action="store_true", help="render mesh-derived 3D Gaussians")
    args = parser.parse_args()
    use_rne = not (args.raycast or args.splat) and S.rne_binary() is not None
    renderer_name = ("3D Gaussian Splatting (mesh-derived)" if args.splat else
                     "RNE wgpu (examples/150_scene_replay_render)" if use_rne else "built-in ray caster")
    print("3D renderer:", renderer_name, flush=True)
    output = Path(OUT)
    output.mkdir(parents=True, exist_ok=True)
    suffix = ("_house" if args.layout == "house" else "") + ("_rne" if args.physics == "rne" else "")
    if args.splat:
        suffix += "_3dgs"
    wood_floor = args.layout == "house" and args.physics == "rne"
    with tempfile.TemporaryDirectory(prefix="navfm_splats_") as directory:
        runs = render_runs(args, use_rne, wood_floor, output, suffix, Path(directory))
    save_comparison(runs, args, output, suffix, renderer_name)


def render_runs(args, use_rne, wood_floor, output, suffix, directory):
    runs = {}
    splat = None
    for k in KINDS:
        recs, env = run_episode(k, args.seed, args.layout, args.physics)
        if args.splat:
            from navfm.splat_render import SplatRenderer

            if splat is None:
                splat = SplatRenderer(env.layout, wood_floor, directory)
                print(f"{len(splat.static):,} static Gaussians", flush=True)
            records = [dict(pose=r["pose"], viz=r["agent"].viz, path=r["env"].path) for r in recs]
            splat.render_episode(k, R.ACCENT[recs[0]["agent"].name], records,
                                 env.layout.obj_boxes[env.target], directory)
            fr = compose_frames(recs, R.map_base(env.layout), directory, renderer_label="3D Gaussian Splatting")
        else:
            fr = compose(k, recs, env, use_rne, wood_floor=wood_floor)
        runs[k] = (fr, env)
        print(k, "success", env.success, "steps", env.t, "frames", len(fr), flush=True)
        if not suffix:
            save_gif(fr, output / f"{k}.gif")
            fr[len(fr) // 2].save(output / f"{k}.png")
    return runs


def save_comparison(runs, args, output, suffix, renderer_name):
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
    if suffix:
        save_small_gif(grid, output / f"hero{suffix}.gif")
    else:
        save_gif(grid, output / "hero.gif", hold=25)
        grid[len(grid) // 2].save(output / "hero.png")
    meta = {k: dict(seed=args.seed, success=bool(e.success), steps=e.t, collisions=e.collisions,
                    instruction=e.instruction,
                    renderer=renderer_name)
            for k, (_f, e) in runs.items()}
    if suffix:
        for data in meta.values():
            data.update(layout=args.layout, physics=args.physics)
    with (output / f"hero{suffix}.json").open("w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
