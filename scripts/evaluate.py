"""Evaluate agents on held-out episodes.

Usage: python scripts/evaluate.py N [vlm vla wm wam ...] [--layout house] [--physics rne]
Results go to docs/results*.json by layout and physics, or NAVFM_RESULTS.
"""

import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")
import numpy as np
import torch

from navfm import world as W
from navfm.agents import load_agent


def run(args):
    kind, seed, layout, physics, ckpt_dir = args
    torch.set_num_threads(1)
    agent = load_agent(kind, ckpt_dir)
    if physics == "rne":
        from navfm.rne_physics import RneNavEnv

        env = RneNavEnv(seed, layout)
    else:
        env = W.NavEnv(seed, layout)
    g0 = env.geodesic(*env.pose[:2])
    agent.reset(env)
    obs = env.observe()
    while not env.done:
        obs, _ = env.step(agent.act(obs))
    length = float(np.sum(np.linalg.norm(np.diff(np.array(env.path)[:, :2], axis=0), axis=1)))
    spl = env.success * g0 / max(g0, length, 1e-6)
    return dict(seed=seed, success=bool(env.success), spl=spl, steps=env.t, collisions=env.collisions)


def evaluate(kind, seeds, layout, physics, ckpt_dir):
    with Pool(4) as p:
        episodes = p.map(run, [(kind, s, layout, physics, ckpt_dir) for s in seeds], chunksize=2)
    return dict(
        success=float(np.mean([x["success"] for x in episodes])),
        spl=float(np.mean([x["spl"] for x in episodes])),
        steps=float(np.mean([x["steps"] for x in episodes if x["success"]] or [0])),
        collisions=float(np.mean([x["collisions"] for x in episodes])),
        episodes=episodes,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", type=int)
    parser.add_argument("kinds", nargs="*")
    parser.add_argument("--layout", choices=("apartment", "house"), default="apartment")
    parser.add_argument("--physics", choices=("python", "rne"), default="python")
    args = parser.parse_intermixed_args()
    if not 1 <= args.n <= 100:
        parser.error("n must be between 1 and 100")
    ckpt = os.environ.get("NAVFM_CKPT", "checkpoints")
    res = {}
    for k in args.kinds or ["vlm", "vla", "wm", "wam"]:
        score = res[k] = evaluate(k, range(args.n), args.layout, args.physics, ckpt)
        print(f"{k:4s} success {score['success']:.2f}  SPL {score['spl']:.2f}  steps {score['steps']:.1f}  coll {score['collisions']:.1f}", flush=True)
    suffix = ("_house" if args.layout == "house" else "") + ("_rne_physics" if args.physics == "rne" else "")
    out = os.environ.get("NAVFM_RESULTS", f"docs/results{suffix}.json")
    try:
        with open(out) as f:
            old = json.load(f)
    except FileNotFoundError:
        old = {}
    old.update(res)
    old["n_episodes"] = args.n
    with open(out, "w") as f:
        json.dump(old, f, indent=1)
