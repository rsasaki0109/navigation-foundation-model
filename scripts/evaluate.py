"""Evaluate agents on held-out episodes.

Usage: python scripts/evaluate.py N [vlm vla wm wam ...] [--layout house]
Results go to docs/results.json (apartment) or docs/results_house.json (house).
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


LAYOUT = "house" if "--layout" in sys.argv and sys.argv[sys.argv.index("--layout") + 1] == "house" else "apartment"


def run(args):
    kind, seed = args
    torch.set_num_threads(1)
    agent = load_agent(kind, os.environ.get("NAVFM_CKPT", "checkpoints"))
    env = W.NavEnv(seed, LAYOUT)
    g0 = env.geodesic(*env.pose[:2])
    agent.reset(env)
    obs = env.observe()
    while not env.done:
        obs, _ = env.step(agent.act(obs))
    length = float(np.sum(np.linalg.norm(np.diff(np.array(env.path)[:, :2], axis=0), axis=1)))
    spl = env.success * g0 / max(g0, length, 1e-6)
    return dict(seed=seed, success=bool(env.success), spl=spl, steps=env.t, collisions=env.collisions)


if __name__ == "__main__":
    n = int(sys.argv[1])
    args = [a for i, a in enumerate(sys.argv[2:], 2) if not a.startswith("--") and sys.argv[i - 1] != "--layout"]
    kinds = args or ["vlm", "vla", "wm", "wam"]
    res = {}
    for k in kinds:
        with Pool(4) as p:
            r = p.map(run, [(k, s) for s in range(n)], chunksize=2)
        res[k] = dict(
            success=float(np.mean([x["success"] for x in r])),
            spl=float(np.mean([x["spl"] for x in r])),
            steps=float(np.mean([x["steps"] for x in r if x["success"]] or [0])),
            collisions=float(np.mean([x["collisions"] for x in r])),
            episodes=r,
        )
        print(f"{k:4s} success {res[k]['success']:.2f}  SPL {res[k]['spl']:.2f}  steps {res[k]['steps']:.1f}  coll {res[k]['collisions']:.1f}", flush=True)
    out = os.environ.get("NAVFM_RESULTS") or ("docs/results.json" if LAYOUT == "apartment" else "docs/results_house.json")
    try:
        old = json.load(open(out))
    except FileNotFoundError:
        old = {}
    old.update(res)
    old["n_episodes"] = n
    json.dump(old, open(out, "w"), indent=1)
