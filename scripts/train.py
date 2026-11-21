"""Train all four paradigms.  Usage: python scripts/train.py [vlm|vla|wm|wam ...]"""

import sys
from multiprocessing import Pool

sys.path.insert(0, ".")
import numpy as np
import torch

from navfm import train as T
from navfm.agents import load_agent
from navfm.data import merge, rollout
from navfm.models import WAM

CK = "checkpoints"
DAGGER_SEED0 = 200_000
CHUNK = (WAM.CHUNK, WAM.FUTURE)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _dagger_job(args):
    kind, seed, beta = args
    torch.set_num_threads(1)
    agent = load_agent(kind, CK)
    return rollout(seed, policy=agent, beta=beta, chunk=CHUNK if kind == "wam" else None)


def _chunk_job(seed):
    return rollout(seed, noise=1.0 if seed % 4 else 0.0, chunk=CHUNK)


def dagger(kind, rounds, n, offset):
    out = []
    for r, beta in enumerate(rounds):
        seeds = range(DAGGER_SEED0 + offset + r * n, DAGGER_SEED0 + offset + (r + 1) * n)
        with Pool(4) as p:
            eps = p.map(_dagger_job, [(kind, s, beta) for s in seeds], chunksize=4)
        print(f"[{kind}] dagger round {r} beta={beta} learner-success {np.mean([e['success'] for e in eps]):.2f}", flush=True)
        out.append(merge(eps))
    return out


def cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


if __name__ == "__main__":
    torch.set_num_threads(4)
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["vlm", "vla", "wm", "wam"]
    base = load("data/expert.npz")
    if "vlm" in which:
        torch.save(T.train_vlm(base).state_dict(), f"{CK}/vlm.pt")
    if "wm" in which:
        import os

        if os.path.exists(f"{CK}/wm.pt") and "--finetune" in sys.argv:
            from navfm.models import WorldModel

            m = WorldModel()
            m.load_state_dict(torch.load(f"{CK}/wm.pt", weights_only=True))
        else:
            m = T.train_wm(base)
            torch.save(m.state_dict(), f"{CK}/wm.pt")
        # on-policy data: the planner visits states the expert never does (and exploits model errors there)
        extra = []
        for r in range(2):
            extra += dagger("wm", [0.3], 200, 50_000 + 10_000 * r)
            m = T.train_wm(cat([base] + [{k: e[k] for k in base} for e in extra]), epochs=2, model=m, lr=5e-4)
            torch.save(m.state_dict(), f"{CK}/wm.pt")
    if "vla" in which:
        m = T.train_vla(base)
        torch.save(m.state_dict(), f"{CK}/vla.pt")
        extra = []
        for r, beta in enumerate([0.3, 0.0, 0.0]):
            extra += dagger("vla", [beta], 400, 10_000 * r)
            m = T.train_vla(cat([base] + [{k: e[k] for k in base} for e in extra]), epochs=4, model=m)
            torch.save(m.state_dict(), f"{CK}/vla.pt")
    if "wam" in which:
        with Pool(4) as p:
            eps = p.map(_chunk_job, range(300_000, 301_200), chunksize=8)
        cdata = merge(eps)
        keys = list(cdata)
        m = T.train_wam(cdata)
        torch.save(m.state_dict(), f"{CK}/wam.pt")
        extra = []
        for r, beta in enumerate([0.3, 0.0, 0.0]):
            extra += dagger("wam", [beta], 400, 100_000 + 10_000 * r)
            m = T.train_wam(cat([cdata] + [{k: e[k] for k in keys} for e in extra]), epochs=4, model=m)
            torch.save(m.state_dict(), f"{CK}/wam.pt")
