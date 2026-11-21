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


def _cf_job(seed):
    from navfm.data import counterfactual

    return merge(counterfactual(seed, 12, 10, seed + 17))


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
    if "wm-heads" in which:
        from navfm.data import counterfactual
        from navfm.models import WorldModel

        m = WorldModel()
        print(m.load_state_dict(torch.load(f"{CK}/wm.pt", weights_only=True), strict=False).missing_keys, flush=True)
        torch.save(m.state_dict(), f"{CK}/wm.pt")  # new heads untrained (has_ens=0): planner unchanged
        with Pool(4) as p:
            cf = merge(p.map(_cf_job, range(402_000, 403_000), chunksize=8))
        on = dagger("wm", [0.3], 200, 90_000)[0]
        keys = list(cf)
        d = {k: np.concatenate([base[k], cf[k], on[k]]) for k in keys}
        m = T.train_wm_heads(m, d)
        torch.save(m.state_dict(), f"{CK}/wm.pt")
    if "wm" in which:
        import os

        if os.path.exists(f"{CK}/wm.pt") and "--finetune" in sys.argv:
            from navfm.models import WorldModel

            m = WorldModel()
            missing = m.load_state_dict(torch.load(f"{CK}/wm.pt", weights_only=True), strict=False)
            print("[wm] resumed; newly initialised:", missing.missing_keys, flush=True)
        else:
            m = T.train_wm(base)
            torch.save(m.state_dict(), f"{CK}/wm.pt")
        # planner-style counterfactual rollouts in the true simulator
        with Pool(4) as p:
            cf = merge(p.map(_cf_job, range(400_000, 401_000), chunksize=8))
        print("[wm] counterfactual samples", len(cf["act"]), flush=True)
        keys = list(cf)
        base = {k: np.concatenate([base[k], cf[k]]) for k in keys}
        m = T.train_wm(base, epochs=3, model=m, lr=7e-4)
        torch.save(m.state_dict(), f"{CK}/wm.pt")
        # on-policy data: the planner visits states the expert never does (and exploits model errors there)
        extra = []
        for r in range(2):
            extra += dagger("wm", [0.3], 200, 60_000 + 10_000 * r)
            m = T.train_wm(cat([base] + [{k: e[k] for k in base} for e in extra]), epochs=2, model=m, lr=5e-4)
            torch.save(m.state_dict(), f"{CK}/wm.pt")
    if "vla" in which:
        if "--finetune" in sys.argv:
            from navfm.models import VLA

            m = VLA()
            m.load_state_dict(torch.load(f"{CK}/vla.pt", weights_only=True))
        else:
            m = T.train_vla(base)
            torch.save(m.state_dict(), f"{CK}/vla.pt")
        extra = []
        for r, beta in enumerate([0.0, 0.0]):
            extra += dagger("vla", [beta], 300, 50_000 + 10_000 * r)
            m = T.train_vla(cat([base] + [{k: e[k] for k in base} for e in extra]), epochs=2, model=m)
            torch.save(m.state_dict(), f"{CK}/vla.pt")
    if "wam" in which:
        with Pool(4) as p:
            eps = p.map(_chunk_job, range(300_000, 301_200), chunksize=8)
        cdata = merge(eps)
        keys = list(cdata)
        if "--finetune" in sys.argv:
            m = WAM()
            m.load_state_dict(torch.load(f"{CK}/wam.pt", weights_only=True))
            m = T.train_wam(cdata, epochs=3, model=m)
        else:
            m = T.train_wam(cdata)
        torch.save(m.state_dict(), f"{CK}/wam.pt")
        extra = []
        for r, beta in enumerate([0.3, 0.0]):
            extra += dagger("wam", [beta], 300, 140_000 + 10_000 * r)
            m = T.train_wam(cat([cdata] + [{k: e[k] for k in keys} for e in extra]), epochs=3, model=m)
            torch.save(m.state_dict(), f"{CK}/wam.pt")
