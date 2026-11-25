"""Train all four paradigms.  Usage: python scripts/train.py [vlm|vla|wm|wam ...]"""

import json
import os
import shutil
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch

from navfm import train as T
from navfm.agents import load_agent
from navfm.data import merge, rollout
from navfm.models import WAM

CK = os.environ.get("NAVFM_CKPT", "checkpoints")
DAGGER_SEED0 = 200_000
CHUNK = (WAM.CHUNK, WAM.FUTURE)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _dagger_job(args):
    kind, seed, beta, ckpt_dir = args
    torch.set_num_threads(1)
    agent = load_agent(kind, ckpt_dir)
    return rollout(seed, policy=agent, beta=beta, chunk=CHUNK if kind == "wam" else None,
                   track_executed_action=kind == "rwm_ep")


def _cf_job(seed):
    from navfm.data import counterfactual

    return merge(counterfactual(seed, 12, 10, seed + 17))


def _cf_prefix_job(seed):
    from navfm.data import counterfactual

    return merge(counterfactual(seed, 12, 10, seed + 23, prefix=13))


def _cf_full_job(seed):
    from navfm.data import counterfactual

    return merge(counterfactual(seed, 6, 10, seed + 29, prefix=-1))


def _chunk_job(seed):
    return rollout(seed, noise=1.0 if seed % 4 else 0.0, chunk=CHUNK)


def dagger(kind, rounds, n, offset, ckpt_dir=CK, keep_episodes=False):
    out = []
    for r, beta in enumerate(rounds):
        seeds = range(DAGGER_SEED0 + offset + r * n, DAGGER_SEED0 + offset + (r + 1) * n)
        with Pool(4) as p:
            eps = p.map(_dagger_job, [(kind, s, beta, ckpt_dir) for s in seeds], chunksize=4)
        print(f"[{kind}] dagger round {r} beta={beta} learner-success {np.mean([e['success'] for e in eps]):.2f}", flush=True)
        out.append(eps if keep_episodes else merge(eps))
    return out


def cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def train_rwm_onpolicy(base, cf=None, default_rounds=5, initial_model=None):
    from scripts.evaluate import evaluate

    rounds = int(os.environ.get("ONPOLICY_ROUNDS", str(default_rounds)))
    rollouts = int(os.environ.get("ONPOLICY_EPISODES", "160"))
    capacity = int(os.environ.get("ONPOLICY_REPLAY", str(2 * rollouts)))
    beta = float(os.environ.get("ONPOLICY_BETA", "0.1"))
    decay = float(os.environ.get("ONPOLICY_LR_DECAY", "0.7"))
    offset = int(os.environ.get("ONPOLICY_OFFSET", "130000"))
    cf_count = int(os.environ.get("ONPOLICY_COUNTERFACTUALS", "700"))
    epochs = int(os.environ.get("ONPOLICY_EPOCHS", "3"))
    start = DAGGER_SEED0 + offset
    if not (rounds > 0 and 0 < rollouts <= 10_000 and capacity > 0 and cf_count > 0 and epochs > 0
            and 0 <= beta <= 1 and 0 < decay <= 1 and start >= 100_000
            and start + (rounds - 1) * 10_000 + rollouts <= 900_000
            and 406_000 + cf_count <= 900_000
            and np.all((base["ep"] >= 100_000) & (base["ep"] < 900_000))):
        raise ValueError("Invalid on-policy parameters or training seed range")

    layout = os.environ.get("NAVFM_LAYOUT", "apartment")
    if layout not in ("apartment", "house", "mixed"):
        raise ValueError(f"Unknown training layout: {layout}")
    output = Path(os.environ.get("ONPOLICY_OUT", f"{CK}/rwm_ep_onpolicy"))
    output.mkdir(parents=True, exist_ok=False)
    current = output / "round_00"
    current.mkdir()
    if initial_model is None:
        shutil.copyfile(Path(CK) / "rwm_ep.pt", current / "rwm_ep.pt")
        initial_model = load_agent("rwm_ep", current).model
    else:
        torch.save(initial_model.state_dict(), current / "rwm_ep.pt")

    validation = []

    def score(path):
        seeds = range(900_000, 900_030)
        suites = {"apartment": seeds[::2], "house": seeds[1::2]} if layout == "mixed" else {layout: seeds}
        scores = {name: evaluate("rwm_ep", subset, name, "python", path) for name, subset in suites.items()}
        success = float(np.mean([x["success"] for x in scores.values()]))
        validation.append(dict(round=path.name, success=success, layouts=scores))
        print(f"[rwm-ep] {path.name} validation success {success:.3f}", flush=True)
        return success

    def record(best):
        temporary = output / "validation.tmp"
        with temporary.open("w") as f:
            json.dump(dict(best=best.name, beta=beta, lr_decay=decay, replay_episodes=capacity,
                           rollouts_per_round=rollouts, counterfactual_seeds=cf_count, epochs=epochs,
                           rounds=validation), f, indent=1)
        temporary.replace(output / "validation.json")

    def publish_best(path):
        temporary = output / "rwm_ep.tmp"
        shutil.copyfile(path / "rwm_ep.pt", temporary)
        temporary.replace(output / "rwm_ep.pt")

    best = current
    best_score = score(current)
    publish_best(best)
    record(best)
    if cf is None:
        with Pool(4) as p:
            cf = merge(p.map(_cf_full_job, range(406_000, 406_000 + cf_count), chunksize=8))
    keys = list(cf)
    fixed = {k: np.concatenate([base[k], cf[k]]) for k in keys}
    max_replay_steps = len(fixed["act"])
    replay = deque(maxlen=capacity)
    model = initial_model.train()
    for r in range(1, rounds + 1):
        episodes = dagger("rwm_ep", [beta], rollouts, offset + 10_000 * (r - 1), current,
                          keep_episodes=True)[0]
        replay.extend(episodes)
        replay_steps = sum(len(e["act"]) for e in replay)
        while replay_steps > max_replay_steps:
            replay_steps -= len(replay.popleft()["act"])
        data = {k: np.concatenate([fixed[k]] + [e[k] for e in replay]) for k in keys}
        lr = 5e-4 * decay ** (r - 1)
        print(f"[rwm-ep] round {r}: {replay_steps} replay steps, lr={lr:g}", flush=True)
        model = T.train_rwm_episodes(data, epochs=epochs, model=model, lr=lr, seed=r)
        current = output / f"round_{r:02d}"
        current.mkdir()
        torch.save(model.state_dict(), current / "rwm_ep.pt")
        success = score(current)
        if success > best_score:
            best, best_score = current, success
            publish_best(best)
        record(best)


def _rne_job(seed):
    from navfm.rne_physics import RneNavEnv

    env = RneNavEnv(seed)
    torch.set_num_threads(1)
    policy = load_agent("wm", CK) if seed % 4 == 0 else None
    return rollout(seed, policy=policy, beta=0.2, noise=1.0, env=env)


def train_rne_wm():
    output = Path(os.environ.get("NAVFM_RNE_OUT", "checkpoints/rne"))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "wm.pt"
    if destination.exists() or destination.resolve() == (Path(CK) / "wm.pt").resolve():
        raise FileExistsError(destination)
    episodes = int(os.environ.get("NAVFM_RNE_EPISODES", "240"))
    if not 1 <= episodes <= 300_000:
        raise ValueError("NAVFM_RNE_EPISODES must be between 1 and 300000")
    with Pool(4) as p:
        data = merge(p.map(_rne_job, range(600_000, 600_000 + episodes), chunksize=4))
    print(f"[wm-rne] {len(data['act'])} RNE steps", flush=True)
    model = load_agent("wm", CK).model
    model = T.train_wm(data, epochs=2, model=model, lr=2e-4, dynamics_only=True)
    torch.save(model.state_dict(), destination)


if __name__ == "__main__":
    torch.set_num_threads(4)
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["vlm", "vla", "wm", "wam"]
    if "wm-rne" in which:
        train_rne_wm()
        which.remove("wm-rne")
        if not which:
            sys.exit(0)
    base = load(os.environ.get("NAVFM_DATA", "data/expert.npz"))
    if "vlm" in which:
        torch.save(T.train_vlm(base).state_dict(), f"{CK}/vlm.pt")
    if "rwm-ep-onpolicy" in which:
        train_rwm_onpolicy(base)
    if "rwm-ep" in which and "rwm-ep-onpolicy" not in which:
        # recurrent world model trained on whole episodes from their start (state carried as deployed)

        with Pool(4) as p:
            cf = merge(p.map(_cf_full_job, range(406_000, 406_700), chunksize=8))
        keys = list(cf)
        d = {k: np.concatenate([base[k], cf[k]]) for k in keys}
        print("[rwm-ep] samples", len(d["act"]), flush=True)
        m = T.train_rwm_episodes(d, epochs=10)
        train_rwm_onpolicy(base, cf, default_rounds=2, initial_model=m)
    if "rwm" in which and "rwm-ep" not in which:
        # recurrent world model: expert data + counterfactual branches with real history + on-policy rounds
        with Pool(4) as p:
            cf = merge(p.map(_cf_prefix_job, range(404_000, 405_200), chunksize=8))
        keys = list(cf)
        d = {k: np.concatenate([base[k], cf[k]]) for k in keys}
        print("[rwm] samples", len(d["act"]), flush=True)
        m = T.train_rwm(d, epochs=6)
        torch.save(m.state_dict(), f"{CK}/rwm.pt")
        extra = []
        for r in range(2):
            extra += dagger("rwm", [0.3], 200, 70_000 + 10_000 * r)
            dd = {k: np.concatenate([d[k]] + [e[k] for e in extra]) for k in keys}
            m = T.train_rwm(dd, epochs=2, model=m, lr=5e-4)
            torch.save(m.state_dict(), f"{CK}/rwm.pt")
    if "wm-heads" in which:
        from navfm.data import counterfactual
        from navfm.models import WorldModel

        m = WorldModel()
        print(m.load_state_dict(torch.load(f"{CK}/wm.pt", weights_only=True), strict=False).missing_keys, flush=True)
        resumed = bool(m.has_ens)
        if not resumed:
            torch.save(m.state_dict(), f"{CK}/wm.pt")  # new heads untrained (has_ens=0): planner unchanged
        with Pool(4) as p:
            cf = merge(p.map(_cf_job, range(402_000, 403_000), chunksize=8))
        parts = [base, cf]
        if not resumed:
            parts.append(dagger("wm", [0.3], 200, 90_000)[0])
        keys = list(cf)
        d = {k: np.concatenate([x[k] for x in parts]) for k in keys}
        m = T.train_wm_heads(m, d, epochs=12 if resumed else 3)
        torch.save(m.state_dict(), f"{CK}/wm.pt")
    if "wm" in which:
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
