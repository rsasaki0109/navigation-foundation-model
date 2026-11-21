"""Rollout collection: expert with DART noise, or DAgger mixtures with a learner."""

from __future__ import annotations

import copy

import numpy as np

from . import world as W
from .expert import expert_action
from .text import tokenize


def expert_chunk(env: W.NavEnv, k: int, future: tuple):
    """Counterfactual expert rollout from the current state: k actions + panoramas at `future` steps."""
    sim = copy.copy(env)
    sim.pose, sim.path, sim.done = env.pose.copy(), [], False
    acts, frames = [], {}
    last = W.pano_tensor(env.observe())
    for i in range(1, k + 1):
        if sim.done:
            a = np.zeros(2, np.float32)
        else:
            a = expert_action(sim)
            last = W.pano_tensor(sim.step(a)[0])
        acts.append(a)
        if i in future:
            frames[i] = last
    return np.array(acts, np.float32), np.array([frames[i] for i in future], np.float16)


def rollout(seed: int, policy=None, beta: float = 1.0, noise: float = 0.0, rng_seed: int | None = None,
            chunk: tuple | None = None):
    """Run one episode. Executed action: expert(+noise) w.p. beta else policy. Labels are always expert."""
    rng = np.random.default_rng(seed + 7919 if rng_seed is None else rng_seed)
    env = W.NavEnv(seed)
    obs = env.observe()
    if policy is not None and hasattr(policy, "reset"):
        policy.reset(env)
    rec = {k: [] for k in ("pano", "mask", "act", "expert", "dist", "coll", "pose", "prev")}
    if chunk is not None:
        rec["chunk"], rec["future"] = [], []
    prev = np.zeros(2, np.float32)
    burst = 0
    while not env.done:
        a_exp = expert_action(env)
        if policy is not None and rng.random() > beta:
            a = np.asarray(policy.act(obs), np.float32)
        else:
            a = a_exp.copy()
            if noise > 0:
                if burst > 0:
                    a = burst_a
                    burst -= 1
                elif rng.random() < 0.04 * noise:
                    burst = int(rng.integers(2, 6))
                    burst_a = np.array([rng.uniform(0, 0.5), rng.uniform(-1.6, 1.6)], np.float32)
                    a = burst_a
                else:
                    a = a + rng.normal(0, [0.12 * noise, 0.5 * noise]).astype(np.float32)
        rec["pano"].append(W.pano_tensor(obs).astype(np.float16))
        rec["mask"].append(obs.pano_label == W.OBJ0 + env.target)
        rec["expert"].append(a_exp)
        rec["dist"].append(env.geodesic(*env.pose[:2]))
        rec["pose"].append(env.pose.copy())
        rec["prev"].append(prev)
        if chunk is not None:
            ca, cf = expert_chunk(env, chunk[0], chunk[1])
            rec["chunk"].append(ca)
            rec["future"].append(cf)
        a = np.clip(a, [W.V_RANGE[0], W.W_RANGE[0]], [W.V_RANGE[1], W.W_RANGE[1]]).astype(np.float32)
        obs, collided = env.step(a)
        rec["act"].append(a)
        rec["coll"].append(collided)
        prev = a
    out = {k: np.array(v) for k, v in rec.items()}
    out["tok"] = np.repeat(tokenize(env.instruction)[None], len(out["act"]), 0)
    out["ep"] = np.full(len(out["act"]), seed)
    out["success"] = env.success
    return out


def merge(eps):
    keys = [k for k in eps[0] if k != "success"]
    return {k: np.concatenate([e[k] for e in eps]) for k in keys}
