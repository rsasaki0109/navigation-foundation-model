"""Rollout collection: expert with DART noise, or DAgger mixtures with a learner."""

from __future__ import annotations

import copy
import os

import numpy as np

from . import world as W
from .expert import expert_action
from .text import tokenize


def layout_for(seed: int) -> str:
    """Training layout from NAVFM_LAYOUT: 'apartment' (default), 'house', or 'mixed' (alternate by seed)."""
    mode = os.environ.get("NAVFM_LAYOUT", "apartment")
    if mode == "mixed":
        return "house" if (seed // 2) % 2 else "apartment"
    return mode


def make_env(seed: int) -> W.NavEnv:
    return W.NavEnv(seed, layout_for(seed))


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
    env = make_env(seed)
    obs = env.observe()
    if policy is not None and hasattr(policy, "reset"):
        policy.reset(env)
    rec = {k: [] for k in ("pano", "mask", "act", "expert", "dist", "tdist", "coll", "pose", "prev")}
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
        rec["tdist"].append(env.target_distance())
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


def counterfactual(seed: int, n_starts: int, horizon: int, rng_seed: int, prefix: int = 0):
    """Planner-style rollouts in the true simulator from states an expert visits.

    From `n_starts` states along a noisy expert episode, execute action sequences of the kind
    CEM proposes (smooth random sequences or perturbed expert actions) and record what really
    happens.  Training the world model on these teaches it the consequences of the very action
    sequences the planner will search over, instead of only those of the expert.  With `prefix`,
    each sequence starts with the `prefix` real steps that led to the branch point (memory context).
    """
    rng = np.random.default_rng(rng_seed)
    base = rollout(seed, noise=1.0, rng_seed=rng_seed)
    env = make_env(seed)
    tok = tokenize(env.instruction)
    out = []
    n = len(base["pose"])
    choices = np.arange(max(prefix, 0), n) if n > prefix else np.arange(n)
    for k, idx in enumerate(rng.choice(choices, size=min(n_starts, len(choices)), replace=False)):
        env.pose = base["pose"][idx].copy()
        env.t, env.done, env.success, env.collisions, env.path = 0, False, False, 0, [env.pose.copy()]
        mode = rng.random()
        mean = np.array([rng.uniform(0.0, 0.6), rng.uniform(-1.6, 1.6)], np.float32)
        rec = {k2: [] for k2 in ("pano", "act", "expert", "dist", "tdist", "coll")}
        obs = env.observe()
        for _t in range(horizon + 1):
            a_exp = expert_action(env)
            if mode < 0.6:
                a = mean + rng.normal(0, [0.1, 0.4]).astype(np.float32)
            else:
                a = a_exp + rng.normal(0, [0.15, 0.6]).astype(np.float32)
            a = np.clip(a, [W.V_RANGE[0], W.W_RANGE[0]], [W.V_RANGE[1], W.W_RANGE[1]]).astype(np.float32)
            rec["pano"].append(W.pano_tensor(obs).astype(np.float16))
            rec["expert"].append(a_exp)
            rec["dist"].append(env.geodesic(*env.pose[:2]))
            rec["tdist"].append(env.target_distance())
            env.done = False  # keep simulating past the goal: the model must know what overshooting looks like
            obs, collided = env.step(a)
            rec["act"].append(a)
            rec["coll"].append(collided)
        o = {k2: np.array(v) for k2, v in rec.items()}
        if prefix < 0:  # full history from the episode start
            sl = slice(0, idx)
            for k2 in o:
                o[k2] = np.concatenate([base[k2][sl], o[k2]])
        elif prefix and idx >= prefix:
            sl = slice(idx - prefix, idx)
            for k2 in o:
                o[k2] = np.concatenate([base[k2][sl], o[k2]])
        L = len(o["act"])
        o["tok"] = np.repeat(tok[None], L, 0)
        o["ep"] = np.full(L, seed * 1000 + k)
        out.append(o)
    return out
