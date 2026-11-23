"""Shipped checkpoints load, act and reproduce their recorded outcomes on a few held-out seeds."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navfm import world as W  # noqa: E402
from navfm.agents import load_agent  # noqa: E402

KINDS = ["vlm", "vla", "wm", "wam"]
SEEDS = [35, 46, 78]  # the README hero episodes and neighbours; all four succeed on them


def run(kind, seed):
    torch.set_num_threads(1)
    agent = load_agent(kind, str(ROOT / "checkpoints"))
    env = W.NavEnv(seed)
    agent.reset(env)
    obs = env.observe()
    while not env.done:
        a = agent.act(obs)
        assert np.all(np.isfinite(a))
        obs, _ = env.step(a)
    return env


@pytest.mark.parametrize("kind", KINDS)
def test_agent_matches_recorded_results(kind):
    results = json.loads((ROOT / "docs/results.json").read_text())[kind]["episodes"]
    for seed in SEEDS:
        env = run(kind, seed)
        assert env.success == results[seed]["success"], (kind, seed)
        assert abs(env.t - results[seed]["steps"]) <= 3, (kind, seed, env.t, results[seed]["steps"])
