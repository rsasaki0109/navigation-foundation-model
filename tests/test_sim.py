"""Simulator, expert and scene-export tests (fast, no trained models needed)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navfm import rne_scene as S  # noqa: E402
from navfm import world as W  # noqa: E402
from navfm.expert import expert_action  # noqa: E402
from navfm.text import tokenize  # noqa: E402


def test_layout_is_deterministic():
    a, b = W.NavEnv(7), W.NavEnv(7)
    assert np.array_equal(a.layout.grid, b.layout.grid)
    assert np.allclose(a.pose, b.pose) and a.instruction == b.instruction


def test_observation_shapes():
    env = W.NavEnv(3)
    obs = env.observe()
    assert obs.pano_rgb.shape == (W.PANO_RAYS, 3)
    assert W.pano_tensor(obs).shape == (5, W.PANO_RAYS)
    assert (obs.pano_low <= obs.pano_depth + 1e-9).all()


def test_expert_reaches_goal_without_collisions():
    for seed in range(5):
        env = W.NavEnv(seed)
        while not env.done:
            env.step(expert_action(env))
        assert env.success and env.collisions == 0, seed


def test_blocked_move_slides_and_counts_collision():
    env = W.NavEnv(0)
    # a free spot next to the outer wall at x=0, heading into it at a slant
    y = next(y for y in np.arange(1.0, 5.0, 0.05) if env.free(0.32, y) and env.free(0.32, y + 0.4))
    env.pose = np.array([0.32, y, np.pi * 0.8])
    for _ in range(4):
        env.step(np.array([0.6, 0.0]))
    assert env.collisions > 0
    assert env.free(env.pose[0], env.pose[1])
    assert env.pose[1] > y + 0.05  # slid along the wall instead of stopping


def test_tokenizer_pads():
    t = tokenize("go to the red chair")
    assert t.shape == (8,) and (t[:5] > 1).all() and (t[5:] == 0).all()


def test_rne_scene_export_is_valid_json(tmp_path):
    env = W.NavEnv(35)
    items = S.static_items(env.layout) + S.robot_items(env.pose, (255, 170, 60))
    views = [S.overview_view("ov.png"), S.fpv_view(env.pose, "fpv.png")]
    text = json.dumps(dict(static_items=items, frames=[dict(items=[], views=views)]))
    for it in json.loads(text)["static_items"]:
        assert it["shape"] in ("box", "sphere", "cylinder", "mesh")
        if it["shape"] == "mesh":
            assert Path(it["path"]).exists() and all(v > 0 for v in it["scale"])
    q = np.array(views[1]["rot"])
    assert abs(np.linalg.norm(q) - 1) < 1e-6


def test_readme_media_exist():
    import re

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    paths = set(re.findall(r'src="(docs/media/[^"]+)"', readme))
    assert paths, "README references no media"
    for p in paths:
        assert (root / p).exists(), p
