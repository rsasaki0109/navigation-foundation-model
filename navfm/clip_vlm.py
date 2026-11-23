"""Zero-shot grounding with a real pretrained vision-language model (CLIP ViT-B/32).

The robot looks around in N_VIEWS directions (camera crops rendered from its pose, by RNE when
available, else the built-in ray caster), CLIP scores each crop against the instruction text
and a few background captions, and the best-scoring direction becomes the goal bearing.  No
training on this simulator: the model never saw these scenes.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import render as R
from . import rne_scene as S
from . import world as W

MODEL_ID = "openai/clip-vit-base-patch32"
N_VIEWS = 8
VIEW_FOV = np.deg2rad(60)
NEGATIVES = ["a photo of a plain wall", "a photo of an empty floor", "a photo of a doorway"]


_COMMAND_WORDS = {"go", "to", "the", "find", "navigate", "head", "over", "take", "me", "where", "is", "a", "an"}


def caption(instruction: str) -> str:
    """Turn a command into a CLIP-style caption: 'take me to the red chair' -> 'a photo of a red chair'."""
    words = [w for w in instruction.lower().split() if w not in _COMMAND_WORDS]
    return "a photo of a " + " ".join(words)


class ClipGrounder:
    def __init__(self, renderer: str = "rne"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(MODEL_ID).eval()
        self.proc = CLIPProcessor.from_pretrained(MODEL_ID)
        self.renderer = renderer if (renderer != "rne" or S.rne_binary() is not None) else "raycast"
        self._static = None

    def views(self, lay: W.Layout, pose) -> list[Image.Image]:
        x, y, th = pose
        angles = [th + k * 2 * np.pi / N_VIEWS for k in range(N_VIEWS)]
        if self.renderer == "rne":
            if self._static is None or self._static[0] is not lay:
                self._static = (lay, S.static_items(lay))
            tmp = Path(tempfile.mkdtemp(prefix="clipviews_"))
            vs = []
            for k, a in enumerate(angles):
                eye = S.P(x + np.cos(a) * 0.1, y + np.sin(a) * 0.1, S.CAM_H)
                tgt = S.P(x + np.cos(a) * 2, y + np.sin(a) * 2, S.CAM_H - 0.1)
                vs.append(dict(out=f"v{k}.png", width=224, height=224, fov_y_rad=float(VIEW_FOV), eye_m=eye,
                               rot=S.look_rot(eye, tgt)))
            script = dict(clear_rgba=[0.07, 0.08, 0.1, 1.0], static_items=self._static[1], frames=[dict(items=[], views=vs)])
            (tmp / "s.json").write_text(json.dumps(script))
            subprocess.run([str(S.rne_binary()), str(tmp / "s.json"), str(tmp)], check=True, stdout=subprocess.DEVNULL)
            return [Image.open(tmp / f"v{k}.png").convert("RGB") for k in range(N_VIEWS)]
        old = R.FOV
        R.FOV = VIEW_FOV
        try:
            return [Image.fromarray(R.fpv(lay, (x, y, a))).resize((224, 224)) for a in angles]
        finally:
            R.FOV = old

    @torch.no_grad()
    def score(self, lay, pose, instruction):
        """Per-view probability that the instructed object is what the view shows (vs background)."""
        imgs = self.views(lay, pose)
        inp = self.proc(text=[caption(instruction)] + NEGATIVES, images=imgs, return_tensors="pt", padding=True)
        logits = self.model(**inp).logits_per_image  # (views, 1 + negatives), CLIP's own temperature
        p = logits.softmax(-1)[:, 0].numpy()
        return p, imgs
