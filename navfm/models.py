"""Small PyTorch models for the four paradigms. All share a panoramic conv encoder + text encoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import world as W
from .text import MAX_TOKENS, VOCAB

C_IN = 5
D = 64

# discrete action tokens for the VLA (RT-style action tokenization)
V_BINS = np.array([0.0, 0.25, 0.55], np.float32)
W_BINS = np.array([-1.6, -0.8, -0.3, 0.0, 0.3, 0.8, 1.6], np.float32)
ACTION_TOKENS = np.array([(v, w) for v in V_BINS for w in W_BINS], np.float32)
N_TOKENS = len(ACTION_TOKENS)


def tokenize_action(a: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(a)
    d = (a[:, None, 0] - ACTION_TOKENS[None, :, 0]) ** 2 + 0.15 * (a[:, None, 1] - ACTION_TOKENS[None, :, 1]) ** 2
    return d.argmin(1)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), 32, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(32, D), nn.ReLU(), nn.Linear(D, D))

    def forward(self, tok):
        m = (tok > 0).float().unsqueeze(-1)
        e = (self.emb(tok) * m).sum(1) / m.sum(1).clamp(min=1)
        return self.mlp(e)


class CConv(nn.Module):
    """Circular 1D conv block (the panorama wraps around)."""

    def __init__(self, cin, cout, k=5, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, stride, padding=k // 2, padding_mode="circular")
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class FiLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.lin = nn.Linear(D, 2 * c)

    def forward(self, x, t):
        g, b = self.lin(t).chunk(2, -1)
        return x * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)


class PanoEncoder(nn.Module):
    """(B, 5, 128) -> (B, D, 32) feature map; optional text FiLM conditioning."""

    def __init__(self, text_cond: bool):
        super().__init__()
        self.c1, self.c2 = CConv(C_IN, 48), CConv(48, D, stride=2)
        self.c3, self.c4 = CConv(D, D), CConv(D, D, stride=2)
        self.film = FiLM(48) if text_cond else None
        self.film2 = FiLM(D) if text_cond else None

    def forward(self, x, t=None, full_res=False):
        h1 = self.c1(x)
        if self.film is not None:
            h1 = self.film(h1, t)
        h = self.c3(self.c2(h1))
        if self.film2 is not None:
            h = self.film2(h, t)
        return (h1, self.c4(h)) if full_res else self.c4(h)


# ---------------------------------------------------------------- VLM + planning
class GroundingVLM(nn.Module):
    """Language-grounded detector: per-column logits 'is the instructed object here?'."""

    def __init__(self):
        super().__init__()
        self.text = TextEncoder()
        self.enc = PanoEncoder(text_cond=True)
        self.up = nn.Sequential(CConv(D + 48, 48), nn.Conv1d(48, 1, 1))

    def forward(self, pano, tok):
        t = self.text(tok)
        h1, h = self.enc(pano, t, full_res=True)
        h = F.interpolate(h, size=pano.shape[-1], mode="linear", align_corners=False)
        return self.up(torch.cat([h, h1], 1)).squeeze(1)  # (B, 128)


# ---------------------------------------------------------------- VLA
class VLA(nn.Module):
    """Pixels + words (+ previous action) -> distribution over discrete action tokens."""

    def __init__(self):
        super().__init__()
        self.text = TextEncoder()
        self.enc = PanoEncoder(text_cond=True)
        self.head = nn.Sequential(
            nn.Linear(D * 32 + D + 2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, N_TOKENS)
        )

    def forward(self, pano, tok, prev_a):
        t = self.text(tok)
        h = self.enc(pano, t).flatten(1)
        return self.head(torch.cat([h, t, prev_a], -1))


# ---------------------------------------------------------------- WM + planning
class WorldModel(nn.Module):
    """Latent world model: encoder, action-conditioned dynamics, decoder, goal-distance and collision heads."""

    Z = 96

    def __init__(self):
        super().__init__()
        self.enc = PanoEncoder(text_cond=False)
        self.to_z = nn.Linear(D * 32, self.Z)
        self.dyn = nn.Sequential(nn.Linear(self.Z + 2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, self.Z))
        self.dec = nn.Sequential(nn.Linear(self.Z, 256), nn.GELU(), nn.Linear(256, C_IN * W.PANO_RAYS))
        self.text = TextEncoder()
        self.dist = nn.Sequential(nn.Linear(self.Z + D, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        self.coll = nn.Sequential(nn.Linear(self.Z + 2, 128), nn.GELU(), nn.Linear(128, 1))
        # policy prior (TD-MPC style): proposes action sequences for the planner to refine
        self.pi = nn.Sequential(nn.Linear(self.Z + D, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2))

    def prior(self, z, t):
        lo = torch.tensor([W.V_RANGE[0], W.W_RANGE[0]])
        hi = torch.tensor([W.V_RANGE[1], W.W_RANGE[1]])
        return lo + (hi - lo) * torch.sigmoid(self.pi(torch.cat([z, t], -1)))

    def encode(self, pano):
        return torch.tanh(self.to_z(self.enc(pano).flatten(1)))

    def step(self, z, a):
        return torch.tanh(torch.atanh(z.clamp(-0.999, 0.999)) + self.dyn(torch.cat([z, a], -1)))

    def decode(self, z):
        return self.dec(z).view(-1, C_IN, W.PANO_RAYS)

    def goal_dist(self, z, t):
        return F.softplus(self.dist(torch.cat([z, t], -1))).squeeze(-1)

    def collision(self, z, a):
        return self.coll(torch.cat([z, a], -1)).squeeze(-1)


# ---------------------------------------------------------------- WAM
class WAM(nn.Module):
    """World-action model: one trunk jointly predicts an action chunk and the future panoramas it leads to."""

    CHUNK = 8
    FUTURE = (2, 4, 6, 8)  # predicted future frames (steps ahead)

    def __init__(self):
        super().__init__()
        self.text = TextEncoder()
        self.enc = PanoEncoder(text_cond=True)
        self.trunk = nn.Sequential(nn.Linear(D * 32 + D + 2, 384), nn.GELU(), nn.Linear(384, 384), nn.GELU())
        self.act = nn.Linear(384, self.CHUNK * 2)
        self.fut = nn.Sequential(nn.Linear(384, 512), nn.GELU(), nn.Linear(512, len(self.FUTURE) * C_IN * W.PANO_RAYS))

    def forward(self, pano, tok, prev_a):
        t = self.text(tok)
        h = self.trunk(torch.cat([self.enc(pano, t).flatten(1), t, prev_a], -1))
        acts = self.act(h).view(-1, self.CHUNK, 2)
        fut = self.fut(h).view(-1, len(self.FUTURE), C_IN, W.PANO_RAYS)
        return acts, fut
