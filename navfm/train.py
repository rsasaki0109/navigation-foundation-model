"""Training loops for the four paradigms (CPU friendly, a few minutes each)."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F

from . import models as M


def _batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n - bs + 1, bs):
        yield idx[i : i + bs]


def _log(tag, ep, loss, t0):
    print(f"[{tag}] epoch {ep} loss {loss:.4f} ({time.time() - t0:.0f}s)", flush=True)


def _roll(x, s):
    return torch.roll(x, s, -1)


def train_vlm(d, epochs=6, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = M.GroundingVLM()
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    pano, tok, mask = torch.tensor(d["pano"]).float(), torch.tensor(d["tok"]), torch.tensor(d["mask"]).float()
    t0 = time.time()
    for ep in range(epochs):
        tot = []
        for b in _batches(len(pano), 256, rng):
            s = int(rng.integers(128))  # rotation augmentation: roll image and labels together
            logit = m(_roll(pano[b], s), tok[b])
            loss = F.binary_cross_entropy_with_logits(logit, _roll(mask[b], s), pos_weight=torch.tensor(3.0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot.append(loss.item())
        _log("vlm", ep, np.mean(tot), t0)
    return m


def train_vla(d, epochs=6, seed=0, model=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = model or M.VLA()
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    pano, tok = torch.tensor(d["pano"]).float(), torch.tensor(d["tok"])
    prev = torch.tensor(d["prev"]).float()
    y = torch.tensor(M.tokenize_action(d["expert"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * (len(pano) // 256))
    t0 = time.time()
    for ep in range(epochs):
        tot = []
        for b in _batches(len(pano), 256, rng):
            p = prev[b] * (torch.rand(len(b), 1) > 0.5) + 0.05 * torch.randn(len(b), 2)
            loss = F.cross_entropy(m(pano[b], tok[b], p), y[b], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot.append(loss.item())
        _log("vla", ep, np.mean(tot), t0)
    return m


def wm_windows(d, horizon):
    ep = d["ep"]
    n = len(ep)
    starts = np.arange(n - horizon)
    ok = ep[starts] == ep[starts + horizon]
    return starts[ok]


def train_wm(d, epochs=4, horizon=8, seed=0, model=None, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = model or M.WorldModel()
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-4)
    pano, tok = torch.tensor(d["pano"]).float(), torch.tensor(d["tok"])
    act = torch.tensor(d["act"]).float()
    dist = torch.tensor(np.minimum(d["dist"], 12.0)).float()
    coll = torch.tensor(d["coll"]).float()
    starts = wm_windows(d, horizon)
    steps = epochs * (len(starts) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time()
    ks = torch.arange(horizon + 1)
    for ep in range(epochs):
        tot = []
        for bi in _batches(len(starts), 128, rng):
            s = torch.tensor(starts[bi])
            idx = s[:, None] + ks[None]  # (B, H+1)
            obs = pano[idx]  # (B, H+1, 5, 128)
            B = len(s)
            z_all = m.encode(obs.flatten(0, 1)).view(B, horizon + 1, -1)
            t = m.text(tok[s])
            z = z_all[:, 0]
            loss = F.mse_loss(m.decode(z), obs[:, 0]) + F.smooth_l1_loss(m.goal_dist(z, t), dist[idx[:, 0]])
            for k in range(horizon):
                a = act[idx[:, k]]
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(m.collision(z, a), coll[idx[:, k]], pos_weight=torch.tensor(4.0)) / horizon
                z = m.step(z, a)
                loss = loss + (
                    F.mse_loss(z, z_all[:, k + 1].detach()) * 2.0
                    + F.mse_loss(m.decode(z), obs[:, k + 1])
                    + F.smooth_l1_loss(m.goal_dist(z, t), dist[idx[:, k + 1]])
                ) / horizon
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot.append(loss.item())
        _log("wm", ep, np.mean(tot), t0)
    return m


def train_wam(d, epochs=8, seed=0, model=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = model or M.WAM()
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    pano, tok = torch.tensor(d["pano"]).float(), torch.tensor(d["tok"])
    prev = torch.tensor(d["prev"]).float()
    chunk, fut = torch.tensor(d["chunk"]).float(), torch.tensor(d["future"]).float()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * (len(pano) // 256))
    wk = torch.linspace(1.0, 0.4, M.WAM.CHUNK)[None, :, None]
    t0 = time.time()
    for ep in range(epochs):
        tot = []
        for b in _batches(len(pano), 256, rng):
            p = prev[b] * (torch.rand(len(b), 1) > 0.5) + 0.05 * torch.randn(len(b), 2)
            a_hat, f_hat = m(pano[b], tok[b], p)
            loss = (F.smooth_l1_loss(a_hat, chunk[b], reduction="none", beta=0.1) * wk).mean() * 4 + F.mse_loss(f_hat, fut[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot.append(loss.item())
        _log("wam", ep, np.mean(tot), t0)
    return m
