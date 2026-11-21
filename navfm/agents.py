"""Deployable agents for the four paradigms.  Each exposes reset(env), act(obs) and a `viz` dict."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from . import models as M
from . import world as W
from .expert import dijkstra, follow_field, pursue, safe_pursue
from .text import tokenize


def _t(x):
    return torch.as_tensor(np.asarray(x))[None]


class Agent:
    name = "agent"

    def reset(self, env):
        self.tok = _t(tokenize(env.instruction))
        self.prev = np.zeros(2, np.float32)
        self.viz = {}
        self.text_feat = None


# ------------------------------------------------------------------ VLM + planning
class VLMPlannerAgent(Agent):
    """Grounding VLM localises the instructed object -> goal in an online map -> Dijkstra/A* + pure pursuit."""

    name = "VLM + planning"

    def __init__(self, model: M.GroundingVLM):
        self.model = model.eval()

    def reset(self, env):
        super().reset(env)
        self.occ = np.zeros((W.NX, W.NY), np.int8)  # 0 unknown, 1 free, 2 occupied
        self.goal = None
        self.goal_conf = 0.0
        self.field = None
        self.k = 0
        self.mode = "explore"
        self.last_pose = None

    def _update_map(self, obs):
        x, y, th = obs.pose
        ang = th + W.PANO_ANGLES
        for depth, mark in ((obs.pano_low, True), (obs.pano_depth, True)):
            r = np.arange(0, W.MAX_RANGE_M, W.RES_M * 0.8)[None, :]
            px = x + np.cos(ang)[:, None] * r
            py = y + np.sin(ang)[:, None] * r
            inside = (r < depth[:, None]) & (px > 0) & (py > 0) & (px < W.WIDTH_M) & (py < W.HEIGHT_M)
            i, j = W.to_cell(px[inside], py[inside])
            sel = self.occ[i, j] != 2
            self.occ[i[sel], j[sel]] = 1
            hit = depth < W.MAX_RANGE_M - 0.05
            hx = x + np.cos(ang[hit]) * (depth[hit] + 0.02)
            hy = y + np.sin(ang[hit]) * (depth[hit] + 0.02)
            ok = (hx > 0) & (hy > 0) & (hx < W.WIDTH_M) & (hy < W.HEIGHT_M)
            i, j = W.to_cell(hx[ok], hy[ok])
            self.occ[i, j] = 2

    @torch.no_grad()
    def act(self, obs):
        self._update_map(obs)
        # bumper: commanded forward motion but did not move -> mark the cell ahead as occupied
        if getattr(self, "last_pose", None) is not None and self.prev[0] > 0.05:
            if np.hypot(*(obs.pose[:2] - self.last_pose[:2])) < 0.01:
                for da in (-0.5, 0.0, 0.5):
                    bx = obs.pose[0] + np.cos(obs.pose[2] + da) * (W.ROBOT_RADIUS_M + 0.06)
                    by = obs.pose[1] + np.sin(obs.pose[2] + da) * (W.ROBOT_RADIUS_M + 0.06)
                    i, j = W.to_cell(np.clip(bx, 0, W.WIDTH_M - 0.01), np.clip(by, 0, W.HEIGHT_M - 0.01))
                    self.occ[i, j] = 2
                self.field = None
        self.last_pose = obs.pose.copy()
        logits = self.model(_t(W.pano_tensor(obs)), self.tok)[0].numpy()
        prob = 1 / (1 + np.exp(-logits))
        x, y, th = obs.pose
        self.viz["prob"] = prob
        conf = float(prob.max())
        if conf > 0.5:
            # strongest connected run of confident columns
            on = prob > 0.5
            lab, n = ndimage.label(np.concatenate([on, on]))
            best, score = None, -1
            for k in range(1, n + 1):
                idx = np.where(lab == k)[0]
                if idx[0] >= W.PANO_RAYS:
                    continue
                s = prob[idx % W.PANO_RAYS].sum()
                if s > score:
                    best, score = idx % W.PANO_RAYS, s
            ang = th + np.angle(np.sum(prob[best] * np.exp(1j * W.PANO_ANGLES[best])))
            depth = float(np.median(obs.pano_depth[best]))
            gx, gy = x + np.cos(ang) * depth, y + np.sin(ang) * depth
            if self.goal is None or conf >= self.goal_conf - 0.2:
                self.goal = np.array([gx, gy]) if self.goal is None else 0.5 * self.goal + 0.5 * np.array([gx, gy])
                self.goal_conf = max(conf, self.goal_conf * 0.98)
            self.viz["det_cols"] = best
        self.viz["goal"] = self.goal
        # plan on the online map (unknown treated as free, occupied inflated by robot radius)
        if self.k % 2 == 0 or self.field is None or (self.goal is not None and self.mode == "explore"):
            occ = self.occ == 2
            clear = ndimage.distance_transform_edt(~occ) * W.RES_M
            self.blocked = clear < W.ROBOT_RADIUS_M + 0.01
            free = clear >= W.ROBOT_RADIUS_M + 0.06
            cost = 1.0 + 1.5 * np.clip(0.45 - clear, 0, None) / 0.45
            if self.goal is not None:
                self.mode = "goal"
                gi, gj = W.to_cell(*self.goal)
                ii, jj = np.meshgrid(np.arange(W.NX), np.arange(W.NY), indexing="ij")
                seeds = (np.hypot(ii - gi, jj - gj) * W.RES_M < 0.42) & free
                if not seeds.any():
                    seeds = (np.hypot(ii - gi, jj - gj) * W.RES_M < 0.7) & free
            else:
                # frontier exploration: known-free cells bordering unknown space, away from the robot
                self.mode = "explore"
                known = ndimage.binary_closing(self.occ > 0, iterations=2)
                known_free = known & (self.occ != 2)
                frontier = known_free & ndimage.binary_dilation(~known, iterations=1)
                ii, jj = np.meshgrid(np.arange(W.NX), np.arange(W.NY), indexing="ij")
                ri, rj = W.to_cell(x, y)
                seeds = frontier & free & (np.hypot(ii - ri, jj - rj) * W.RES_M > 1.0)
                self.viz["frontier"] = seeds
            fld = dijkstra(free, seeds, cost)
            bad = ~np.isfinite(fld)
            if bad.all():
                fld = np.zeros_like(fld)
                bad[:] = False
            d, (a, b) = ndimage.distance_transform_edt(bad, return_indices=True)
            self.field = np.where(bad, fld[a, b] + d * W.RES_M * 2, fld)
        self.k += 1
        path = follow_field(self.field, x, y, 400)
        self.viz["path"] = path
        if self.goal is not None and len(path) < 3:
            a = pursue(obs.pose, self.goal, 0.2)
        elif len(path) < 2:
            a = np.array([0.0, 1.2], np.float32)
        else:
            a = safe_pursue(obs.pose, path[:12], self.blocked)
        self.viz["mode"] = self.mode
        self.viz["conf"] = conf
        self.viz["occ"] = self.occ
        self.prev = a
        return a


# ------------------------------------------------------------------ VLA
class VLAAgent(Agent):
    """End-to-end vision-language-action policy over 21 discrete action tokens."""

    name = "VLA"

    def __init__(self, model: M.VLA):
        self.model = model.eval()

    @torch.no_grad()
    def act(self, obs):
        logits = self.model(_t(W.pano_tensor(obs)), self.tok, _t(self.prev))[0]
        p = torch.softmax(logits, -1).numpy()
        tok = int(p.argmax())
        a = M.ACTION_TOKENS[tok].copy()
        self.viz = {"token_prob": p, "token": tok}
        self.prev = a
        return a


# ------------------------------------------------------------------ WM + planning
class WMPlannerAgent(Agent):
    """Model-predictive control: CEM over action sequences imagined in the learned latent world model."""

    name = "WM + planning"
    H, N, ITERS, ELITE = 10, 192, 4, 16
    TERM, GAMMA = 10.0, 1.0

    def __init__(self, model: M.WorldModel, seed: int = 0):
        self.model = model.eval()
        self.rng = torch.Generator().manual_seed(seed)

    def reset(self, env):
        super().reset(env)
        self.mean = torch.tensor([[0.3, 0.0]]).repeat(self.H, 1)
        self.rng.manual_seed(env.seed)
        with torch.no_grad():
            self.text_feat = self.model.text(self.tok)

    @torch.no_grad()
    def _rollout(self, z0, acts):
        n = acts.shape[0]
        z = z0.expand(n, -1)
        t = self.text_feat.expand(n, -1)
        cost = torch.zeros(n)
        zs, ds = [], []
        for k in range(self.H):
            c = torch.sigmoid(self.model.collision(z, acts[:, k]))
            z = self.model.step(z, acts[:, k])
            d = self.model.goal_dist(z, t)
            cost += (self.GAMMA**k) * d + 3.0 * c
            zs.append(z)
            ds.append(d)
        cost += self.TERM * ds[-1]
        return cost, zs, torch.stack(ds, 1)

    @torch.no_grad()
    def act(self, obs):
        z0 = self.model.encode(_t(W.pano_tensor(obs)))
        lo = torch.tensor([W.V_RANGE[0], W.W_RANGE[0]])
        hi = torch.tensor([W.V_RANGE[1], W.W_RANGE[1]])
        # unroll the policy prior in imagination: a proposal the planner then refines
        prior, z = [], z0
        for _k in range(self.H):
            a = self.model.prior(z, self.text_feat)
            prior.append(a[0])
            z = self.model.step(z, a)
        prior = torch.stack(prior)
        mean, std = 0.5 * (self.mean + prior), torch.tensor([[0.25, 0.9]]).repeat(self.H, 1)
        for it in range(self.ITERS):
            eps = torch.randn(self.N, self.H, 2, generator=self.rng)
            acts = torch.max(torch.min(mean + std * eps, hi), lo)
            acts[0] = mean
            acts[1] = prior
            cost, _, _ = self._rollout(z0, acts)
            elite = acts[cost.argsort()[: self.ELITE]]
            mean, std = elite.mean(0), elite.std(0) + 0.03
        cost, zs, ds = self._rollout(z0, acts)
        order = cost.argsort()
        best = acts[order[0]]
        imag = [self.model.decode(zs[k][order[0] : order[0] + 1])[0].numpy() for k in (1, 3, 5, 7) if k < len(zs)]
        self.viz = {
            "samples": acts[order[:: max(1, self.N // 48)]].numpy(),
            "sample_cost": cost[order[:: max(1, self.N // 48)]].numpy(),
            "best": best.numpy(),
            "imagined": imag,
            "pred_dist": float(ds[order[0], 0]),
        }
        a = best[0].numpy()
        self.mean = torch.cat([best[1:], best[-1:]], 0)
        self.prev = a
        return a


# ------------------------------------------------------------------ WAM
class WAMAgent(Agent):
    """World-action model: predicts an 8-step action chunk + the future it expects; executes the chunk head."""

    name = "WAM"
    EXEC = 2

    def __init__(self, model: M.WAM):
        self.model = model.eval()

    def reset(self, env):
        super().reset(env)
        self.queue = []

    @torch.no_grad()
    def act(self, obs):
        if not self.queue:
            acts, fut = self.model(_t(W.pano_tensor(obs)), self.tok, _t(self.prev))
            acts = acts[0].numpy()
            self.viz = {"chunk": acts, "future": fut[0].numpy(), "chunk_pose": obs.pose.copy()}
            self.queue = list(acts[: self.EXEC])
        a = np.asarray(self.queue.pop(0), np.float32)
        self.prev = a
        return a


def load_agent(kind: str, ckpt_dir: str = "checkpoints"):
    cls = {"vlm": (M.GroundingVLM, VLMPlannerAgent), "vla": (M.VLA, VLAAgent), "wm": (M.WorldModel, WMPlannerAgent),
           "wam": (M.WAM, WAMAgent)}[kind]
    model = cls[0]()
    model.load_state_dict(torch.load(f"{ckpt_dir}/{kind}.pt", weights_only=True))
    return cls[1](model)
