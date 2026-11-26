"""Ground a text target in a live 3DGS camera and approach it in Rapier."""

import argparse
from collections import deque
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, distance_transform_edt, find_objects, label

from navfm import rne_scene as S
from navfm.splat_render import SplatRaster
from scripts.make_gifs import save_small_gif
from scripts.simulate_rgbd import CELL, TICKS, TRACK, WHEEL, create_sim, make_map, read_cloud, register_floor, route

MODEL = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
SIZE = (320, 240)
INTRINSICS = (170, 170, 160, 120)


def yaw_error(desired, current):
    return (desired - current + np.pi) % (2 * np.pi) - np.pi


def ground_target(box, depth, eye, basis):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    rows, cols = np.indices(depth.shape)
    mask = ((cols > x0 + .25*w) & (cols < x1 - .25*w) &
            (rows > y0 + .25*h) & (rows < y1 - .25*h) & (depth > 0))
    if mask.sum() < 12:
        return None
    fx, fy, cx, cy = INTRINSICS
    z = depth[mask]
    local = np.stack([(cols[mask] + .5 - cx) * z / fx, (rows[mask] + .5 - cy) * z / fy, z], axis=1)
    return eye + np.median(local, axis=0) @ basis


class Detector:
    def __init__(self, target):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.target = target
        self.processor = Owlv2Processor.from_pretrained(MODEL, revision=MODEL_REVISION, use_fast=False,
                                                        size={"height": 480, "width": 480})
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL, revision=MODEL_REVISION).eval()

    @torch.inference_mode()
    def detect(self, image):
        # The instruction specifies a blue screen. Color proposes candidate regions
        # and OWLv2 checks that a region resembles a screen.
        red, green, blue = np.moveaxis(np.asarray(image).astype(int), -1, 0)
        mask = (blue > 105) & (blue - green > 22) & (green - red > 9)
        components, _ = label(binary_closing(mask, iterations=2))
        sizes = np.bincount(components.ravel())
        regions = []
        for slot, slices in enumerate(find_objects(components), 1):
            if slices is None or sizes[slot] < 150:
                continue
            y, x = slices
            regions.append([x.start, y.start, x.stop, y.stop])
        self.last_proposals = regions
        if not regions:
            return 0, None
        inputs = self.processor(text=[[f"a photo of a {self.target}", "a computer monitor",
                                       "a laptop screen", "a desk", "a chair"]],
                                images=image, return_tensors="pt")
        result = self.processor.post_process_grounded_object_detection(
            self.model(**inputs, interpolate_pos_encoding=True),
            threshold=.004, target_sizes=[(image.height, image.width)])[0]
        best = (0, None)
        for score, kind, box in zip(result["scores"], result["labels"], result["boxes"]):
            if int(kind) >= 3 or float(score) <= best[0]:
                continue
            x0, y0, x1, y1 = box.tolist()
            for region in regions:
                a, b, c, d = region
                intersection = max(0, min(c, x1) - max(a, x0)) * max(0, min(d, y1) - max(b, y0))
                union = (c - a) * (d - b) + (x1 - x0) * (y1 - y0) - intersection
                if union > 0 and intersection / union > .24:
                    best = (float(score), region)
        return best


class Episode:
    def __init__(self, cloud, free, obstacles, origin, start, directory, *, capture_frames=True):
        self.cloud, self.free, self.obstacles, self.origin = cloud, free, obstacles, origin
        self.safe = distance_transform_edt(free) * CELL >= .13
        self.obstacle_distance = distance_transform_edt(~obstacles) * CELL
        self.collider_centers = origin + (np.argwhere(obstacles) + .5) * CELL
        self.min_collider_clearance = float("inf")
        self.envelope_overlap_ticks = 0
        self.sim = create_sim(directory, obstacles, origin, start)
        self.obs = self.sim.reset()
        self.raster = SplatRaster(directory)
        self.capture_frames = capture_frames
        self.probe_images = []
        self.confirmations = []
        self.confirmation_steps = 0
        self.contact_pairs_available = hasattr(self.sim, "contact_pairs")
        self.contact_summary = {}
        self.active_obstacle_pairs = set()
        self.obstacle_contact_onsets = 0
        self.obstacle_contact_ticks = 0
        self.path, self.frames, self.probes = [], [], []
        self.max_contacts = 0
        pixels = np.full((*free.T.shape, 3), (31, 42, 52), dtype=np.uint8)
        pixels[free.T] = (110, 122, 136)
        pixels[obstacles.T] = (144, 77, 72)
        self.map = Image.fromarray(pixels).resize((200, 200), Image.Resampling.NEAREST)
        self.target_xy = self.approach = None

    @property
    def position(self):
        return np.array([self.obs.base_x, self.obs.base_z])

    def cell(self, p):
        return tuple(np.floor((p - self.origin) / CELL).astype(int))

    def center(self, cell):
        return self.origin + (np.asarray(cell) + .5) * CELL

    def step(self, v, w, phase):
        for _ in range(TICKS):
            self.obs = self.sim.step(float((v - w * TRACK/2) / WHEEL),
                                     float((v + w * TRACK/2) / WHEEL))
            self.max_contacts = max(self.max_contacts, self.sim.contact_count)
            if self.contact_pairs_available:
                self.track_contacts(self.sim.contact_pairs)
            # Distance from the center to the actual axis-aligned box footprints.
            gap = np.maximum(abs(self.collider_centers - self.position) - CELL / 2, 0)
            distance = float(np.linalg.norm(gap, axis=1).min(initial=np.inf))
            self.min_collider_clearance = min(self.min_collider_clearance, distance)
            self.envelope_overlap_ticks += distance < .10
        cell = self.cell(self.position)
        if not all(0 <= k < n for k, n in zip(cell, self.free.shape)) or not self.free[cell]:
            raise RuntimeError("Robot left the surveyed floor")
        self.path.append(dict(position=self.position.tolist(), yaw=self.obs.base_yaw, phase=phase,
                              obstacle_cell_clearance_m=float(self.obstacle_distance[cell])))

    def track_contacts(self, pairs):
        obstacle_pairs = set()
        for a, b, impulse in pairs:
            names = tuple(sorted((a, b)))
            if any(name.startswith("scan_") for name in names) and "ground" not in names:
                kind = "obstacle"
                if impulse > 0:
                    obstacle_pairs.add(names)
            elif "ground" in names:
                kind = "wheel_floor" if any("wheel" in name for name in names) else "other_floor"
            else:
                kind = "other"
            key = (kind, *names)
            record = self.contact_summary.setdefault(key, dict(kind=kind, entities=list(names),
                                                                reports=0, positive_impulse_reports=0,
                                                                max_impulse_ns=0.0))
            record["reports"] += 1
            record["positive_impulse_reports"] += impulse > 0
            record["max_impulse_ns"] = max(record["max_impulse_ns"], impulse)
        self.obstacle_contact_onsets += len(obstacle_pairs - self.active_obstacle_pairs)
        self.obstacle_contact_ticks += bool(obstacle_pairs)
        self.active_obstacle_pairs = obstacle_pairs

    def turn(self, yaw, phase="search"):
        for _ in range(40):
            err = yaw_error(yaw, self.obs.base_yaw)
            if abs(err) < .065:
                return
            self.step(0, np.clip(3*err, -1.6, 1.6), phase)
        raise RuntimeError("Robot did not complete the commanded turn")

    def observe(self):
        x, z = self.position
        yaw = self.obs.base_yaw
        eye = np.array([x, .95, z])
        rot = S.look_rot(eye, eye + [np.cos(yaw), -.16, -np.sin(yaw)])
        basis = S._quat_mat(rot).T * np.array([[1], [-1], [-1]])
        image, depth = self.raster.render(self.cloud, eye, basis, INTRINSICS, SIZE, with_depth=True)
        return image, depth, eye, basis

    def reachable(self):
        start = self.cell(self.position)
        parents = {start: None}
        queue = deque([start])
        while queue:
            i, j = current = queue.popleft()
            for c in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if all(0 <= k < n for k, n in zip(c, self.safe.shape)) and self.safe[c] and c not in parents:
                    parents[c] = current
                    queue.append(c)
        return parents

    def move_to(self, parents, goal, phase):
        cells = []
        while parents[goal] is not None:
            cells.append(goal)
            goal = parents[goal]
        for c in reversed(cells):
            target = self.center(c)
            for _ in range(35):
                delta = target - self.position
                distance = np.linalg.norm(delta)
                if distance < .025:
                    break
                angle = yaw_error(np.arctan2(-delta[1], delta[0]), self.obs.base_yaw)
                self.step(min(.16, 2*distance) if abs(angle) < .3 else 0,
                          np.clip(3*angle, -1.6, 1.6), phase)
                if self.capture_frames and len(self.path) % 4 == 0:
                    self.add_frame(self.observe()[0], phase)
            else:
                raise RuntimeError(f"Could not reach {phase} waypoint")

    def add_frame(self, image, phase, box=None):
        if not self.capture_frames:
            return
        panel = Image.new("RGB", (image.width + self.map.width, image.height + 26), "#1b2632")
        panel.paste(image, (0, 0))
        draw = ImageDraw.Draw(panel)
        if box is not None:
            draw.rectangle(box, outline="#77ebad", width=2)
        panel.paste(self.map, (image.width, 0))
        scale = np.asarray(self.map.size) / self.free.shape

        def mark(position, radius, color):
            x, y = (np.asarray(position) - self.origin) / CELL * scale + [image.width, 0]
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

        for state in self.path[::4]:
            mark(state["position"], 1, "#bc9970")
        if self.target_xy is not None:
            mark(self.target_xy, 4, "#77ebad")
        if self.approach is not None:
            mark(self.approach, 3, "#c2adf0")
        mark(self.position, 3, "#ffca63")
        draw.text((8, image.height + 7), phase, fill="white")
        draw.text((image.width + 8, 209), "Robot / screen / approach", fill="white")
        self.frames.append(panel)

    def probe(self, detector, threshold, phase):
        image, depth, eye, basis = self.observe()
        score, box = detector.detect(image)
        hit = ground_target(box, depth, eye, basis) if box is not None and score >= threshold else None
        self.probe_images.append(image)
        self.probes.append(dict(phase=phase, blue_proposals=detector.last_proposals,
                                position=self.position.tolist(), yaw=self.obs.base_yaw,
                                score=score, box=box, grounded=None if hit is None else hit.tolist()))
        return image, box, score, hit

    def confirm(self, detector, threshold, candidate):
        first = self.position
        first_probe = len(self.probes)
        direction = candidate[[0, 2]] - first
        parents = self.reachable()
        lateral = np.array([-direction[1], direction[0]])
        norm = np.linalg.norm(lateral)
        if norm < 1e-6:
            self.confirmations.append(dict(candidate_probe=first_probe, reason="degenerate sightline"))
            return None
        lateral /= norm
        options = []
        for cell in parents:
            shift = self.center(cell) - first
            if .12 <= np.linalg.norm(shift) <= .25:
                options.append((abs(shift @ lateral), cell))
        if not options:
            self.confirmations.append(dict(candidate_probe=first_probe, reason="no safe second viewpoint"))
            return None
        _, waypoint = max(options)
        start_steps = len(self.path)
        try:
            self.move_to(parents, waypoint, "moving for second view")
            displacement = float(np.linalg.norm(self.position - first))
            if displacement < .10:
                self.confirmations.append(dict(candidate_probe=first_probe, reason="camera baseline too short",
                                               displacement_m=displacement))
                return None
            delta = candidate[[0, 2]] - self.position
            self.turn(np.arctan2(-delta[1], delta[0]), "aiming second view")
            image, box, score, observed = self.probe(detector, threshold, "confirmation")
            error = None if observed is None else float(np.linalg.norm(candidate - observed))
            accepted = error is not None and error <= .15
            reason = "confirmed" if accepted else ("missing second detection" if error is None else "position mismatch")
            self.confirmations.append(dict(candidate_probe=first_probe, reason=reason, displacement_m=displacement,
                                           position_error_m=error, probe=len(self.probes)))
            self.add_frame(image, f"{reason} | {score:.3f}", box)
            return (candidate + observed) / 2 if accepted else None
        finally:
            self.confirmation_steps += len(self.path) - start_steps

    def search(self, detector, threshold, *, confirm_views=False):
        anchor = self.position
        parents = self.reachable()
        endpoint = self.center(max(parents, key=lambda c: np.linalg.norm(self.center(c) - anchor)))
        for station in range(3):
            if station:
                waypoint = anchor + (.65 if station == 1 else 1.0) * (endpoint - anchor)
                parents = self.reachable()
                goal = min(parents, key=lambda c: np.linalg.norm(self.center(c) - waypoint))
                self.move_to(parents, goal, "exploring surveyed floor")
            initial_yaw = self.obs.base_yaw
            for n in range(8):
                self.turn(initial_yaw + n * np.pi / 4)
                image, box, score, hit = self.probe(detector, threshold, "search")
                self.add_frame(image, f"{detector.target}: {'candidate' if hit is not None else 'searching'} | {score:.3f}",
                               box if hit is not None else None)
                if hit is not None:
                    if confirm_views:
                        hit = self.confirm(detector, threshold, hit)
                    if hit is not None:
                        self.target_xy = hit[[0, 2]]
                        return hit
        return None


def approach(episode, detector, target, threshold):
    parents = episode.reachable()
    cell = min(parents, key=lambda c: np.linalg.norm(episode.center(c) - target[[0, 2]]))
    goal = episode.center(cell)
    episode.approach = goal
    episode.move_to(parents, cell, "approaching visual goal")
    error = float(np.linalg.norm(episode.position - goal))
    delta = target[[0, 2]] - episode.position
    episode.turn(np.arctan2(-delta[1], delta[0]), "facing visual target")
    image, _, _, _ = episode.observe()
    score, final_box = detector.detect(image)
    episode.add_frame(image, "reached mapped approach point",
                      final_box if score >= threshold else None)
    return goal, error, score, final_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=.012)
    parser.add_argument("--confirm-views", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cloud, _ = register_floor(read_cloud(args.source))
    free, obstacles, origin = make_map(cloud)
    _, start = route(free, origin)
    cloud[:, 9:] *= .72
    detector = Detector("blue computer screen")
    with tempfile.TemporaryDirectory(prefix="scan_nav_") as directory:
        episode = Episode(cloud, free, obstacles, origin, start, Path(directory))
        target = episode.search(detector, args.threshold, confirm_views=args.confirm_views)
        goal = None
        error = None
        if target is not None:
            goal, error, _, _ = approach(episode, detector, target, args.threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_small_gif(episode.frames, args.output, frame_ms=450)
        args.output.with_suffix(".json").write_text(json.dumps(dict(
            source=args.source.name, detector=MODEL, detector_revision=MODEL_REVISION,
            target=detector.target, threshold=args.threshold, confirm_views=args.confirm_views,
            confirmations=episode.confirmations,
            confirmation_seconds=episode.confirmation_steps * TICKS * episode.sim.fixed_delta_s,
            initial_position=start.tolist(), probes=episode.probes, path=episode.path,
            reached=error is not None and error < .05, goal=None if goal is None else goal.tolist(),
            goal_error_before_final_turn_m=error,
            final_position=episode.position.tolist(), max_contacts_per_tick=episode.max_contacts,
            min_collider_clearance_m=episode.min_collider_clearance,
            obstacle_contact_onsets=episode.obstacle_contact_onsets if episode.contact_pairs_available else None,
            contact_pairs=list(episode.contact_summary.values()) if episode.contact_pairs_available else None,
            footprint_envelope_m=.10, footprint_envelope_overlap_ticks=episode.envelope_overlap_ticks,
            min_obstacle_cell_clearance_m=min((step["obstacle_cell_clearance_m"] for step in episode.path),
                                              default=None),
        ), indent=2) + "\n")
        print("goal", goal, "final", episode.position, flush=True)


if __name__ == "__main__":
    main()
