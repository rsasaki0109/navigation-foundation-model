"""Evaluate the frozen blue-screen policy at five starts and two headings."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from scripts.navigate_rgbd import MODEL, MODEL_REVISION, Detector, Episode, approach
from scripts.simulate_rgbd import CELL, TICKS, make_map, read_cloud, register_floor, route

THRESHOLD = .012
# Reference annotation from the visually inspected original replay. Evaluation only.
SCREEN_XZ = np.array([-.29629325, -1.07850050])
LOCALIZATION_LIMIT_M = .25
APPROACH_LIMIT_M = .85


def starts(free, origin):
    a, b = route(free, origin)
    clearance = distance_transform_edt(free) * CELL >= .13
    cells = np.argwhere(clearance)
    centers = origin + (cells + .5) * CELL
    return [centers[np.argmin(np.linalg.norm(centers - p, axis=1))] for p in np.linspace(b, a, 5)]


def trial(episode, detector, start, yaw, confirm_views):
    fault = None
    target = goal = None
    actual_yaw = None
    try:
        episode.turn(yaw, "initial alignment")
        actual_yaw = float(episode.obs.base_yaw)
        target = episode.search(detector, THRESHOLD, confirm_views=confirm_views)
        if target is not None:
            goal, _, _, _ = approach(episode, detector, target, THRESHOLD)
    except RuntimeError as exc:
        fault = str(exc)
    final = episode.position
    target_error = None if target is None else float(np.linalg.norm(target[[0, 2]] - SCREEN_XZ))
    correctly_localized = target_error is not None and target_error <= LOCALIZATION_LIMIT_M
    goal_error = None if goal is None else float(np.linalg.norm(final - goal))
    screen_distance = float(np.linalg.norm(final - SCREEN_XZ))
    reached = fault is None and correctly_localized and goal_error is not None and goal_error < .05
    reached = reached and screen_distance <= APPROACH_LIMIT_M
    return dict(start=start.tolist(), requested_yaw_rad=yaw, camera_start_yaw_rad=actual_yaw,
                detected=target is not None, correctly_localized=correctly_localized, reached=reached,
                initial_hit=bool(episode.probes and episode.probes[0]["grounded"] is not None),
                target_error_m=target_error, goal_error_m=goal_error, final_screen_distance_m=screen_distance,
                final_position=final.tolist(), obstacle_contact_onsets=episode.obstacle_contact_onsets,
                obstacle_contact_ticks=episode.obstacle_contact_ticks, fault=fault,
                simulated_seconds=len(episode.path) * TICKS * episode.sim.fixed_delta_s,
                confirmation_seconds=episode.confirmation_steps * TICKS * episode.sim.fixed_delta_s,
                confirmations=episode.confirmations,
                contacts=list(episode.contact_summary.values()), probes=episode.probes)


def diagnostics(rows, output):
    # Retain all probe metadata and a contact sheet of the most informative images.
    panes = []
    for number, (record, images) in enumerate(rows, 1):
        if not images:
            continue
        probes = record["probes"]
        ranked = sorted(range(len(probes)),
                        key=lambda i: (probes[i]["grounded"] is not None, probes[i]["score"],
                                       len(probes[i]["blue_proposals"])), reverse=True)
        selected = list(dict.fromkeys([0, *ranked[:3]])) if not record["reached"] else [0, ranked[0]]
        for i in selected:
            frame = images[i].copy()
            draw = ImageDraw.Draw(frame)
            for box in probes[i]["blue_proposals"]:
                draw.rectangle(box, outline="#ffcf70")
            if probes[i]["grounded"] is not None:
                draw.rectangle(probes[i]["box"], outline="#70e6ad", width=2)
            draw.rectangle((0, 216, 320, 240), fill="#172431")
            draw.text((6, 222), f"Run {number} / probe {i+1} / {probes[i]['score']:.3f}"
                                f" / {'reached' if record['reached'] else 'failed'}", fill="white")
            panes.append(frame)
    if panes:
        sheet = Image.new("RGB", (960, 240 * ((len(panes) + 2) // 3)), "#172431")
        for i, frame in enumerate(panes):
            sheet.paste(frame, (320 * (i % 3), 240 * (i // 3)))
        sheet.save(output, quality=86)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confirm-views", action="store_true")
    parser.add_argument("--unseen", action="store_true", help="use held-out floor cells and headings")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cloud, _ = register_floor(read_cloud(args.source))
    free, obstacles, origin = make_map(cloud)
    cloud[:, 9:] *= .72
    baseline_starts = starts(free, origin)
    if args.unseen:
        cells = np.argwhere(distance_transform_edt(free) * CELL >= .13)
        centers = origin + (cells + .5) * CELL
        selected = list(baseline_starts)
        for wanted in baseline_starts:
            remaining = [p for p in centers if all(np.linalg.norm(p - used) > .049 for used in selected)]
            selected.append(min(remaining, key=lambda p: np.linalg.norm(p - wanted)))
        poses = [(p, yaw) for p in selected[5:] for yaw in (-np.pi / 2, np.pi / 2)]
    else:
        poses = [(p, yaw) for p in baseline_starts for yaw in (0.0, np.pi)]
    if len({(*position, yaw) for position, yaw in poses}) != 10:
        raise RuntimeError("Expected ten distinct start poses")
    detector = Detector("blue computer screen")
    rows = []
    for start, yaw in poses:
        with tempfile.TemporaryDirectory(prefix="rgbd_benchmark_") as tmp:
            episode = Episode(cloud, free, obstacles, origin, start, Path(tmp), capture_frames=False)
            if not episode.contact_pairs_available:
                raise RuntimeError("The benchmark requires an rne_py wheel with DiffDriveSim.contact_pairs")
            record = trial(episode, detector, start, yaw, args.confirm_views)
            rows.append((record, episode.probe_images))
            print(len(rows), "detected", record["detected"], "reached", record["reached"],
                  "obstacle contact onsets", record["obstacle_contact_onsets"], "fault", record["fault"], flush=True)
    records = [record for record, _ in rows]
    summary = dict(runs=len(records), detections=sum(r["detected"] for r in records),
                   correct_localizations=sum(r["correctly_localized"] for r in records),
                   reaches=sum(r["reached"] for r in records),
                   runs_with_obstacle_contacts=sum(r["obstacle_contact_onsets"] > 0 for r in records),
                   obstacle_contact_onsets=sum(r["obstacle_contact_onsets"] for r in records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, source=args.source.name, detector=MODEL,
                                          detector_revision=MODEL_REVISION, threshold=THRESHOLD,
                                          confirm_views=args.confirm_views, unseen_starts=args.unseen,
                                          reference_screen_xz_m=SCREEN_XZ.tolist(),
                                          localization_limit_m=LOCALIZATION_LIMIT_M,
                                          approach_limit_m=APPROACH_LIMIT_M, trials=records),
                                    indent=2) + "\n")
    diagnostics(rows, args.output.with_suffix(".jpg"))
    print(summary)


if __name__ == "__main__":
    main()
