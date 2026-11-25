"""Drive a small Rapier robot in the registered TUM desk reconstruction."""

import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, distance_transform_edt

from navfm import rne_scene as S
from navfm.splat_render import SplatRaster, _cloud
from scripts.make_gifs import save_small_gif

CELL = .05
WHEEL = .03
TRACK = .14
TICKS = 12


def read_cloud(path):
    open_ply = gzip.open if path.suffix == ".gz" else Path.open
    with open_ply(path, "rb") as f:
        header = []
        for line in f:
            header.append(line)
            if line.strip() == b"end_header":
                break
        fields = [line.split()[-1].decode() for line in header if line.startswith(b"property float ")]
        expected = "x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3"
        if fields != expected.split() or b"format binary_little_endian 1.0\n" not in header:
            raise ValueError("Expected the binary Gaussian PLY produced by scripts/rgbd_splat.py")
        vertices = np.frombuffer(f.read(), "<f4").reshape(-1, len(fields))
    color = np.clip(.5 + .28209479177387814 * vertices[:, 6:9], 0, 1)
    return np.ascontiguousarray(np.column_stack([vertices[:, :6], color, np.exp(vertices[:, [10, 12]])]),
                                dtype=np.float32)


def register_floor(cloud):
    xyz = cloud[:, :3]
    floor = xyz[(abs(cloud[:, 5]) > .92) & (abs(xyz[:, 2]) < .095)]
    if len(floor) < 100:
        raise ValueError("The scan has insufficient evidence for the TUM floor plane")
    design = np.column_stack([floor[:, :2], np.ones(len(floor))])
    inliers = np.ones(len(floor), dtype=bool)
    for _ in range(4):
        a, b, c = np.linalg.lstsq(design[inliers], floor[inliers, 2], rcond=None)[0]
        residual = floor[:, 2] - design @ [a, b, c]
        inliers = abs(residual) < .025
    up = np.array([-a, -b, 1])
    up /= np.linalg.norm(up)
    right = np.cross([0, 1, 0], up)
    right /= np.linalg.norm(right)
    transform = np.array([right, up, np.cross(right, up)])
    registered = cloud.copy()
    registered[:, :3] = (xyz - [0, 0, c]) @ transform.T
    registered[:, 3:6] = cloud[:, 3:6] @ transform.T
    return registered, dict(transform=transform.tolist(), source_origin=[0, 0, float(c)],
                            floor_inliers=int(inliers.sum()), floor_rmse_m=float(np.std(residual[inliers])))


def make_map(cloud):
    xy = cloud[:, [0, 2]]
    origin = xy.min(axis=0) - CELL * 3
    index = ((xy - origin) / CELL).astype(int)
    shape = tuple(index.max(axis=0) + 4)
    floor, obstacles = np.zeros(shape, bool), np.zeros(shape, bool)
    y = cloud[:, 1]
    measured = (abs(y) < .031) & (abs(cloud[:, 4]) > .89)
    floor[tuple(index[measured].T)] = True
    obstacles[tuple(index[(y > .035) & (y < .24)].T)] = True
    free = binary_closing(floor, iterations=2) & ~obstacles
    return free, obstacles, origin


def route(free, origin):
    # Require clearance for the body, including in-place turning.
    clearance = distance_transform_edt(free) * CELL >= .13
    cells = np.argwhere(clearance)
    if len(cells) < 2:
        raise ValueError("The surveyed floor is too small for this robot")
    delta = cells[:, None] - cells
    distances = (delta**2).sum(axis=-1)
    for n in np.argsort(distances, axis=None)[::-1]:
        i, j = np.unravel_index(n, distances.shape)
        a, b = cells[i], cells[j]
        line = np.rint(np.linspace(a, b, num=int(np.linalg.norm(b-a) * 3) + 2)).astype(int)
        if clearance[tuple(line.T)].all():
            return origin + (a + .5) * CELL, origin + (b + .5) * CELL
    raise RuntimeError("No collision-free route on surveyed floor")


def drive(directory, free, obstacles, origin, start, goal):
    import rne_py

    robot = f"""kind = "diff_drive"
model_name = "rgbd_scan_bot"

[diff_drive]
wheel_radius_m = {WHEEL}
track_width_m = {TRACK}
base_half_extents_m = [0.075, 0.04, 0.06]
max_wheel_velocity_rad_s = 20.0
initial_translation_m = [{start[0]}, 0.07, {start[1]}]
drive_mode = "joint_driven"
"""
    (directory / "robot.rne.robot.toml").write_text(robot)
    scene = directory / "scene.rne.scene.toml"
    chunks = ['[world]\ngravity_m_s2 = [0, -9.81, 0]\n[ground]\nenabled = true',
              '[[robots]]\npath = "robot.rne.robot.toml"']
    for n, (i, j, k, l) in enumerate(S._rects(obstacles)):
        x, z = origin + CELL * np.array([(i+k+1)/2, (j+l+1)/2])
        chunks.append(f'[[obstacles]]\nname = "scan_{n}"\nbody_type = "fixed"\n'
                      f'translation_m = [{x}, 0.12, {z}]\n'
                      f'half_extents_m = [{CELL*(k-i+1)/2}, 0.12, {CELL*(l-j+1)/2}]')
    scene.write_text("\n\n".join(chunks))
    sim = rne_py.DiffDriveSim.from_scene(str(scene))
    obs = sim.reset()
    path = []
    max_contacts = 0
    for _ in range(180):
        position = np.array([obs.base_x, obs.base_z])
        cell = np.floor((position - origin) / CELL).astype(int)
        if (cell < 0).any() or (cell >= free.shape).any() or not free[tuple(cell)]:
            raise RuntimeError("Robot left the surveyed free floor")
        path.append([*position, obs.base_yaw])
        error = goal - position
        distance = np.linalg.norm(error)
        if distance < .065:
            break
        angle = (np.arctan2(-error[1], error[0]) - obs.base_yaw + np.pi) % (2*np.pi) - np.pi
        v = min(.17, distance) if abs(angle) < .22 else 0
        w = np.clip(3 * angle, -1.6, 1.6)
        for _ in range(TICKS):
            obs = sim.step(float((v - w * TRACK / 2) / WHEEL), float((v + w * TRACK / 2) / WHEEL))
            max_contacts = max(max_contacts, sim.contact_count)
    else:
        raise RuntimeError("Rapier robot did not reach the goal")
    return np.array(path), max_contacts, TICKS * sim.fixed_delta_s


def render(cloud, directory, free, obstacles, origin, path, goal):
    raster = SplatRaster(directory)
    w, h = 480, 300
    images = []
    map_pixels = np.zeros((*free.T.shape, 3), np.uint8)
    map_pixels[:] = (35, 42, 49)
    map_pixels[free.T] = (108, 121, 135)
    map_pixels[obstacles.T] = (152, 76, 67)
    base = Image.fromarray(map_pixels).resize((240, 300), Image.Resampling.NEAREST)
    scale = np.array(base.size) / free.shape

    def marker(draw, pos, color, r=4):
        x, y = ((pos-origin) / CELL * scale).tolist()
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)

    robot = _cloud([dict(shape="box", size_m=[.15, .09, .12], pos_m=[0, .07, 0],
                         color_rgba=S.lin((255, 190, 55))),
                    dict(shape="box", size_m=[.035, .03, .09], pos_m=[.055, .126, 0],
                         color_rgba=S.lin((45, 58, 68)))], .012, 17)
    focus = np.array([path[:, 0].mean(), .18, path[:, 1].mean()])
    eye = focus + [-.72, 1.28, 1.12]
    basis = S._quat_mat(S.look_rot(eye, focus)).T * np.array([[1], [-1], [-1]])
    intrinsics = (290, 290, w/2, h/2)

    for n, (x, z, yaw) in enumerate(path):
        overview = base.copy()
        draw = ImageDraw.Draw(overview)
        for pos in path[:n, :2]:
            marker(draw, pos, "#e0a44a", r=2)
        marker(draw, goal, "#63da96")
        marker(draw, [x, z], "#f9c453")
        moving = robot.copy()
        rot = S._quat_mat(S.yaw_q(yaw))
        moving[:, :3] = moving[:, :3] @ rot.T + [x, 0, z]
        moving[:, 3:6] = moving[:, 3:6] @ rot.T
        image = raster.render(np.concatenate([cloud, moving]), eye, basis, intrinsics, (w, h))
        panel = Image.new("RGB", (w+240, h+34), "#131c24")
        panel.paste(image, (0, 0))
        panel.paste(overview, (w, 0))
        draw = ImageDraw.Draw(panel)
        draw.text((12, h+10), "TUM RGB-D  /  3DGS RECONSTRUCTION", fill="white")
        draw.text((w+12, h+10), "RAPIER  /  SURVEYED FLOOR", fill="white")
        images.append(panel)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sigma-scale", type=float, default=.55)
    args = parser.parse_args()
    if not .3 <= args.sigma_scale <= 2:
        parser.error("sigma scale must be between 0.3 and 2")
    cloud, metadata = register_floor(read_cloud(args.source))
    free, obstacles, origin = make_map(cloud)
    start, goal = route(free, origin)
    cloud[:, 9:] *= args.sigma_scale
    with tempfile.TemporaryDirectory(prefix="scan_sim_") as tmp:
        directory = Path(tmp)
        path, contacts, control_dt = drive(directory, free, obstacles, origin, start, goal)
        print("Reached", goal, "from", start, "in", len(path), "control frames", flush=True)
        frames = render(cloud, directory, free, obstacles, origin, path, goal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_small_gif(frames, args.output, frame_ms=round(control_dt * 1000))
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        **metadata, source=args.source.name, goal=goal.tolist(), path=path.tolist(),
        surveyed_free_cells=int(free.sum()), collider_cells=int(obstacles.sum()),
        physics="RNE Rapier (joint-driven differential drive)", control_dt_s=control_dt,
        path_length_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
        max_contacts_per_tick=contacts, goal_error_m=float(np.linalg.norm(goal - path[-1, :2])),
        sigma_scale=args.sigma_scale), indent=2) + "\n")


if __name__ == "__main__":
    main()
