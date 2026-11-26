"""Fuse real TUM Freiburg 1 RGB-D frames into world-space Gaussians."""

import argparse
import hashlib
import json
from io import BytesIO
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from navfm.splat_render import SplatRaster

SEQUENCE = "rgbd_dataset_freiburg1_desk"
SOURCE = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg1/{SEQUENCE}.tgz"
INTRINSICS = np.array([517.3, 516.5, 318.6, 255.3])
DISTORTION = (-.2624, .9999, -.0009, .0011, -1.3054)
DEPTH_FACTOR = 5000
VOXEL = .012


def read_index(archive, name):
    with archive.extractfile(f"{SEQUENCE}/{name}.txt") as f:
        return [line.split() for line in f.read().decode().splitlines() if line and not line.startswith("#")]


def closest(timestamps, target):
    right = np.searchsorted(timestamps, target).clip(1, len(timestamps) - 1)
    left = right - 1
    return np.where(target - timestamps[left] < timestamps[right] - target, left, right)


def associations(archive):
    rgb = read_index(archive, "rgb")
    depth = read_index(archive, "depth")
    rt, dt = (np.array([float(row[0]) for row in rows]) for rows in (rgb, depth))
    ground = np.asarray(read_index(archive, "groundtruth"), dtype=float)
    gt = ground[:, 0]
    # RGB and depth are registered by the camera. Pose interpolation uses the depth timestamp.
    di = closest(dt, rt)
    gi = closest(gt, dt[di])
    valid = (abs(rt - dt[di]) < .025) & (abs(gt[gi] - dt[di]) < .015) & (dt[di] > gt[0]) & (dt[di] < gt[-1])
    indices = np.flatnonzero(valid)
    times = dt[di[indices]]
    rotations = Slerp(gt, Rotation.from_quat(ground[:, 4:]))(times).as_matrix()
    positions = np.stack([np.interp(times, gt, ground[:, i]) for i in range(1, 4)], axis=-1)
    return [(t, rgb[i][1], depth[di[i]][1], rot, pos)
            for t, i, rot, pos in zip(times, indices, rotations, positions)]


def read_image(payloads, name):
    with Image.open(BytesIO(payloads[name])) as image:
        return np.array(image)


def lift(depth):
    fx, fy, cx, cy = INTRINSICS
    v, u = np.indices(depth.shape, dtype=np.float32)
    xd, yd = (u - cx) / fx, (v - cy) / fy
    x, y = xd.copy(), yd.copy()
    k1, k2, p1, p2, k3 = DISTORTION
    for _ in range(6):
        r2 = x*x + y*y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x, y = ((xd - 2*p1*x*y - p2*(r2 + 2*x*x)) / radial,
                (yd - p1*(r2 + 2*y*y) - 2*p2*x*y) / radial)
    return np.stack([x*depth, y*depth, depth], axis=-1)


def sample(payloads, frame):
    _, rgb_path, depth_path, rotation, translation = frame
    depth = read_image(payloads, depth_path).astype(np.float32) / DEPTH_FACTOR
    rgb = read_image(payloads, rgb_path)[..., :3]
    xyz = lift(depth)
    interior = (slice(1, -1, 2), slice(1, -1, 2))
    points = xyz[1:-1, 1:-1][::2, ::2]
    horizontal = xyz[1:-1, 2:][::2, ::2] - xyz[1:-1, :-2][::2, ::2]
    vertical = xyz[2:, 1:-1][::2, ::2] - xyz[:-2, 1:-1][::2, ::2]
    normals = np.cross(vertical, horizontal)
    norm = np.linalg.norm(normals, axis=-1)
    valid = ((points[..., 2] > .45) & (points[..., 2] < 4.5) & (norm > 1e-7)
             & (np.linalg.norm(horizontal, axis=-1) < .07) & (np.linalg.norm(vertical, axis=-1) < .07))
    normals = normals[valid] / norm[valid, None]
    return np.concatenate([points[valid] @ rotation.T + translation, normals @ rotation.T,
                           rgb[interior][valid] / 255], axis=1)


def fuse(samples):
    points = np.concatenate(samples)
    cells = np.floor(points[:, :3] / VOXEL).astype(np.int32)
    # Keep surfaces with different orientations separate at corners.
    normals = points[:, 3:6]
    axis = abs(normals).argmax(axis=1)
    side = 2 * axis + (normals[np.arange(len(axis)), axis] > 0)
    indices = np.unique(np.column_stack([cells, side]), axis=0, return_inverse=True)[1]
    n = np.bincount(indices)
    averages = np.stack([np.bincount(indices, weights=channel) / n for channel in points.T], axis=1)
    lengths = np.linalg.norm(averages[:, 3:6], axis=1, keepdims=True)
    averages[:, 3:6] /= lengths
    sigma = np.broadcast_to([.65 * VOXEL, .12 * VOXEL], (len(n), 2))
    return np.ascontiguousarray(np.hstack([averages, sigma]), dtype=np.float32)


def write_ply(path, cloud):
    normals = cloud[:, 3:6]
    # Rotate each Gaussian's local Z (its short axis) onto the measured surface normal.
    w = np.sqrt(np.clip((1 + normals[:, 2]) / 2, 0, 1))
    rotations = np.zeros((len(cloud), 4), dtype=np.float32)
    valid = w > 1e-5
    rotations[valid, 0] = w[valid]
    rotations[valid, 1] = -normals[valid, 1] / (2 * w[valid])
    rotations[valid, 2] = normals[valid, 0] / (2 * w[valid])
    rotations[~valid, 1] = 1
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    dc = (cloud[:, 6:9] - .5) / .28209479177387814
    opacity = np.full((len(cloud), 1), np.log(.92 / .08))
    scales = np.log(cloud[:, [9, 9, 10]])
    vertices = np.hstack([cloud[:, :6], dc, opacity, scales, rotations]).astype("<f4")
    properties = ("x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 "
                  "rot_0 rot_1 rot_2 rot_3").split()
    with path.open("wb") as file:
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(cloud)}"]
        header += [f"property float {name}" for name in properties]
        file.write(("\n".join(header + ["end_header"]) + "\n").encode("ascii"))
        vertices.tofile(file)


def run(archive_path, output):
    with tarfile.open(archive_path) as archive:
        frames = associations(archive)
        start = frames[0][0] + 2
        # Reserve alternating views of the same interval for rendering only.
        interval = [f for f in frames if start <= f[0] < start + 9]
        selected = interval[::3]
        fit, replay = selected[::2], selected[1::2]
        names = {name for frame in fit for name in frame[1:3]}
        members = sorted((archive.getmember(f"{SEQUENCE}/{name}") for name in names),
                         key=lambda member: member.offset_data)
        payloads = {}
        for member in members:
            with archive.extractfile(member) as f:
                payloads[member.name.split("/", 1)[1]] = f.read()
        cloud = fuse([sample(payloads, frame) for frame in fit])
    with tempfile.TemporaryDirectory(prefix="rgbd_splat_") as directory:
        raster = SplatRaster(directory)
        images = []
        for time, _, _, rotation, position in replay:
            # The output is an undistorted pinhole camera at the measured pose.
            image = raster.render(cloud, position, rotation.T, INTRINSICS * .75, (480, 360))
            frame = Image.new("RGB", (480, 386), (23, 28, 35))
            frame.paste(image, (0, 0))
            ImageDraw.Draw(frame).text((10, 366), f"TUM RGB-D  /  FREIBURG 1 DESK  /  {time-start:04.1f}s", fill="white")
            images.append(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    from scripts.make_gifs import save_gif

    for stride in (1, 2, 3, 4):
        save_gif(images[:-1:stride] + images[-1:], output, hold=max(1, 5 // stride),
                 frame_ms=200 * stride)
        if output.stat().st_size <= 3_000_000:
            break
    else:
        output.unlink()
        raise RuntimeError("RGB-D replay exceeds 3 MB")
    with archive_path.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    write_ply(output.with_suffix(".ply"), cloud)
    output.with_suffix(".json").write_text(json.dumps(dict(
        source=SOURCE, sequence=SEQUENCE, license="CC BY 4.0", archive_sha256=checksum,
        gaussian_count=len(cloud), voxel_m=VOXEL, depth_factor=DEPTH_FACTOR,
        input_intrinsics=INTRINSICS.tolist(), distortion=list(DISTORTION),
        fit_depth_timestamps=[frame[0] for frame in fit],
        replay_depth_timestamps=[frame[0] for frame in replay],
        frame_stride=stride,
    ), indent=2) + "\n")
    print(len(fit), "RGB-D inputs;", len(cloud), "Gaussians;", output.stat().st_size, "bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.output)
