"""Render RNE replay geometry as surface-sampled 3D Gaussians on the CPU."""

import ctypes
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from . import rne_scene as S


@lru_cache(maxsize=None)
def _meshes(path):
    import trimesh

    return trimesh.load_scene(path).dump()


def _item_meshes(item):
    import trimesh

    shape = item["shape"]
    if shape == "mesh":
        return _meshes(item["path"]), np.asarray(item["scale"])
    if shape == "box":
        return [trimesh.creation.box()], np.asarray(item["size_m"])
    if shape == "cylinder":
        return [trimesh.creation.cylinder(radius=1, height=1, sections=16)], np.array(
            [item["radius_m"], item["radius_m"], item["length_m"]])
    if shape == "sphere":
        return [trimesh.creation.icosphere(subdivisions=1)], np.full(3, item["radius_m"])
    raise ValueError(f"Unsupported replay shape: {shape}")


def _sample_item(item, spacing, rng):
    import trimesh

    meshes, scale = _item_meshes(item)
    rot = S._quat_mat(item.get("rot", [0, 0, 0, 1]))
    translation = np.asarray(item["pos_m"])
    for mesh in meshes:
        vertices = (mesh.vertices * scale) @ rot.T + translation
        surface = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        area = surface.area
        if area <= 0:
            continue
        count = max(8, int(area / spacing**2))
        points, faces = trimesh.sample.sample_surface(surface, count, seed=rng)
        normals = surface.face_normals[faces]
        color = np.asarray(item["color_rgba"][:3]) ** (1 / 2.2)
        if item["shape"] == "mesh" and not item.get("flat_color") and mesh.visual.kind == "texture":
            bary = trimesh.triangles.points_to_barycentric(surface.triangles[faces], points)
            uv = np.einsum("ni,nij->nj", bary, mesh.visual.uv[mesh.faces[faces]])
            color = np.asarray(mesh.visual.material.to_color(uv))[:, :3] / 255
        # Bake simple directional diffuse lighting into the view-independent colors.
        light = .65 + .35 * np.maximum(0, normals @ np.array([-.36, .86, .36]))
        colors = np.asarray(color) * light[:, None]
        sigma = np.full((count, 1), .78 * np.sqrt(area / count))
        yield np.concatenate([points, normals, colors, sigma, sigma * .14], axis=1)


def _cloud(items, spacing, seed):
    rng = np.random.default_rng(seed)
    parts = [cloud for item in items for cloud in _sample_item(item, spacing, rng)]
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


class SplatRaster:
    def __init__(self, work_dir):
        library = Path(work_dir) / "splat_raster.so"
        source = Path(__file__).with_name("splat_raster.cpp")
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3", "-shared", "-fPIC",
                        str(source), "-o", str(library)], check=True)
        self.library = ctypes.CDLL(str(library))
        self.raster = self.library.splat_raster
        fp32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.raster.argtypes = [fp32, ctypes.c_int, fp32, fp32, ctypes.c_int, ctypes.c_int, fp32]
        self.raster.restype = None
    def render(self, cloud, eye, basis, intrinsics, size):
        w, h = size
        camera = np.array([*eye, *basis.ravel()], dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        pixels = np.zeros((h, w, 3), dtype=np.float32)
        self.raster(cloud, len(cloud), camera, intrinsics, w, h, pixels)
        return Image.fromarray(np.uint8(np.clip(pixels, 0, 1) * 255))


class SplatRenderer(SplatRaster):
    def __init__(self, layout, wood_floor, work_dir):
        super().__init__(work_dir)
        self.static = _cloud(S.static_items(layout, wood_floor=wood_floor), .035, 53)

    def image(self, cloud, view):
        w, h = view["width"], view["height"]
        basis = S._quat_mat(view["rot"]).T * np.array([[1], [-1], [-1]])
        focal = .5 * h / np.tan(.5 * view["fov_y_rad"])
        return self.render(cloud, view["eye_m"], basis, (focal, focal, w / 2, h / 2), (w, h))

    def render_episode(self, kind, accent, records, target_box, output):
        for i, r in enumerate(records):
            robot = _cloud(S.robot_items(r["pose"], accent), .016, 1)
            scene = np.concatenate([self.static, robot])
            overlays = S.overlay_items(kind, r["viz"], r["pose"], r["path"], target_box, accent)
            if overlays:
                overview = np.concatenate([scene, _cloud(overlays, .024, 2)])
            else:
                overview = scene
            for cloud, view in ((overview, S.overview_view(f"ov_{i:04d}.png")),
                                (scene, S.fpv_view(r["pose"], f"fpv_{i:04d}.png"))):
                self.image(cloud, view).save(output / view["out"])
