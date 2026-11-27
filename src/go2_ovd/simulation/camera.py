from __future__ import annotations

import math
from typing import Any

import mujoco as mj
import numpy as np


def intrinsics(width: int, height: int, vertical_fov_deg: float) -> dict[str, float]:
    focal = 0.5 * float(height) / math.tan(0.5 * math.radians(vertical_fov_deg))
    return {"fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0}


def pose_matrix(position: np.ndarray, rotation_flat: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation_flat, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


class CameraRenderer:
    def __init__(self, model: Any, width: int, height: int, camera_name: str, occluder_group: int = 4) -> None:
        self.model = model
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.occluder_group = occluder_group
        self.renderer = mj.Renderer(model, height=height, width=width)
        self.all_geoms = mj.MjvOption()
        self.no_occluder = mj.MjvOption()
        # MuJoCo's default visualization options do not display every geom
        # group. Group 4 is reserved for the controlled experiment occluder.
        self.all_geoms.geomgroup[occluder_group] = 1
        self.no_occluder.geomgroup[occluder_group] = 0

    def rgb(self, data: Any) -> np.ndarray:
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(data, camera=self.camera_name, scene_option=self.all_geoms)
        return np.asarray(self.renderer.render(), dtype=np.uint8).copy()

    def segmentation(self, data: Any, hide_occluder: bool = False) -> np.ndarray:
        self.renderer.enable_segmentation_rendering()
        option = self.no_occluder if hide_occluder else self.all_geoms
        self.renderer.update_scene(data, camera=self.camera_name, scene_option=option)
        result = np.asarray(self.renderer.render(), dtype=np.int32).copy()
        self.renderer.disable_segmentation_rendering()
        return result

    def close(self) -> None:
        self.renderer.close()


def geom_mask(segmentation: np.ndarray, geom_id: int) -> np.ndarray:
    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise ValueError(f"unexpected MuJoCo segmentation shape: {segmentation.shape}")
    object_type = int(mj.mjtObj.mjOBJ_GEOM)
    return (segmentation[:, :, 0] == geom_id) & (segmentation[:, :, 1] == object_type)


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        return None
    return float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1)
