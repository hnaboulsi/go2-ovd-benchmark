from __future__ import annotations

import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from typing import Any

import mujoco as mj
import numpy as np

from go2_ovd.config import load_yaml, repository_root
from go2_ovd.io import sha256_bytes, write_json
from go2_ovd.simulation.camera import CameraRenderer, geom_mask, intrinsics, pose_matrix
from go2_ovd.simulation.scene_builder import build_scene


def _set_initial_pose(model: Any, data: Any) -> None:
    data.qpos[0:3] = [-1.5, 0.0, 0.27]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj.mj_forward(model, data)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise RuntimeError("calibration marker is absent from segmentation")
    return float(columns.mean()), float(rows.mean())


def _append_markers(scene_path: Path, marker_positions: dict[str, np.ndarray]) -> None:
    tree = ET.parse(scene_path)
    world = tree.getroot().find("worldbody")
    if world is None:
        raise ValueError("calibration scene has no worldbody")
    colors = {
        "marker_center": "1 0 0 1",
        "marker_world_y": "0 1 0 1",
        "marker_world_z": "0 0 1 1",
    }
    for name, position in marker_positions.items():
        ET.SubElement(
            world,
            "geom",
            {
                "name": name,
                "type": "sphere",
                "size": "0.018",
                "pos": " ".join(f"{value:.12g}" for value in position),
                "rgba": colors[name],
                "contype": "0",
                "conaffinity": "0",
            },
        )
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)


def validate_camera(config_path: str | Path | None = None) -> dict[str, Any]:
    root = repository_root()
    config_file = Path(config_path) if config_path else root / "configs" / "episodes" / "baseball_softball_v1.yaml"
    config = load_yaml(config_file)
    camera = config["camera"]
    validation_dir = root / "runs" / "camera_validation"
    if validation_dir.exists():
        shutil.rmtree(validation_dir)
    scene = build_scene(
        validation_dir / "generated_model",
        root / ".deps" / "go2-convex-mpc",
        camera,
        "target",
        (0.0, 0.0),
        0.0,
    )

    initial_model = mj.MjModel.from_xml_path(str(scene["scene_path"]))
    initial_data = mj.MjData(initial_model)
    _set_initial_pose(initial_model, initial_data)
    camera_id = mj.mj_name2id(initial_model, mj.mjtObj.mjOBJ_CAMERA, str(camera["name"]))
    camera_position = np.asarray(initial_data.cam_xpos[camera_id], dtype=float)
    camera_rotation = np.asarray(initial_data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    optical_center = camera_position + camera_rotation @ np.array([0.0, 0.0, -1.0])
    marker_positions = {
        "marker_center": optical_center,
        "marker_world_y": optical_center + np.array([0.0, 0.10, 0.0]),
        "marker_world_z": optical_center + np.array([0.0, 0.0, 0.10]),
    }
    _append_markers(Path(scene["scene_path"]), marker_positions)

    model = mj.MjModel.from_xml_path(str(scene["scene_path"]))
    data = mj.MjData(model)
    _set_initial_pose(model, data)
    renderer = CameraRenderer(
        model,
        int(camera["width_px"]),
        int(camera["height_px"]),
        str(camera["name"]),
        int(config["occlusion"]["geom_group"]),
    )
    try:
        # The first offscreen draw initializes the renderer's buffers. Compare
        # settled captures rather than an initialization draw with a later draw.
        renderer.rgb(data)
        rgb_a = renderer.rgb(data)
        rgb_b = renderer.rgb(data)
        segmentation = renderer.segmentation(data)
        marker_centers = {}
        for name in marker_positions:
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            marker_centers[name] = _centroid(geom_mask(segmentation, geom_id))
        camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, str(camera["name"]))
        transform_before = pose_matrix(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
        data.qpos[0] += 0.10
        mj.mj_forward(model, data)
        transform_after = pose_matrix(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
    finally:
        renderer.close()

    expected = intrinsics(int(camera["width_px"]), int(camera["height_px"]), float(camera["vertical_fov_deg"]))
    center_u, center_v = marker_centers["marker_center"]
    y_u, _ = marker_centers["marker_world_y"]
    _, z_v = marker_centers["marker_world_z"]
    center_error = math.hypot(center_u - expected["cx"], center_v - expected["cy"])
    checks = {
        "image_size": list(rgb_a.shape) == [int(camera["height_px"]), int(camera["width_px"]), 3],
        "rgb_uint8": rgb_a.dtype == np.uint8,
        "center_marker_visible": center_error <= 10.0,
        "world_positive_y_moves_left": y_u < center_u,
        "world_positive_z_moves_up": z_v < center_v,
        "repeat_render_hash_identical": sha256_bytes(rgb_a.tobytes()) == sha256_bytes(rgb_b.tobytes()),
        "camera_translation_tracks_base": np.allclose(
            transform_after[:3, 3] - transform_before[:3, 3], [0.10, 0.0, 0.0], atol=1e-9
        ),
        "camera_rotation_unchanged_for_translation": np.allclose(
            transform_after[:3, :3], transform_before[:3, :3], atol=1e-12
        ),
        "intrinsics_finite": all(math.isfinite(value) and value > 0.0 for value in (expected["fx"], expected["fy"])),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "image_shape_hwc": list(rgb_a.shape),
        "camera_intrinsics": expected,
        "marker_centers_px": {name: list(value) for name, value in marker_centers.items()},
        "center_error_px": center_error,
        "transform_convention": "T_world_camera maps homogeneous camera-frame points into world coordinates",
        "T_world_camera_row_major": transform_before.reshape(-1).tolist(),
        "repeat_rgb_sha256": sha256_bytes(rgb_a.tobytes()),
    }
    write_json(validation_dir / "report.json", report)
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"camera validation failed: {failed}")
    return report
