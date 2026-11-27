from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameRecord:
    schema_version: str
    run_id: str
    episode_id: str
    frame_id: int
    sim_time_s: float
    rgb_path: str
    rgb_sha256: str
    pixel_sha256: str
    width_px: int
    height_px: int
    camera_name: str
    camera_intrinsics: dict[str, float]
    T_world_camera_row_major: tuple[float, ...]
    T_world_base_row_major: tuple[float, ...]
    base_qpos: tuple[float, ...]
    base_qvel: tuple[float, ...]
    scene_config_sha256: str
    episode_seed: int


@dataclass(frozen=True)
class GroundTruthRecord:
    frame_id: int
    instance_id: str
    class_label: str
    present_in_scene: bool
    evaluable: bool
    bbox_xyxy_px: tuple[float, float, float, float] | None
    visible_pixel_count: int
    unoccluded_pixel_count: int
    visible_fraction: float
    object_pose_world: dict[str, tuple[float, ...]]

