from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import mujoco as mj
import numpy as np
from PIL import Image
import pinocchio as pin

from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import Gait, LEG_ORDER
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.leg_controller import LegController
from convex_mpc.safety_checks import clip_torque, saturation_metrics

from go2_ovd.config import config_sha256, load_yaml, repository_root
from go2_ovd.io import sha256_bytes, sha256_file, write_json, write_jsonl
from go2_ovd.schemas import FrameRecord, GroundTruthRecord
from go2_ovd.simulation.camera import CameraRenderer, geom_mask, intrinsics, mask_box, pose_matrix
from go2_ovd.simulation.scene_builder import build_scene


SIM_HZ = 1000
CTRL_HZ = 200
SIM_DT = 1.0 / SIM_HZ
CTRL_DT = 1.0 / CTRL_HZ
CTRL_DECIM = SIM_HZ // CTRL_HZ
GAIT_HZ = 3.0
GAIT_DUTY = 0.6
MPC_DT = (1.0 / GAIT_HZ) / 16.0
MPC_HZ = 1.0 / MPC_DT
STEPS_PER_MPC = max(1, int(CTRL_HZ // MPC_HZ))
TAU_LIM = 0.9 * np.asarray([23.7, 23.7, 45.43] * 4, dtype=float)
LEG_SLICE = {leg: slice(3 * index, 3 * index + 3) for index, leg in enumerate(LEG_ORDER)}


def _seed(primary: int, episode_id: str) -> int:
    digest = hashlib.sha256(f"{primary}:{episode_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _robot_start(range_m: float, bearing_deg: float, fallback: bool) -> tuple[np.ndarray, float, np.ndarray]:
    bearing = math.radians(bearing_deg)
    if fallback:
        robot_xy = np.array([-range_m, 0.0])
        yaw = 0.0
        target_xy = robot_xy + range_m * np.array([math.cos(bearing), math.sin(bearing)])
    else:
        robot_xy = np.array([-range_m * math.cos(bearing), -range_m * math.sin(bearing)])
        yaw = bearing
        target_xy = np.zeros(2)
    return robot_xy, yaw, target_xy


def _update_pin(model: Any, data: Any, go2: PinGo2Model) -> None:
    q = np.asarray(data.qpos, dtype=float).reshape(-1)
    dq = np.asarray(data.qvel, dtype=float).reshape(-1)
    qw, qx, qy, qz = q[3:7]
    rotation = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
    q_pin = np.concatenate([q[:3], [qx, qy, qz, qw], q[7:]])
    dq_pin = np.concatenate([rotation.T @ dq[:3], dq[3:]])
    go2.update_model(q_pin, dq_pin)


def _set_torque(model: Any, data: Any, torque: np.ndarray) -> None:
    for leg, values in zip(LEG_ORDER, torque.reshape(4, 3)):
        for joint, value in zip(("hip", "thigh", "calf"), values):
            actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{leg}_{joint}")
            if actuator_id < 0:
                raise ValueError(f"missing actuator {leg}_{joint}")
            data.ctrl[actuator_id] = float(value)


def _object_pose(model: Any, data: Any, body_name: str) -> dict[str, tuple[float, ...]]:
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    position = tuple(float(value) for value in data.xpos[body_id])
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    quaternion = np.zeros(4, dtype=float)
    mj.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return {"position_m": position, "quaternion_wxyz": tuple(float(value) for value in quaternion)}


def _calibrate_occluder(
    model: Any,
    data: Any,
    renderer: CameraRenderer,
    desired: float,
    tolerance: float,
    target_visual_center: np.ndarray,
) -> dict[str, Any] | None:
    occluder_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "target_occluder")
    baseball_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "baseball_geom")
    if occluder_id < 0:
        return None
    camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, renderer.camera_name)
    # geom_xpos is the mesh reference frame, not the visible mesh center. The
    # YCB mesh is recentered in X/Y and placed on the floor in scene_builder,
    # so use that known visible center for ray placement.
    target_position = np.asarray(target_visual_center, dtype=float)
    camera_position = np.asarray(data.cam_xpos[camera_id], dtype=float)
    toward_camera = camera_position - target_position
    toward_camera /= np.linalg.norm(toward_camera)
    position = target_position + 0.20 * toward_camera
    position[2] = target_position[2]
    model.geom_pos[occluder_id] = position
    yaw = math.atan2(toward_camera[1], toward_camera[0])
    model.geom_quat[occluder_id] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    model.geom_size[occluder_id, 0] = 0.01
    model.geom_size[occluder_id, 2] = 0.06
    mj.mj_forward(model, data)
    unoccluded = int(geom_mask(renderer.segmentation(data, hide_occluder=True), baseball_id).sum())
    if unoccluded == 0:
        raise RuntimeError("baseball is not visible during occluder calibration")
    low, high = 0.0005, 0.08
    best = None
    for _ in range(18):
        width = (low + high) / 2
        model.geom_size[occluder_id, 1] = width
        mj.mj_forward(model, data)
        visible = int(geom_mask(renderer.segmentation(data), baseball_id).sum())
        fraction = visible / unoccluded
        if best is None or abs(fraction - desired) < abs(best[1] - desired):
            best = (width, fraction, visible, unoccluded)
        if fraction > desired:
            low = width
        else:
            high = width
    assert best is not None
    model.geom_size[occluder_id, 1] = best[0]
    mj.mj_forward(model, data)
    if abs(best[1] - desired) > tolerance:
        raise RuntimeError(f"could not calibrate occluder: visible_fraction={best[1]:.3f}")
    return {
        "geom_position_world_m": model.geom_pos[occluder_id].tolist(),
        "geom_quaternion_wxyz": model.geom_quat[occluder_id].tolist(),
        "geom_half_size_m": model.geom_size[occluder_id].tolist(),
        "initial_visible_fraction": best[1],
    }


def run_episode(
    config: dict[str, Any],
    run_root: Path,
    episode_id: str,
    bearing_deg: float,
    condition: str,
    fallback: bool = False,
) -> dict[str, Any]:
    root = repository_root()
    controller_root = root / ".deps" / "go2-convex-mpc"
    episode_dir = run_root / "episodes" / episode_id
    rgb_dir = episode_dir / "rgb"
    segmentation_dir = episode_dir / "segmentation"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    robot_xy, yaw, target_xy = _robot_start(float(config["initial_range_m"]), bearing_deg, fallback)
    generated_dir = episode_dir / "generated_model"
    scene = build_scene(generated_dir, controller_root, config["camera"], condition, tuple(target_xy), math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0]))

    model = mj.MjModel.from_xml_path(str(scene["scene_path"]))
    data = mj.MjData(model)
    model.opt.timestep = SIM_DT
    go2 = PinGo2Model()
    q_init = go2.current_config.get_q().copy()
    q_init[0:2] = robot_xy
    q_init[3:7] = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
    px, py, pz, qx, qy, qz, qw, *joints = q_init.tolist()
    data.qpos[:] = [px, py, pz, qw, qx, qy, qz, *joints]
    mj.mj_forward(model, data)

    gait = Gait(GAIT_HZ, GAIT_DUTY)
    trajectory = ComTraj(go2)
    leg_controller = LegController()
    _update_pin(model, data, go2)
    trajectory.generate_traj(go2, gait, 0.0, 0.0, 0.0, 0.27, 0.0, MPC_DT)
    mpc = CentroidalMPC(go2, trajectory)
    renderer = CameraRenderer(model, int(config["camera"]["width_px"]), int(config["camera"]["height_px"]), str(config["camera"]["name"]), int(config["occlusion"]["geom_group"]))
    baseball_info = scene["objects"].get("baseball")
    target_visual_center = (
        np.array([*baseball_info["world_xy"], baseball_info["extent_m"][2] / 2], dtype=float)
        if baseball_info is not None else np.zeros(3, dtype=float)
    )
    occluder = _calibrate_occluder(
        model,
        data,
        renderer,
        float(config["occlusion"]["desired_visible_fraction"]),
        float(config["occlusion"]["tolerance"]),
        target_visual_center,
    )

    episode_seed = _seed(int(config["seed"]), episode_id)
    episode_config = {
        "episode_id": episode_id,
        "bearing_deg": bearing_deg,
        "condition": condition,
        "fallback_scene_relative": fallback,
        "robot_initial_xy": robot_xy.tolist(),
        "robot_initial_yaw_rad": yaw,
        "target_center_xy": target_xy.tolist(),
        "episode_seed": episode_seed,
        "source_config": config,
    }
    scene_hash = config_sha256(episode_config)
    camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, str(config["camera"]["name"]))
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link")
    object_geom_ids = {
        name: mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        for name in ("baseball", "softball")
    }
    frames: list[FrameRecord] = []
    truth: list[GroundTruthRecord] = []
    control_ms: list[float] = []
    mpc_ms: list[float] = []
    solver_status: list[str] = []
    torque_demands: list[np.ndarray] = []
    applied_torques: list[np.ndarray] = []
    failure: str | None = None
    U_raw = np.zeros((12, trajectory.N), dtype=float)
    last_contact = None
    tau_hold = np.zeros(12, dtype=float)
    ctrl_index = 0
    frame_index = 0
    capture_dt = 1.0 / float(config["capture_hz"])
    next_capture = 0.0
    last_capture_time = -math.inf
    last_sim_time: float | None = None
    sim_steps = int(round(float(config["duration_s"]) * SIM_HZ))
    intr = intrinsics(int(config["camera"]["width_px"]), int(config["camera"]["height_px"]), float(config["camera"]["vertical_fov_deg"]))
    minimum_pixels = int(config["objects"]["minimum_visible_pixels"])

    try:
        for sim_index in range(sim_steps):
            now = float(data.time)
            if not math.isfinite(now) or (last_sim_time is not None and now <= last_sim_time):
                raise RuntimeError(f"simulation time is non-finite or non-increasing: {now}")
            last_sim_time = now
            if now + 1e-12 >= next_capture and frame_index < int(round(float(config["duration_s"]) * float(config["capture_hz"]))):
                if not math.isfinite(now) or now <= last_capture_time:
                    raise RuntimeError(f"simulation time is non-finite or non-increasing: {now}")
                rgb = renderer.rgb(data)
                segmentation = renderer.segmentation(data)
                unoccluded_segmentation = renderer.segmentation(data, hide_occluder=True)
                rgb_path = rgb_dir / f"{frame_index:06d}.png"
                Image.fromarray(rgb, mode="RGB").save(rgb_path, format="PNG", optimize=False)
                instance_map = np.zeros(rgb.shape[:2], dtype=np.uint8)
                unoccluded_target = np.zeros(rgb.shape[:2], dtype=np.uint8)
                for instance_value, name in enumerate(("baseball", "softball"), start=1):
                    geom_id = object_geom_ids[name]
                    present = geom_id >= 0
                    mask = geom_mask(segmentation, geom_id) if present else np.zeros(rgb.shape[:2], dtype=bool)
                    unoccluded_mask = geom_mask(unoccluded_segmentation, geom_id) if present else mask
                    instance_map[mask] = instance_value
                    if name == "baseball":
                        unoccluded_target[unoccluded_mask] = 1
                    visible = int(mask.sum())
                    unoccluded = int(unoccluded_mask.sum())
                    truth.append(GroundTruthRecord(
                        frame_id=frame_index,
                        instance_id=f"{name}_0",
                        class_label=name,
                        present_in_scene=present,
                        evaluable=bool(present and visible >= minimum_pixels),
                        bbox_xyxy_px=mask_box(mask),
                        visible_pixel_count=visible,
                        unoccluded_pixel_count=unoccluded,
                        visible_fraction=float(visible / unoccluded) if unoccluded else 0.0,
                        object_pose_world=_object_pose(model, data, f"{name}_0") if present else {"position_m": (), "quaternion_wxyz": ()},
                    ))
                np.savez_compressed(segmentation_dir / f"{frame_index:06d}.npz", instances=instance_map, unoccluded_baseball=unoccluded_target)
                camera_pose = pose_matrix(data.cam_xpos[camera_id], data.cam_xmat[camera_id])
                base_pose = pose_matrix(data.xpos[base_id], data.xmat[base_id])
                numeric_frame_data = np.concatenate(
                    [camera_pose.reshape(-1), base_pose.reshape(-1), np.asarray(data.qpos), np.asarray(data.qvel)]
                )
                if not np.all(np.isfinite(numeric_frame_data)):
                    raise RuntimeError("camera, base, or state record contains non-finite data")
                frames.append(FrameRecord(
                    schema_version="1.0",
                    run_id=run_root.name,
                    episode_id=episode_id,
                    frame_id=frame_index,
                    sim_time_s=now,
                    rgb_path=str(rgb_path.relative_to(episode_dir)),
                    rgb_sha256=sha256_file(rgb_path),
                    pixel_sha256=sha256_bytes(rgb.tobytes(order="C")),
                    width_px=rgb.shape[1],
                    height_px=rgb.shape[0],
                    camera_name=str(config["camera"]["name"]),
                    camera_intrinsics=intr,
                    T_world_camera_row_major=tuple(float(value) for value in camera_pose.reshape(-1)),
                    T_world_base_row_major=tuple(float(value) for value in base_pose.reshape(-1)),
                    base_qpos=tuple(float(value) for value in data.qpos),
                    base_qvel=tuple(float(value) for value in data.qvel),
                    scene_config_sha256=scene_hash,
                    episode_seed=episode_seed,
                ))
                frame_index += 1
                last_capture_time = now
                next_capture += capture_dt

            if sim_index % CTRL_DECIM == 0:
                tick_start = time.perf_counter()
                _update_pin(model, data, go2)
                state = go2.compute_com_x_vec().reshape(12)
                if not np.all(np.isfinite(state)):
                    raise RuntimeError("non-finite controller state")
                if state[2] < float(config["locomotion_limits"]["minimum_base_height_m"]):
                    raise RuntimeError(f"base height below limit: {state[2]:.6f}")
                if np.max(np.abs(state[3:5])) > float(config["locomotion_limits"]["maximum_abs_roll_pitch_rad"]):
                    raise RuntimeError(f"roll/pitch exceeded limit: {state[3:5].tolist()}")
                current_contact = gait.compute_current_mask(now)
                transition = last_contact is None or not np.array_equal(current_contact, last_contact)
                if ctrl_index % STEPS_PER_MPC == 0 or transition:
                    trajectory.generate_traj(go2, gait, now, float(config["forward_velocity_mps"]), 0.0, 0.27, 0.0, MPC_DT)
                    solution = mpc.solve_QP(go2, trajectory)
                    _, U_raw = mpc.extract_solution(solution, trajectory, current_contact)
                    solver_status.append(str(mpc.last_status))
                    if str(mpc.last_status) != "solved":
                        raise RuntimeError(f"QP status was {mpc.last_status}")
                    mpc_ms.append(float(mpc.update_time + mpc.solve_time))
                applied_force = mpc.apply_contact_mask(U_raw[:, 0], current_contact)
                tau_raw = np.zeros(12, dtype=float)
                for leg in LEG_ORDER:
                    leg_slice = LEG_SLICE[leg]
                    output = leg_controller.compute_leg_torque(leg, go2, gait, applied_force[leg_slice], now, current_contact)
                    tau_raw[leg_slice] = output.tau
                tau_hold = clip_torque(tau_raw, TAU_LIM)
                torque_demands.append(tau_raw.copy())
                applied_torques.append(tau_hold.copy())
                if not np.array_equal(tau_raw, tau_hold):
                    raise RuntimeError("torque clipping occurred")
                last_contact = current_contact.copy()
                ctrl_index += 1

            mj.mj_step1(model, data)
            _set_torque(model, data, tau_hold)
            mj.mj_step2(model, data)
            if (sim_index + 1) % CTRL_DECIM == 0:
                control_ms.append((time.perf_counter() - tick_start) * 1e3)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        renderer.close()

    write_jsonl(episode_dir / "frames.jsonl", frames)
    write_jsonl(episode_dir / "ground_truth.jsonl", truth)
    np.savez_compressed(
        episode_dir / "controller_log.npz",
        control_tick_ms=np.asarray(control_ms, dtype=float),
        mpc_total_ms=np.asarray(mpc_ms, dtype=float),
        raw_torque_demand_Nm=np.asarray(torque_demands, dtype=float),
        final_clipped_torque_Nm=np.asarray(applied_torques, dtype=float),
        solver_status=np.asarray(solver_status, dtype="U64"),
    )
    saturation = saturation_metrics(np.asarray(torque_demands), np.asarray(applied_torques)) if torque_demands else {}
    metrics = {
        "valid": failure is None and len(frames) == int(round(float(config["duration_s"]) * float(config["capture_hz"]))),
        "failure": failure,
        "frame_count": len(frames),
        "control_ticks": len(control_ms),
        "mpc_solves": len(solver_status),
        "solver_status_counts": {status: solver_status.count(status) for status in sorted(set(solver_status))},
        "mpc_p95_ms": float(np.percentile(mpc_ms, 95)) if mpc_ms else None,
        "control_p95_ms": float(np.percentile(control_ms, 95)) if control_ms else None,
        "control_deadline_overruns": int(np.sum(np.asarray(control_ms) > 5.0)) if control_ms else 0,
        "torque_saturation": saturation,
    }
    write_json(episode_dir / "episode_manifest.json", {**episode_config, "scene": scene, "scene_config_sha256": scene_hash, "occluder": occluder, "metrics": metrics})
    return metrics


def capture_matrix(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    run_root = Path(output)
    run_root.mkdir(parents=True, exist_ok=True)
    attempts: dict[str, Any] = {}
    for bearing in config["bearings_deg"]:
        angle = "m" + str(abs(int(bearing))) if bearing < 0 else "p" + str(int(bearing)) if bearing > 0 else "000"
        for condition in config["conditions"]:
            episode_id = f"angle_{angle}_{condition}"
            attempts[episode_id] = run_episode(config, run_root, episode_id, float(bearing), str(condition), fallback=False)

    failed = [episode for episode, metrics in attempts.items() if not metrics["valid"]]
    fallback_results: dict[str, Any] = {}
    for episode_id in failed:
        parts = episode_id.split("_")
        angle_token = parts[1]
        condition = "_".join(parts[2:])
        bearing = -float(angle_token[1:]) if angle_token.startswith("m") else float(angle_token[1:]) if angle_token.startswith("p") else 0.0
        fallback_id = f"{episode_id}_scene_relative_fallback"
        fallback_results[fallback_id] = run_episode(config, run_root, fallback_id, bearing, condition, fallback=True)
    usable_episode_ids = [episode for episode, value in attempts.items() if value["valid"]]
    usable_episode_ids.extend(episode for episode, value in fallback_results.items() if value["valid"])
    matrix_control_ms: list[float] = []
    matrix_mpc_ms: list[float] = []
    for episode_id in usable_episode_ids:
        with np.load(run_root / "episodes" / episode_id / "controller_log.npz") as log:
            matrix_control_ms.extend(float(value) for value in log["control_tick_ms"])
            matrix_mpc_ms.extend(float(value) for value in log["mpc_total_ms"])
    matrix_timing = {
        "control_tick_p95_ms": float(np.percentile(matrix_control_ms, 95)) if matrix_control_ms else None,
        "control_tick_budget_ms": float(config["locomotion_limits"]["maximum_control_p95_ms"]),
        "mpc_total_p95_ms": float(np.percentile(matrix_mpc_ms, 95)) if matrix_mpc_ms else None,
        "mpc_budget_ms": float(config["locomotion_limits"]["maximum_mpc_p95_ms"]),
    }
    timing_budget_passed = bool(
        matrix_timing["control_tick_p95_ms"] is not None
        and matrix_timing["control_tick_p95_ms"] < matrix_timing["control_tick_budget_ms"]
        and matrix_timing["mpc_total_p95_ms"] is not None
        and matrix_timing["mpc_total_p95_ms"] < matrix_timing["mpc_budget_ms"]
    )
    manifest = {
        "schema_version": "1.0",
        "run_id": run_root.name,
        "source_config": str(Path(config_path).resolve()),
        "config_sha256": config_sha256(config),
        "primary_attempts": attempts,
        "fallback_attempts": fallback_results,
        "valid_primary_count": sum(bool(value["valid"]) for value in attempts.values()),
        "valid_fallback_count": sum(bool(value["valid"]) for value in fallback_results.values()),
        "matrix_timing": matrix_timing,
        "timing_budget_passed": timing_budget_passed,
    }
    write_json(run_root / "run_manifest.json", manifest)
    return manifest


def validate_controller(repeats: int = 3) -> dict[str, Any]:
    root = repository_root()
    controller = root / ".deps" / "go2-convex-mpc"
    python = Path(os.environ.get("GO2_SIM_PYTHON", os.sys.executable))
    environment = {**os.environ, "PYTHONPATH": str(controller / "src"), "PYTHONNOUSERSITE": "1", "MPLBACKEND": "Agg"}
    tests = subprocess.run([str(python), "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=controller, env=environment, text=True, capture_output=True)
    results = []
    for index in range(repeats):
        output = root / "runs" / "controller_validation" / f"repeat_{index + 1}"
        completed = subprocess.run([str(python), "-m", "examples.validate_mpc", "--output-dir", str(output)], cwd=controller, env=environment, text=True, capture_output=True)
        report_path = output / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        results.append({"returncode": completed.returncode, "report": report, "stderr": completed.stderr[-2000:]})
    summary = {"tests_returncode": tests.returncode, "tests_output": (tests.stdout + tests.stderr)[-5000:], "repeats": results, "passed": tests.returncode == 0 and all(item["returncode"] == 0 and item["report"] and item["report"]["passed"] for item in results)}
    write_json(root / "runs" / "controller_validation" / "summary.json", summary)
    return summary
