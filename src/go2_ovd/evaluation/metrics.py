from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from go2_ovd.config import load_yaml
from go2_ovd.io import read_jsonl, write_json


def intersection_over_union(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union > 0.0 else 0.0


def _longest(values: list[bool], target: bool) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value is target else 0
        best = max(best, current)
    return best


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _match_frame(
    detections: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    target_label: str,
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    predicted = sorted(
        [item for item in detections if item["label"] == target_label and float(item["score"]) >= score_threshold],
        key=lambda item: -float(item["score"]),
    )
    truth = [
        item for item in ground_truth
        if item["class_label"] == target_label and item["present_in_scene"] and item["evaluable"]
    ]
    unmatched = set(range(len(truth)))
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for detection in predicted:
        candidates = [
            (index, intersection_over_union(detection["bbox_xyxy_px"], truth[index]["bbox_xyxy_px"]))
            for index in unmatched
        ]
        if not candidates:
            continue
        index, iou = max(candidates, key=lambda item: item[1])
        if iou >= iou_threshold:
            unmatched.remove(index)
            matches.append((detection, truth[index], iou))
    return {
        "tp": len(matches),
        "fp": len(predicted) - len(matches),
        "fn": len(unmatched),
        "truth_count": len(truth),
        "matched": bool(matches),
        "matches": matches,
        "activation": bool(predicted),
        "top_score": max((float(item["score"]) for item in predicted), default=0.0),
        "matched_score": max((float(item[0]["score"]) for item in matches), default=0.0),
    }


def _average_precision(points: list[dict[str, float]]) -> float:
    if not points:
        return 0.0
    ordered = sorted(points, key=lambda item: item["recall"])
    recalls = np.array([0.0] + [item["recall"] for item in ordered] + [1.0])
    precisions = np.array([1.0] + [item["precision"] for item in ordered] + [0.0])
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    changing = np.flatnonzero(recalls[1:] != recalls[:-1]) + 1
    return float(np.sum((recalls[changing] - recalls[changing - 1]) * precisions[changing]))


def _write_plots(metrics: dict[str, Any], output_dir: Path, inference: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    points = metrics["threshold_points"]
    axes[0].plot([row["recall"] for row in points], [row["precision"] for row in points], marker="o")
    coordinate_thresholds: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in points:
        coordinate_thresholds[(round(row["recall"], 8), round(row["precision"], 8))].append(row["threshold"])
    for (recall, precision), thresholds in coordinate_thresholds.items():
        label = f"{thresholds[0]:.2g}" if len(thresholds) == 1 else f"{min(thresholds):.2g}-{max(thresholds):.2g}"
        axes[0].annotate(label, (recall, precision), xytext=(5, -12), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Threshold sweep", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[0].grid(alpha=0.25)

    names = list(metrics["episode_metrics"])
    coverage = [metrics["episode_metrics"][name]["matched_frame_coverage"] for name in names]
    short_names = [name.replace("angle_", "") for name in names]
    axes[1].bar(range(len(names)), coverage, color="#3b82f6")
    axes[1].set_xticks(range(len(names)), short_names, rotation=80, ha="right", fontsize=7)
    axes[1].set(ylabel="Matched-frame coverage", title="Persistence by episode", ylim=(0, 1.05))
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].hist(inference, bins=min(20, max(1, len(inference))), color="#7c3aed", edgecolor="white")
    axes[2].set(xlabel="Inference latency (ms)", ylabel="Frames", title="Detector inference latency")
    axes[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "summary_metrics.png", dpi=160)
    plt.close(figure)


def evaluate_run(run_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    run_root = Path(run_path)
    config = load_yaml(config_path)
    target = str(config["target_label"])
    score_threshold = float(config["score_threshold"])
    iou_threshold = float(config["iou_threshold"])

    frame_map: dict[tuple[str, int], dict[str, Any]] = {}
    truth_map: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    episode_manifest: dict[str, dict[str, Any]] = {}
    for episode_dir in sorted((run_root / "episodes").glob("*")):
        if not episode_dir.is_dir():
            continue
        for frame in read_jsonl(episode_dir / "frames.jsonl"):
            frame_map[(frame["episode_id"], int(frame["frame_id"]))] = frame
        for record in read_jsonl(episode_dir / "ground_truth.jsonl"):
            truth_map[(episode_dir.name, int(record["frame_id"]))].append(record)
        manifest_path = episode_dir / "episode_manifest.json"
        if manifest_path.exists():
            import json
            episode_manifest[episode_dir.name] = json.loads(manifest_path.read_text(encoding="utf-8"))

    detection_path = run_root / "detections" / "grounding_dino_tiny.jsonl"
    batches = read_jsonl(detection_path)
    detection_map = {
        (item["episode_id"], int(item["frame_id"])): item
        for item in batches if "detections" in item
    }
    errors = [item for item in batches if "error_type" in item]
    archival_path = run_root / "detections" / "grounding_dino_tiny_archival_0p05.jsonl"
    archival_batches = read_jsonl(archival_path) if archival_path.exists() else batches
    archival_map = {
        (item["episode_id"], int(item["frame_id"])): item
        for item in archival_batches if "detections" in item
    }

    def aggregate(threshold: float, source: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
        totals = {"tp": 0, "fp": 0, "fn": 0}
        per_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for key in sorted(frame_map):
            batch = source.get(key, {"detections": []})
            result = _match_frame(batch["detections"], truth_map.get(key, []), target, threshold, iou_threshold)
            for name in totals:
                totals[name] += result[name]
            per_episode[key[0]].append({"frame_id": key[1], **result})
        precision = _safe_div(totals["tp"], totals["tp"] + totals["fp"])
        recall = _safe_div(totals["tp"], totals["tp"] + totals["fn"])
        return {**totals, "precision": precision, "recall": recall, "f1": _safe_div(2 * precision * recall, precision + recall)}, per_episode

    overall, per_episode = aggregate(score_threshold, detection_map)
    threshold_points = []
    for threshold in config["threshold_sweep"]:
        summary, _ = aggregate(float(threshold), archival_map)
        threshold_points.append({"threshold": float(threshold), "precision": summary["precision"], "recall": summary["recall"]})

    ious: list[float] = []
    center_errors: list[float] = []
    size_errors: list[float] = []
    episode_metrics: dict[str, Any] = {}
    absent_episodes = false_active_episodes = 0
    absent_frames = false_active_frames = 0
    per_frame_metrics: list[dict[str, Any]] = []
    for episode_id, rows in sorted(per_episode.items()):
        eligible = [row for row in rows if row["truth_count"] > 0]
        matched_series = [bool(row["matched"]) for row in eligible]
        scores = [float(row["matched_score"]) for row in eligible]
        matched_scores = [float(row["matched_score"]) for row in eligible if row["matched"]]
        toggles = sum(a != b for a, b in zip(matched_series, matched_series[1:]))
        target_present_in_scene = any(
            item["class_label"] == target and bool(item["present_in_scene"])
            for (truth_episode, _), records in truth_map.items()
            if truth_episode == episode_id
            for item in records
        )
        if not target_present_in_scene:
            absent_episodes += 1
            if any(row["activation"] for row in rows):
                false_active_episodes += 1
            absent_frames += len(rows)
            false_active_frames += sum(bool(row["activation"]) for row in rows)
        for row in rows:
            for detection, truth, iou in row["matches"]:
                ious.append(float(iou))
                dx1, dy1, dx2, dy2 = (float(value) for value in detection["bbox_xyxy_px"])
                gx1, gy1, gx2, gy2 = (float(value) for value in truth["bbox_xyxy_px"])
                frame = frame_map[(episode_id, int(row["frame_id"]))]
                center_errors.append(math.hypot((dx1 + dx2 - gx1 - gx2) / (2 * frame["width_px"]), (dy1 + dy2 - gy1 - gy2) / (2 * frame["height_px"])))
                size_errors.append((abs((dx2 - dx1) - (gx2 - gx1)) / frame["width_px"] + abs((dy2 - dy1) - (gy2 - gy1)) / frame["height_px"]) / 2)
        episode_tp = sum(int(row["tp"]) for row in rows)
        episode_fp = sum(int(row["fp"]) for row in rows)
        episode_fn = sum(int(row["fn"]) for row in rows)
        episode_precision = _safe_div(episode_tp, episode_tp + episode_fp)
        episode_recall = _safe_div(episode_tp, episode_tp + episode_fn)
        episode_metrics[episode_id] = {
            "tp": episode_tp,
            "fp": episode_fp,
            "fn": episode_fn,
            "precision": episode_precision,
            "recall": episode_recall,
            "f1": _safe_div(2 * episode_precision * episode_recall, episode_precision + episode_recall),
            "evaluable_target_frames": len(eligible),
            "matched_frame_coverage": _safe_div(sum(matched_series), len(matched_series)),
            "longest_matched_run": _longest(matched_series, True),
            "longest_missed_run": _longest(matched_series, False),
            "toggle_count": toggles,
            "toggle_rate": _safe_div(toggles, max(0, len(matched_series) - 1)),
            "mean_score_with_misses_zero": float(np.mean(scores)) if scores else None,
            "matched_score_mean": float(np.mean(matched_scores)) if matched_scores else None,
            "matched_score_std": float(np.std(matched_scores)) if matched_scores else None,
            "median_absolute_consecutive_score_change": median([abs(b - a) for a, b in zip(scores, scores[1:])]) if len(scores) > 1 else None,
            "score_continuity_definition": "top matched score with missed evaluable frames assigned zero",
        }
        for row in rows:
            frame_precision = _safe_div(row["tp"], row["tp"] + row["fp"])
            frame_recall = _safe_div(row["tp"], row["tp"] + row["fn"])
            per_frame_metrics.append(
                {
                    "episode_id": episode_id,
                    "frame_id": int(row["frame_id"]),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "precision": frame_precision,
                    "recall": frame_recall,
                }
            )

    preprocess = [float(item["preprocessing_ms"]) for item in detection_map.values()]
    inference = [float(item["inference_ms"]) for item in detection_map.values()]
    postprocess = [float(item["postprocessing_ms"]) for item in detection_map.values()]
    total = [a + b + c for a, b, c in zip(preprocess, inference, postprocess)]
    devices = sorted({str(item.get("device", "unknown")) for item in detection_map.values()})
    simulated_minutes = sum(float(frame["sim_time_s"]) >= 0 for frame in frame_map.values()) / 5.0 / 60.0
    comparison_groups: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for episode_id, rows in per_episode.items():
        manifest = episode_manifest.get(episode_id, {})
        condition = str(manifest.get("condition", "unknown"))
        bearing = float(manifest.get("bearing_deg", 999.0))
        bearing_group = "central" if abs(bearing) < 1e-9 else "oblique"
        for group in (f"bearing:{bearing_group}", f"condition:{condition}"):
            comparison_groups[group]["tp"] += sum(int(row["tp"]) for row in rows)
            comparison_groups[group]["fp"] += sum(int(row["fp"]) for row in rows)
            comparison_groups[group]["fn"] += sum(int(row["fn"]) for row in rows)
    comparisons: dict[str, Any] = {}
    for group, counts in sorted(comparison_groups.items()):
        precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
        comparisons[group] = {**counts, "precision": precision, "recall": recall, "f1": _safe_div(2 * precision * recall, precision + recall)}
    metrics = {
        "schema_version": "1.0",
        "target_label": target,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "counts": overall,
        "average_precision_threshold_sweep": _average_precision(threshold_points),
        "threshold_points": threshold_points,
        "false_positives_per_frame": _safe_div(overall["fp"], len(frame_map)),
        "false_positives_per_simulated_minute": _safe_div(overall["fp"], simulated_minutes),
        "absent_target_episode_count": absent_episodes,
        "absent_target_false_activation_count": false_active_episodes,
        "absent_target_false_activation_rate": _safe_div(false_active_episodes, absent_episodes),
        "absent_target_frame_count": absent_frames,
        "absent_target_false_activation_frame_count": false_active_frames,
        "absent_target_false_activation_frame_rate": _safe_div(false_active_frames, absent_frames),
        "localization": {
            "matched_iou_median": median(ious) if ious else None,
            "matched_iou_iqr": (_percentile(ious, 75) - _percentile(ious, 25)) if ious else None,
            "normalized_center_error_median": median(center_errors) if center_errors else None,
            "normalized_size_error_median": median(size_errors) if size_errors else None,
        },
        "latency_ms": {
            "devices": devices,
            "preprocessing_median": median(preprocess) if preprocess else None,
            "inference_median": median(inference) if inference else None,
            "inference_p95": _percentile(inference, 95),
            "inference_max": max(inference) if inference else None,
            "postprocessing_median": median(postprocess) if postprocess else None,
            "end_to_end_p95": _percentile(total, 95),
        },
        "detector_error_count": len(errors),
        "per_frame_metrics": per_frame_metrics,
        "episode_metrics": episode_metrics,
        "primary_comparisons": comparisons,
        "threshold_sweep_source": archival_path.name if archival_path.exists() else detection_path.name,
    }
    output = run_root / "evaluation" / "metrics.json"
    write_json(output, metrics)
    _write_plots(metrics, output.parent, inference)
    return metrics
