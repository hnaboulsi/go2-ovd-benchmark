from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from go2_ovd.evaluation.metrics import _match_frame, evaluate_run, intersection_over_union
from go2_ovd.io import write_json, write_jsonl


def _truth(frame: int, *, evaluable: bool = True, label: str = "baseball") -> dict:
    return {
        "frame_id": frame,
        "instance_id": f"{label}_0",
        "class_label": label,
        "present_in_scene": True,
        "evaluable": evaluable,
        "bbox_xyxy_px": [10, 10, 30, 30],
        "visible_pixel_count": 400 if evaluable else 10,
        "unoccluded_pixel_count": 400,
        "visible_fraction": 1.0 if evaluable else 0.025,
        "object_pose_world": {"position_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
    }


def _detection(score: float, box: list[float] | None = None, label: str = "baseball") -> dict:
    return {
        "prompt_id": 0,
        "label": label,
        "score": score,
        "bbox_xyxy_px": box or [10, 10, 30, 30],
        "mask_rle": None,
        "adapter_metadata": {},
    }


class MatchingTests(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertEqual(intersection_over_union([0, 0, 2, 2], [0, 0, 2, 2]), 1.0)
        self.assertEqual(intersection_over_union([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_duplicate_detection_is_false_positive(self) -> None:
        result = _match_frame([_detection(0.9), _detection(0.8)], [_truth(0)], "baseball", 0.35, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))

    def test_multiple_truth_miss_and_unevaluable(self) -> None:
        second = _truth(0)
        second["instance_id"] = "baseball_1"
        second["bbox_xyxy_px"] = [40, 40, 60, 60]
        result = _match_frame([_detection(0.9)], [_truth(0), second, _truth(0, evaluable=False)], "baseball", 0.35, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 1))

    def test_absent_target_activation(self) -> None:
        result = _match_frame([_detection(0.9)], [], "baseball", 0.35, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 0))
        self.assertTrue(result["activation"])

    def test_unmatched_activation_has_zero_matched_continuity_score(self) -> None:
        result = _match_frame(
            [_detection(0.9, box=[60, 60, 80, 80])],
            [_truth(0)],
            "baseball",
            0.35,
            0.5,
        )
        self.assertFalse(result["matched"])
        self.assertEqual(result["top_score"], 0.9)
        self.assertEqual(result["matched_score"], 0.0)


class RunEvaluationTests(unittest.TestCase):
    def test_end_to_end_metrics_persistence_latency_and_archival_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            target_dir = root / "episodes" / "angle_000_target"
            absent_dir = root / "episodes" / "angle_000_distractor_only"
            for episode_dir, condition in ((target_dir, "target"), (absent_dir, "distractor_only")):
                frames = [
                    {
                        "episode_id": episode_dir.name,
                        "frame_id": index,
                        "sim_time_s": index * 0.2,
                        "width_px": 100,
                        "height_px": 100,
                    }
                    for index in range(4)
                ]
                write_jsonl(episode_dir / "frames.jsonl", frames)
                write_jsonl(episode_dir / "ground_truth.jsonl", [_truth(index) for index in range(4)] if condition == "target" else [])
                write_json(episode_dir / "episode_manifest.json", {"condition": condition, "bearing_deg": 0})
            batches = []
            archival = []
            for episode_dir in (target_dir, absent_dir):
                for frame_id in range(4):
                    detections = []
                    if episode_dir == target_dir and frame_id in (0, 1, 3):
                        detections = [_detection(0.8 - frame_id * 0.1)]
                    if episode_dir == absent_dir and frame_id == 2:
                        detections = [_detection(0.7)]
                    batch = {
                        "schema_version": "1.0", "run_id": "run", "episode_id": episode_dir.name,
                        "frame_id": frame_id, "image_sha256": "x", "adapter_name": "grounding_dino",
                        "model_id": "model", "model_revision": "a" * 40, "device": "cpu",
                        "prompts": ["baseball", "softball"], "preprocessing_ms": 1 + frame_id,
                        "inference_ms": 10 + frame_id, "postprocessing_ms": 2 + frame_id,
                        "detections": detections,
                    }
                    batches.append(batch)
                    archival.append(batch)
            write_jsonl(root / "detections" / "grounding_dino_tiny.jsonl", batches)
            write_jsonl(root / "detections" / "grounding_dino_tiny_archival_0p05.jsonl", archival)
            config = root / "evaluation.yaml"
            config.write_text(
                "target_label: baseball\niou_threshold: 0.5\nscore_threshold: 0.35\nthreshold_sweep: [0.05, 0.35, 0.9]\n",
                encoding="utf-8",
            )
            metrics = evaluate_run(root, config)

        self.assertEqual(metrics["counts"], {"tp": 3, "fp": 1, "fn": 1, "precision": 0.75, "recall": 0.75, "f1": 0.75})
        self.assertEqual(metrics["absent_target_false_activation_frame_count"], 1)
        self.assertEqual(metrics["absent_target_false_activation_count"], 1)
        target = metrics["episode_metrics"]["angle_000_target"]
        self.assertEqual(target["longest_matched_run"], 2)
        self.assertEqual(target["longest_missed_run"], 1)
        self.assertEqual(target["toggle_count"], 2)
        self.assertAlmostEqual(target["matched_frame_coverage"], 0.75)
        self.assertEqual(metrics["latency_ms"]["devices"], ["cpu"])
        self.assertEqual(metrics["detector_error_count"], 0)
        self.assertEqual(metrics["threshold_sweep_source"], "grounding_dino_tiny_archival_0p05.jsonl")

    def test_unevaluable_present_target_is_not_counted_as_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            episode = root / "episodes" / "angle_000_target"
            write_jsonl(episode / "frames.jsonl", [{"episode_id": episode.name, "frame_id": 0, "sim_time_s": 0.0, "width_px": 100, "height_px": 100}])
            write_jsonl(episode / "ground_truth.jsonl", [_truth(0, evaluable=False)])
            write_json(episode / "episode_manifest.json", {"condition": "target", "bearing_deg": 0})
            batch = {
                "schema_version": "1.0", "run_id": "run", "episode_id": episode.name,
                "frame_id": 0, "image_sha256": "x", "adapter_name": "grounding_dino",
                "model_id": "model", "model_revision": "a" * 40, "device": "cpu",
                "prompts": ["baseball", "softball"], "preprocessing_ms": 1.0,
                "inference_ms": 2.0, "postprocessing_ms": 1.0, "detections": [],
            }
            write_jsonl(root / "detections" / "grounding_dino_tiny.jsonl", [batch])
            config = root / "evaluation.yaml"
            config.write_text(
                "target_label: baseball\niou_threshold: 0.5\nscore_threshold: 0.35\nthreshold_sweep: [0.35]\n",
                encoding="utf-8",
            )
            metrics = evaluate_run(root, config)
        self.assertEqual(metrics["absent_target_episode_count"], 0)


if __name__ == "__main__":
    unittest.main()
