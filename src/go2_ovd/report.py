from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from go2_ovd.config import repository_root
from go2_ovd.io import read_jsonl, write_json


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _overlay_episode(episode_dir: Path, batch: dict[str, Any] | None, output: Path) -> None:
    frames = read_jsonl(episode_dir / "frames.jsonl")
    truth = [row for row in read_jsonl(episode_dir / "ground_truth.jsonl") if row["frame_id"] == 0]
    image = Image.open(episode_dir / frames[0]["rgb_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in truth:
        if row["bbox_xyxy_px"] is not None and row["evaluable"]:
            color = "lime" if row["class_label"] == "baseball" else "cyan"
            draw.rectangle(row["bbox_xyxy_px"], outline=color, width=2)
            draw.text((row["bbox_xyxy_px"][0], max(0, row["bbox_xyxy_px"][1] - 12)), f"GT {row['class_label']}", fill=color)
    if batch:
        for detection in batch.get("detections", []):
            box = detection["bbox_xyxy_px"]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], min(image.height - 12, box[3] + 2)), f"{detection['label']} {detection['score']:.2f}", fill="red")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def create_report(run_path: str | Path) -> Path:
    run_root = Path(run_path).resolve()
    metrics_path = run_root / "evaluation" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("evaluation/metrics.json is missing; run evaluate first")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    detection_path = run_root / "detections" / "grounding_dino_tiny.jsonl"
    batches = read_jsonl(detection_path)
    first_batches = {
        item["episode_id"]: item
        for item in batches
        if item.get("frame_id") == 0 and "detections" in item
    }
    overlay_paths: list[Path] = []
    for episode_dir in sorted((run_root / "episodes").glob("*")):
        manifest = episode_dir / "episode_manifest.json"
        if not manifest.exists():
            continue
        output = run_root / "reports" / "overlays" / f"{episode_dir.name}.png"
        _overlay_episode(episode_dir, first_batches.get(episode_dir.name), output)
        overlay_paths.append(output)

    counts = metrics["counts"]
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else None
    lines = [
        "# Go2 open-vocabulary detection experiment report",
        "",
        "This report evaluates Grounding DINO Tiny on frozen MuJoCo frames. Detector inference was offline and could not affect locomotion.",
        "",
    ]
    if run_manifest is not None:
        matrix = run_manifest["matrix_timing"]
        primary_attempts = tuple(run_manifest["primary_attempts"].values())
        mpc_solves = sum(int(attempt["mpc_solves"]) for attempt in primary_attempts)
        solved_qps = sum(int(attempt["solver_status_counts"].get("solved", 0)) for attempt in primary_attempts)
        saturated_ticks = sum(int(attempt["torque_saturation"]["saturated_ticks"]) for attempt in primary_attempts)
        lines.extend(
            [
                "## Locomotion and capture status",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Valid primary episodes | {run_manifest['valid_primary_count']} / {len(run_manifest['primary_attempts'])} |",
                f"| Fallback episodes | {len(run_manifest['fallback_attempts'])} |",
                f"| Solved QPs | {solved_qps} / {mpc_solves} |",
                f"| Torque clipping events | {saturated_ticks} |",
                f"| MPC p95 / budget | {_fmt(matrix['mpc_total_p95_ms'], 3)} / {_fmt(matrix['mpc_budget_ms'], 1)} ms |",
                f"| Full control tick p95 / target | {_fmt(matrix['control_tick_p95_ms'], 3)} / {_fmt(matrix['control_tick_budget_ms'], 1)} ms |",
                f"| Matrix timing budget | {'PASS' if run_manifest['timing_budget_passed'] else 'FAIL'} |",
                "",
            ]
        )
        if not run_manifest["timing_budget_passed"]:
            lines.extend(
                [
                    "The 12 walking episodes were dynamically valid, but the full Python control-tick p95 exceeded its declared 5 ms target. This is the current simulation acceptance blocker; the controller was not retuned to hide it.",
                    "",
                ]
            )
    lines.extend([
        "## Overall baseball metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {_fmt(counts['precision'])} |",
        f"| Recall | {_fmt(counts['recall'])} |",
        f"| F1 | {_fmt(counts['f1'])} |",
        f"| TP / FP / FN | {counts['tp']} / {counts['fp']} / {counts['fn']} |",
        f"| Threshold-sweep AP | {_fmt(metrics['average_precision_threshold_sweep'])} |",
        f"| False positives per frame | {_fmt(metrics['false_positives_per_frame'])} |",
        f"| Absent-target false activation rate | {_fmt(metrics['absent_target_false_activation_rate'])} |",
        f"| Median matched IoU | {_fmt(metrics['localization']['matched_iou_median'])} |",
        f"| Inference median / p95 ms | {_fmt(metrics['latency_ms']['inference_median'], 1)} / {_fmt(metrics['latency_ms']['inference_p95'], 1)} |",
        f"| Detector error records | {metrics['detector_error_count']} |",
        "",
        f"At the declared 0.35 operating threshold, the detector emitted only {counts['tp'] + counts['fp']} baseball-labeled boxes. The apparently high precision is therefore conditional on sparse activations; the dominant result is {counts['fn']} misses and {_fmt(counts['recall'])} recall.",
        "",
        "## Episode persistence",
        "",
        "| Episode | Coverage | Longest hit | Longest miss | Toggles | Mean score including misses |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for episode_id, row in sorted(metrics["episode_metrics"].items()):
        lines.append(
            f"| {episode_id} | {_fmt(row['matched_frame_coverage'])} | {row['longest_matched_run']} | "
            f"{row['longest_missed_run']} | {row['toggle_count']} | {_fmt(row['mean_score_with_misses_zero'])} |"
        )
    lines.extend(["", "## First-frame overlays", ""])
    for path in overlay_paths:
        relative = path.relative_to(run_root / "reports")
        lines.extend([f"### {path.stem}", "", f"![{path.stem}]({relative.as_posix()})", ""])
    summary_plot = run_root / "evaluation" / "summary_metrics.png"
    if summary_plot.exists():
        lines.extend(
            [
                "## Metric plots",
                "",
                "![Threshold, persistence, and latency plots](../evaluation/summary_metrics.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These measurements apply to this deterministic MuJoCo corpus, this exact model revision, and the recorded CPU or MPS device. They do not establish real-time performance on NVIDIA hardware, controller robustness beyond the tested episodes, or physical Go2 safety.",
            "",
        ]
    )
    output = run_root / "reports" / "friday_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    write_json(run_root / "reports" / "report_manifest.json", {"report": str(output), "overlays": [str(path) for path in overlay_paths]})
    return output


def copy_latest_results_into_brief(run_path: str | Path) -> None:
    root = repository_root()
    marker = "<!-- GENERATED_RESULTS -->"
    brief = root / "docs" / "friday_brief.md"
    if not brief.exists():
        return
    run_root = Path(run_path).resolve()
    report = run_root / "reports" / "friday_report.md"
    metrics = json.loads((run_root / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    text = brief.read_text(encoding="utf-8")
    prefix = text.split(marker, 1)[0].rstrip()
    relative = Path("..") / report.resolve().relative_to(root.resolve())
    counts = metrics["counts"]
    generated = [
        prefix,
        "",
        marker,
        "",
        "## Measured matrix result",
        "",
        f"All {run_manifest['valid_primary_count']} of {len(run_manifest['primary_attempts'])} primary episodes were valid; "
        f"{len(run_manifest['fallback_attempts'])} fallbacks were needed. The matrix timing-budget check was "
        f"{'pass' if run_manifest['timing_budget_passed'] else 'fail'}.",
        f"MPC p95 was {run_manifest['matrix_timing']['mpc_total_p95_ms']:.3f} ms against 20.8 ms; the full control-tick p95 was "
        f"{run_manifest['matrix_timing']['control_tick_p95_ms']:.3f} ms against 5.0 ms.",
        "",
        "| Precision | Recall | F1 | TP | FP | FN | Absent-target activation | Inference p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {counts['precision']:.3f} | {counts['recall']:.3f} | {counts['f1']:.3f} | {counts['tp']} | "
        f"{counts['fp']} | {counts['fn']} | {metrics['absent_target_false_activation_rate']:.3f} | "
        f"{_fmt(metrics['latency_ms']['inference_p95'], 1)} ms |",
        "",
        f"The detector produced {counts['tp'] + counts['fp']} baseball-labeled boxes and missed {counts['fn']} evaluable baseball instances. "
        "The precision value should not be read as strong detection performance; very low recall is the dominant result.",
        "",
        f"The complete per-episode persistence table, overlays, localization metrics, threshold sweep, and timing plots are in "
        f"[{relative}]({relative.as_posix()}).",
        "",
    ]
    brief.write_text("\n".join(generated), encoding="utf-8")
