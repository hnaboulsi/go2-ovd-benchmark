from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="go2-ovd", description="Simulation-first Go2 open-vocabulary detection experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    controller = sub.add_parser("validate-controller")
    controller.add_argument("--repeats", type=int, default=3)
    camera = sub.add_parser("validate-camera")
    camera.add_argument("--config", type=Path)
    capture = sub.add_parser("capture")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    detect = sub.add_parser("detect")
    detect.add_argument("--config", type=Path, required=True)
    detect.add_argument("--frames", type=Path, required=True)
    detect.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-controller":
        from go2_ovd.simulation.runner import validate_controller
        result = validate_controller(args.repeats)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.command == "validate-camera":
        from go2_ovd.simulation.camera_validation import validate_camera
        result = validate_camera(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "capture":
        from go2_ovd.simulation.runner import capture_matrix
        result = capture_matrix(args.config, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        expected = len(result["primary_attempts"])
        usable = result["valid_primary_count"] + result["valid_fallback_count"]
        return 0 if usable >= expected and result["timing_budget_passed"] else 1
    if args.command == "detect":
        from go2_ovd.detectors.grounding_dino import run_detector
        print(run_detector(args.config, args.frames, args.device))
        return 0
    if args.command == "evaluate":
        from go2_ovd.evaluation.metrics import evaluate_run
        result = evaluate_run(args.run, args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        from go2_ovd.report import copy_latest_results_into_brief, create_report
        result = create_report(args.run)
        copy_latest_results_into_brief(args.run)
        print(result)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
