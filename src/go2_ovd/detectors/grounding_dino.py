from __future__ import annotations

import gc
import math
from pathlib import Path
import re
import resource
import time
from typing import Any

import numpy as np
from PIL import Image

from go2_ovd.config import load_yaml
from go2_ovd.io import read_jsonl, sha256_bytes, sha256_file, write_jsonl
from go2_ovd.schemas import Detection, DetectionBatch, DetectorRequest, ErrorBatch
from go2_ovd.schemas.detector import normalize_prompt


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        requested = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        return "cpu"
    if requested not in {"cpu", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    return requested


class GroundingDinoAdapter:
    def __init__(self, config: dict[str, Any], device: str = "auto") -> None:
        self.config = config
        self.model_id = str(config["model_id"])
        self.model_revision = str(config["model_revision"])
        self.prompts = tuple(normalize_prompt(item) for item in config["prompts"])
        self.box_threshold = float(config["box_threshold"])
        self.text_threshold = float(config["text_threshold"])
        if not all(math.isfinite(value) for value in (self.box_threshold, self.text_threshold)):
            raise ValueError("detector thresholds must be finite")
        if not (0.0 <= self.box_threshold <= 1.0 and 0.0 <= self.text_threshold <= 1.0):
            raise ValueError("detector thresholds must be within [0, 1]")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("model_revision must be an immutable 40-character git SHA")
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = _resolve_device(torch, device)
        kwargs = {
            "revision": self.model_revision,
            "local_files_only": bool(config.get("local_files_only", False)),
        }
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, **kwargs)
        loaded_revision = getattr(self.model.config, "_commit_hash", None)
        loaded_model_id = str(getattr(self.model.config, "name_or_path", ""))
        if loaded_revision != self.model_revision:
            raise RuntimeError(
                f"loaded model revision mismatch: expected {self.model_revision}, got {loaded_revision!r}"
            )
        if loaded_model_id != self.model_id:
            raise RuntimeError(f"loaded model identity mismatch: expected {self.model_id}, got {loaded_model_id!r}")
        self.model.eval().to(self.device)

    def _sync(self) -> None:
        if self.device == "mps":
            self.torch.mps.synchronize()

    def _prompt_index(self, raw_label: Any) -> int:
        if isinstance(raw_label, (int, np.integer)):
            index = int(raw_label)
            if 0 <= index < len(self.prompts):
                return index
        normalized = normalize_prompt(str(raw_label).strip(" ."))
        exact = [index for index, prompt in enumerate(self.prompts) if normalized == prompt]
        if exact:
            return exact[0]
        contained = [index for index, prompt in enumerate(self.prompts) if prompt in normalized or normalized in prompt]
        if len(contained) == 1:
            return contained[0]
        raise ValueError(f"cannot map Grounding DINO label {raw_label!r} to prompts {self.prompts!r}")

    def _prompt_token_indices(self, prompt_text: str, prompts: tuple[str, ...]) -> list[list[int]]:
        encoded = self.processor.tokenizer(prompt_text, add_special_tokens=True, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        spans: list[tuple[int, int]] = []
        cursor = 0
        for prompt in prompts:
            start = prompt_text.find(prompt, cursor)
            if start < 0:
                raise ValueError(f"prompt {prompt!r} is absent from constructed model text")
            end = start + len(prompt)
            spans.append((start, end))
            cursor = end
        result: list[list[int]] = []
        for start, end in spans:
            indices = [
                index for index, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start and token_start < end and token_end > start
            ]
            if not indices:
                raise ValueError(f"no tokenizer positions map to prompt span {(start, end)}")
            result.append(indices)
        return result

    def detect(self, request: DetectorRequest, *, run_id: str, episode_id: str) -> DetectionBatch:
        image = Image.fromarray(request.image_rgb, mode="RGB")
        prompt_text = ". ".join(request.prompts) + "."
        start = time.perf_counter()
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        preprocess_ms = (time.perf_counter() - start) * 1e3

        self._sync()
        start = time.perf_counter()
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        self._sync()
        inference_ms = (time.perf_counter() - start) * 1e3

        start = time.perf_counter()
        input_ids = inputs.get("input_ids")
        target_sizes = [(request.image_rgb.shape[0], request.image_rgb.shape[1])]
        try:
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            processed = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
        result = processed[0]
        raw_labels = result["text_labels"] if "text_labels" in result else result.get("labels", [])
        prompt_token_indices = self._prompt_token_indices(prompt_text, request.prompts)
        probabilities = self.torch.sigmoid(outputs.logits[0])
        kept_query_indices = self.torch.nonzero(probabilities.max(dim=-1).values > self.box_threshold).flatten().tolist()
        if len(kept_query_indices) != len(result["boxes"]):
            raise RuntimeError("postprocessor box count does not match Grounding DINO query filtering")
        detections: list[Detection] = []
        for query_index, raw_box, raw_score, raw_label in zip(kept_query_indices, result["boxes"], result["scores"], raw_labels):
            prompt_scores = [
                float(probabilities[query_index, indices].max().detach().cpu().item())
                for indices in prompt_token_indices
            ]
            prompt_id = int(np.argmax(prompt_scores))
            if prompt_scores[prompt_id] <= self.text_threshold:
                continue
            box = tuple(float(value) for value in raw_box.detach().cpu().tolist())
            detection = Detection(
                prompt_id=prompt_id,
                label=self.prompts[prompt_id],
                score=float(raw_score.detach().cpu().item()),
                bbox_xyxy_px=box,
                mask_rle=None,
                adapter_metadata={
                    "raw_label": str(raw_label),
                    "grounding_query_index": int(query_index),
                    "assigned_prompt_token_score": prompt_scores[prompt_id],
                    "prompt_token_scores": {
                        prompt: score for prompt, score in zip(request.prompts, prompt_scores)
                    },
                },
            ).validated(request.image_rgb.shape[1], request.image_rgb.shape[0], request.prompts)
            detections.append(detection)
        postprocess_ms = (time.perf_counter() - start) * 1e3
        return DetectionBatch(
            schema_version="1.0",
            run_id=run_id,
            episode_id=episode_id,
            frame_id=request.frame_id,
            image_sha256=request.image_sha256,
            adapter_name="grounding_dino",
            model_id=self.model_id,
            model_revision=self.model_revision,
            device=self.device,
            prompts=request.prompts,
            preprocessing_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocessing_ms=postprocess_ms,
            detections=tuple(detections),
            peak_process_memory_mb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)),
        ).sorted()

    def warmup(self, request: DetectorRequest, count: int) -> None:
        for _ in range(max(0, count)):
            self.detect(request, run_id="warmup", episode_id="warmup")
        gc.collect()


def run_detector(config_path: str | Path, run_path: str | Path, device: str = "auto") -> Path:
    config = load_yaml(config_path)
    primary_threshold = float(config["box_threshold"])
    archival_threshold = float(config.get("archival_box_threshold", primary_threshold))
    if archival_threshold > primary_threshold:
        raise ValueError("archival_box_threshold must not exceed box_threshold")
    inference_config = dict(config)
    inference_config["box_threshold"] = archival_threshold
    run_root = Path(run_path)
    frames: list[tuple[str, Path, dict[str, Any]]] = []
    for frame_file in sorted((run_root / "episodes").glob("*/frames.jsonl")):
        for record in read_jsonl(frame_file):
            frames.append((record["episode_id"], frame_file.parent, record))
    if not frames:
        raise ValueError(f"no frames found under {run_root / 'episodes'}")

    adapter = GroundingDinoAdapter(inference_config, device=device)
    prompts = tuple(normalize_prompt(item) for item in config["prompts"])

    def request_for(item: tuple[str, Path, dict[str, Any]]) -> DetectorRequest:
        _, episode_dir, frame = item
        image_path = episode_dir / frame["rgb_path"]
        file_digest = sha256_file(image_path)
        if file_digest != frame["rgb_sha256"]:
            raise ValueError(f"PNG SHA-256 mismatch for {image_path}")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        pixel_digest = sha256_bytes(image.tobytes(order="C"))
        if pixel_digest != frame["pixel_sha256"]:
            raise ValueError(f"RGB pixel SHA-256 mismatch for {image_path}")
        return DetectorRequest(frame["frame_id"], image, pixel_digest, prompts)

    fallback_reason = None
    warmup_request = request_for(frames[0])
    try:
        adapter.warmup(warmup_request, int(config.get("warmup_frames", 0)))
    except Exception as exc:
        if adapter.device != "mps":
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        del adapter
        gc.collect()
        adapter = GroundingDinoAdapter(inference_config, device="cpu")
        adapter.warmup(warmup_request, int(config.get("warmup_frames", 0)))
    batches: list[DetectionBatch | ErrorBatch] = []
    run_id = str(frames[0][2]["run_id"])
    for episode_id, episode_dir, frame in frames:
        try:
            batches.append(adapter.detect(request_for((episode_id, episode_dir, frame)), run_id=run_id, episode_id=episode_id))
        except Exception as exc:
            batches.append(
                ErrorBatch(
                    schema_version="1.0",
                    run_id=run_id,
                    episode_id=episode_id,
                    frame_id=int(frame["frame_id"]),
                    image_sha256=str(frame["pixel_sha256"]),
                    adapter_name="grounding_dino",
                    model_id=str(config["model_id"]),
                    model_revision=str(config["model_revision"]),
                    device=adapter.device,
                    prompts=prompts,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
    archival_output = run_root / "detections" / "grounding_dino_tiny_archival_0p05.jsonl"
    write_jsonl(archival_output, batches)
    primary_batches: list[DetectionBatch | ErrorBatch] = []
    for batch in batches:
        if isinstance(batch, ErrorBatch):
            primary_batches.append(batch)
        else:
            primary_batches.append(
                DetectionBatch(
                    **{
                        **batch.__dict__,
                        "detections": tuple(item for item in batch.detections if item.score >= primary_threshold),
                    }
                ).sorted()
            )
    output = run_root / "detections" / "grounding_dino_tiny.jsonl"
    write_jsonl(output, primary_batches)
    from go2_ovd.io import write_json
    write_json(
        run_root / "detections" / "run_metadata.json",
        {
            "model_id": adapter.model_id,
            "model_revision": adapter.model_revision,
            "requested_device": device,
            "actual_device": adapter.device,
            "mps_fallback_reason": fallback_reason,
            "primary_box_threshold": primary_threshold,
            "archival_box_threshold": archival_threshold,
            "primary_output": str(output.name),
            "archival_output": str(archival_output.name),
        },
    )
    return output
