from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from go2_ovd.io import sha256_bytes


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


@dataclass(frozen=True)
class DetectorRequest:
    frame_id: int
    image_rgb: np.ndarray
    image_sha256: str
    prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be nonnegative")
        if not isinstance(self.image_rgb, np.ndarray) or self.image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be a uint8 numpy array")
        if self.image_rgb.ndim != 3 or self.image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape H x W x 3")
        normalized = tuple(normalize_prompt(item) for item in self.prompts)
        if not normalized or any(not item for item in normalized):
            raise ValueError("prompts must be nonempty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("prompts must be unique after normalization")
        actual = sha256_bytes(self.image_rgb.tobytes(order="C"))
        if actual != self.image_sha256:
            raise ValueError("image_sha256 does not match RGB pixel bytes")


@dataclass(frozen=True)
class Detection:
    prompt_id: int
    label: str
    score: float
    bbox_xyxy_px: tuple[float, float, float, float]
    mask_rle: dict[str, Any] | None = None
    adapter_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt_id < 0:
            raise ValueError("prompt_id must be nonnegative")
        if not self.label or self.label != normalize_prompt(self.label):
            raise ValueError("label must be a nonempty normalized string")
        if not math.isfinite(self.score):
            raise ValueError("detection score must be finite")
        if len(self.bbox_xyxy_px) != 4 or not all(math.isfinite(float(value)) for value in self.bbox_xyxy_px):
            raise ValueError("detection box must contain four finite coordinates")
        x1, y1, x2, y2 = self.bbox_xyxy_px
        if x1 >= x2 or y1 >= y2:
            raise ValueError("detection box must have positive area")

    def validated(self, width: int, height: int, prompts: tuple[str, ...]) -> "Detection":
        if not (0 <= self.prompt_id < len(prompts)):
            raise ValueError("prompt_id is out of range")
        label = normalize_prompt(self.label)
        if label != normalize_prompt(prompts[self.prompt_id]):
            raise ValueError("detection label does not match prompt_id")
        if not math.isfinite(self.score):
            raise ValueError("detection score must be finite")
        x1, y1, x2, y2 = (float(value) for value in self.bbox_xyxy_px)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("detection box must be finite")
        clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise ValueError("detection box has zero area after clipping")
        return Detection(self.prompt_id, label, float(self.score), clipped, self.mask_rle, dict(self.adapter_metadata))


@dataclass(frozen=True)
class DetectionBatch:
    schema_version: str
    run_id: str
    episode_id: str
    frame_id: int
    image_sha256: str
    adapter_name: str
    model_id: str
    model_revision: str
    device: str
    prompts: tuple[str, ...]
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    detections: tuple[Detection, ...]
    peak_process_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be nonnegative")
        if not all(math.isfinite(value) and value >= 0.0 for value in (self.preprocessing_ms, self.inference_ms, self.postprocessing_ms)):
            raise ValueError("detector timings must be finite and nonnegative")
        if self.peak_process_memory_mb is not None and (not math.isfinite(self.peak_process_memory_mb) or self.peak_process_memory_mb < 0.0):
            raise ValueError("peak process memory must be finite and nonnegative")
        normalized = tuple(normalize_prompt(item) for item in self.prompts)
        if normalized != self.prompts or not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("batch prompts must be normalized, nonempty, and unique")

    def sorted(self) -> "DetectionBatch":
        ordered = tuple(sorted(self.detections, key=lambda item: (-item.score, item.prompt_id, *item.bbox_xyxy_px)))
        return DetectionBatch(**{**self.__dict__, "detections": ordered})


@dataclass(frozen=True)
class ErrorBatch:
    schema_version: str
    run_id: str
    episode_id: str
    frame_id: int
    image_sha256: str
    adapter_name: str
    model_id: str
    model_revision: str
    device: str
    prompts: tuple[str, ...]
    error_type: str
    error_message: str
