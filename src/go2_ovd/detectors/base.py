from __future__ import annotations

from typing import Protocol, Sequence

from go2_ovd.schemas import DetectionBatch, DetectorRequest, FrameRecord


class DetectorAdapter(Protocol):
    def detect(self, request: DetectorRequest, *, run_id: str, episode_id: str) -> DetectionBatch:
        ...


class RejectionModule(Protocol):
    """Reserved interface; v1 deliberately provides no learned filter."""

    def filter(
        self,
        current: DetectionBatch,
        history: Sequence[DetectionBatch],
        frame_context: FrameRecord,
    ) -> DetectionBatch:
        ...
