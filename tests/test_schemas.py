from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np

from go2_ovd.config import config_sha256
from go2_ovd.io import read_jsonl, sha256_bytes, write_jsonl
from go2_ovd.schemas import Detection, DetectionBatch, DetectorRequest, ErrorBatch
from go2_ovd.detectors.grounding_dino import GroundingDinoAdapter


class DetectorSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((4, 6, 3), dtype=np.uint8)
        self.digest = sha256_bytes(self.image.tobytes(order="C"))

    def test_request_accepts_ordered_unique_prompts(self) -> None:
        request = DetectorRequest(2, self.image, self.digest, (" Baseball ", "softball"))
        self.assertEqual(request.prompts, (" Baseball ", "softball"))

    def test_request_rejects_wrong_dtype_shape_prompt_and_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "uint8"):
            DetectorRequest(0, self.image.astype(np.float32), self.digest, ("baseball",))
        with self.assertRaisesRegex(ValueError, "shape"):
            DetectorRequest(0, self.image[:, :, 0], self.digest, ("baseball",))
        with self.assertRaisesRegex(ValueError, "unique"):
            DetectorRequest(0, self.image, self.digest, ("Baseball", " baseball "))
        with self.assertRaisesRegex(ValueError, "does not match"):
            DetectorRequest(0, self.image, "0" * 64, ("baseball",))

    def test_box_clipping_and_zero_area_rejection(self) -> None:
        clipped = Detection(0, "baseball", 0.5, (-2, -1, 9, 8)).validated(6, 4, ("baseball",))
        self.assertEqual(clipped.bbox_xyxy_px, (0.0, 0.0, 6.0, 4.0))
        with self.assertRaisesRegex(ValueError, "zero area"):
            Detection(0, "baseball", 0.5, (8, 0, 9, 1)).validated(6, 4, ("baseball",))

    def test_detection_sorting_and_model_identity(self) -> None:
        detections = (
            Detection(1, "softball", 0.7, (1, 2, 3, 4)),
            Detection(0, "baseball", 0.9, (2, 2, 3, 3)),
            Detection(0, "baseball", 0.9, (1, 2, 3, 3)),
        )
        batch = DetectionBatch(
            "1.0", "run", "episode", 0, self.digest, "grounding_dino",
            "IDEA-Research/grounding-dino-tiny", "a" * 40, "cpu",
            ("baseball", "softball"), 1.0, 2.0, 3.0, detections,
        ).sorted()
        self.assertEqual([item.prompt_id for item in batch.detections], [0, 0, 1])
        self.assertEqual(batch.detections[0].bbox_xyxy_px[0], 1)
        self.assertEqual(batch.model_revision, "a" * 40)

    def test_jsonl_round_trip_and_explicit_error_record(self) -> None:
        error = ErrorBatch(
            "1.0", "run", "episode", 3, self.digest, "grounding_dino",
            "model", "b" * 40, "cpu", ("baseball",), "RuntimeError", "deliberate",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "values.jsonl"
            write_jsonl(path, [error])
            loaded = read_jsonl(path)
        self.assertEqual(loaded, [asdict(error) | {"prompts": ["baseball"]}])
        self.assertNotIn("detections", loaded[0])

    def test_configuration_and_image_hashing_are_stable(self) -> None:
        self.assertEqual(config_sha256({"b": 2, "a": 1}), config_sha256({"a": 1, "b": 2}))
        changed = self.image.copy()
        changed[0, 0, 0] = 1
        self.assertNotEqual(self.digest, sha256_bytes(changed.tobytes(order="C")))

    def test_nonimmutable_model_revision_fails_before_model_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable"):
            GroundingDinoAdapter(
                {
                    "model_id": "does-not-matter",
                    "model_revision": "main",
                    "prompts": ["baseball"],
                    "box_threshold": 0.35,
                    "text_threshold": 0.25,
                },
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
