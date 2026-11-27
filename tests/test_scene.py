from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from go2_ovd.config import repository_root
from go2_ovd.simulation.camera import intrinsics, mask_box
from go2_ovd.simulation.runner import _seed
from go2_ovd.simulation.scene_builder import mesh_bounds


class SceneTests(unittest.TestCase):
    def test_intrinsics(self) -> None:
        value = intrinsics(640, 480, 70)
        self.assertAlmostEqual(value["fx"], 342.75552161810754)
        self.assertEqual((value["cx"], value["cy"]), (320.0, 240.0))

    def test_half_open_mask_box(self) -> None:
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:5, 3:9] = True
        self.assertEqual(mask_box(mask), (3.0, 2.0, 9.0, 5.0))
        self.assertIsNone(mask_box(np.zeros((2, 2), dtype=bool)))

    def test_episode_seed_is_stable_and_identifier_specific(self) -> None:
        self.assertEqual(_seed(20260918, "angle_000_target"), _seed(20260918, "angle_000_target"))
        self.assertNotEqual(_seed(20260918, "angle_000_target"), _seed(20260918, "angle_p30_target"))

    def test_ycb_mesh_units_are_plausible_meters(self) -> None:
        root = repository_root()
        expected = {
            "055_baseball": np.array([0.073078, 0.073712, 0.072563]),
            "054_softball": np.array([0.095370, 0.096174, 0.094591]),
        }
        for object_id, extent in expected.items():
            mesh = root / ".cache" / "assets" / "ycb" / object_id / "google_16k" / "textured.obj"
            minimum, maximum = mesh_bounds(mesh)
            self.assertTrue(np.allclose(maximum - minimum, extent, atol=1e-6))
            self.assertTrue(np.all((maximum - minimum) > 0.05))
            self.assertTrue(np.all((maximum - minimum) < 0.15))


if __name__ == "__main__":
    unittest.main()
