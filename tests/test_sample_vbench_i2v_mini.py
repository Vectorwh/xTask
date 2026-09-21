import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_sampler():
    path = Path(__file__).resolve().parents[1] / "tools" / "sample_vbench_i2v_mini_full_info.py"
    spec = importlib.util.spec_from_file_location("sample_vbench_i2v_mini_full_info", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SampleVBenchI2VMiniTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_sampler()

    def test_camera_motion_parses_seven_types_from_prompt(self):
        mapping = {
            "camera pans left": "pan_left",
            "camera pans right": "pan_right",
            "camera tilts up": "tilt_up",
            "camera tilts down": "tilt_down",
            "camera zooms in": "zoom_in",
            "camera zooms out": "zoom_out",
            "camera static": "static",
        }
        for phrase, name in mapping.items():
            record = {
                "prompt_en": f"a lake, {phrase}",
                "dimension": ["camera_motion"],
                "image_type": "scenery",
            }
            self.assertEqual(self.mod.camera_motion_type(record), name)
        subject = {
            "prompt_en": "a cat, camera pans left",
            "dimension": ["i2v_subject", "subject_consistency"],
            "image_type": "animal",
        }
        self.assertEqual(self.mod.camera_motion_type(subject), "none")

    def test_stratum_key_is_dimension_image_type_camera(self):
        record = {
            "prompt_en": "a building, camera zooms in",
            "dimension": ["camera_motion"],
            "image_type": "architecture",
        }
        self.assertEqual(
            self.mod.stratum_key(record),
            ("camera_motion", "architecture", "zoom_in"),
        )

    def test_filter_downloaded_keeps_only_existing_crop_or_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop = root / "crop" / "16-9"
            crop.mkdir(parents=True)
            (crop / "kept.jpg").write_bytes(b"img")
            records = [
                {"prompt_en": "a", "dimension": ["i2v_subject"], "image_type": "other", "image_name": "kept.jpg"},
                {"prompt_en": "b", "dimension": ["i2v_subject"], "image_type": "other", "image_name": "missing.jpg"},
            ]
            kept = self.mod.filter_downloaded(records, root, "16-9")
            self.assertEqual([item["image_name"] for item in kept], ["kept.jpg"])

    def test_hamilton_allocate_hits_target_and_is_reproducible(self):
        sizes = {
            ("camera_motion", "scenery", "pan_left"): 40,
            ("camera_motion", "scenery", "static"): 8,
            ("i2v_subject", "animal", "none"): 46,
            ("i2v_background", "indoor", "none"): 21,
        }
        first = self.mod.hamilton_allocate(sizes, 107)
        second = self.mod.hamilton_allocate(sizes, 107)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 107)
        for key, count in first.items():
            self.assertGreaterEqual(count, 0)
            self.assertLessEqual(count, sizes[key])

    def test_centroid_pick_takes_nearest_member_of_each_cluster(self):
        import numpy as np

        embeddings = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 0.0],
                [10.1, 0.0],
            ],
            dtype=np.float64,
        )
        picked = self.mod.pick_centroid_representatives(embeddings, k=2, random_state=42)
        self.assertEqual(len(picked), 2)
        self.assertEqual(len(set(picked)), 2)
        self.assertTrue(bool(set(picked) & {0, 1}))
        self.assertTrue(bool(set(picked) & {2, 3}))

    def test_sample_records_keeps_original_order_and_schema(self):
        import numpy as np

        records = []
        for index in range(10):
            records.append(
                {
                    "prompt_en": f"scene {index}, camera static",
                    "dimension": ["camera_motion"],
                    "image_type": "scenery",
                    "image_name": f"{index}.jpg",
                    "extra": index,
                }
            )
        embeddings = np.eye(10, dtype=np.float64)
        selected = self.mod.sample_records(records, embeddings, target=3, random_state=42)
        self.assertEqual(len(selected), 3)
        names = [item["image_name"] for item in selected]
        self.assertEqual(names, sorted(names, key=lambda name: int(name.split(".")[0])))
        self.assertTrue(all("extra" in item for item in selected))

    def test_refuses_to_write_default_full_info_symlink(self):
        args = self.mod.parse_args(
            [
                "--full-info",
                str(self.mod.DEFAULT_FULL_INFO),
                "--output",
                str(Path(self.mod.DEFAULT_DATA_HOME) / "full_info.json"),
                "--dry-run",
            ]
        )
        with self.assertRaises(SystemExit):
            self.mod.run(args)

    def test_dry_run_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop = root / "crop" / "16-9"
            crop.mkdir(parents=True)
            (crop / "a.jpg").write_bytes(b"img")
            full_info = root / "full_info.json"
            output = root / "vbench2_i2v_mini_full_info.json"
            full_info.write_text(
                json.dumps(
                    [
                        {
                            "prompt_en": "a lake, camera static",
                            "dimension": ["camera_motion"],
                            "image_type": "scenery",
                            "image_name": "a.jpg",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = self.mod.parse_args(
                [
                    "--full-info",
                    str(full_info),
                    "--image-root",
                    str(root),
                    "--output",
                    str(output),
                    "--target",
                    "1",
                    "--dry-run",
                ]
            )
            self.assertEqual(self.mod.run(args), 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
