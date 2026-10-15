import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.benchmarks.vbench import run_i2v


class VBenchRunnerTests(unittest.TestCase):
    def _args(self, root, full_info, output_full_info, **overrides):
        values = {
            "full_info": str(full_info),
            "image_root": str(root / "images"),
            "resolution": "16-9",
            "save_path": str(root / "videos"),
            "output_full_info": str(output_full_info),
            "endpoint": "http://127.0.0.1:14830/v1/video/generation",
            "model": "wan2_2",
            "size": "832*480",
            "num_inference_steps": 40,
            "guidance_scale": 1.5,
            "guidance_scale_2": None,
            "num_videos_per_prompt": None,
            "max_sequence_length": None,
            "num_frames": 81,
            "fps": 16.0,
            "seconds": None,
            "boundary_ratio": None,
            "flow_shift": None,
            "true_cfg_scale": None,
            "seed": 42,
            "request_timeout": 10.0,
            "limit": None,
            "allow_failures": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_generate_encodes_image_and_writes_manifest_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "images" / "first.png").write_bytes(b"png-bytes")
            full_info = root / "full_info.json"
            full_info.write_text(
                json.dumps([{"image_name": "first.png", "prompt_en": "A test scene"}]),
                encoding="utf-8",
            )
            output_manifest = root / "run" / "videos.json"
            args = self._args(root, full_info, output_manifest)
            encoded_video = base64.b64encode(b"fake-mp4").decode("ascii")
            with mock.patch.object(
                run_i2v,
                "_request_video",
                return_value={"output": {"results": [{"video": encoded_video}]}},
            ) as request:
                self.assertEqual(run_i2v.generate(args), 0)

            request.assert_called_once()
            payload = request.call_args.args[1]
            self.assertEqual(payload["input"]["prompt"], "A test scene")
            media_sources = payload["input"]["media_sources"]
            self.assertEqual(len(media_sources), 1)
            self.assertEqual(media_sources[0]["type"], "base64")
            self.assertEqual(media_sources[0]["name"], "image")
            self.assertEqual(media_sources[0]["modality"], "image")
            self.assertEqual(
                base64.b64decode(media_sources[0]["base64"]), b"png-bytes"
            )
            self.assertEqual(payload["parameters"]["seed"], 42)
            self.assertEqual((root / "videos" / "00000.mp4").read_bytes(), b"fake-mp4")
            manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
            self.assertTrue(manifest[0]["video_list"][0].endswith("00000.mp4"))
            metrics = json.loads(
                (root / "generation_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["generated"], 1)
            self.assertEqual(metrics["failed"], 0)

    def test_missing_image_is_reported_and_fails_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            full_info = root / "full_info.json"
            full_info.write_text(
                json.dumps([{"image_name": "missing.png", "prompt": "scene"}]),
                encoding="utf-8",
            )
            args = self._args(root, full_info, root / "run" / "videos.json")
            self.assertEqual(run_i2v.generate(args), 1)
            metrics = json.loads(
                (root / "generation_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["failed"], 1)
            self.assertEqual(len(metrics["failures"]), 1)

    def test_existing_video_is_resumed_without_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            existing = root / "videos" / "old.mp4"
            existing.parent.mkdir()
            existing.write_bytes(b"old")
            full_info = root / "full_info.json"
            full_info.write_text(
                json.dumps(
                    [{
                        "image_name": "unused.png",
                        "prompt": "scene",
                        "video_list": [str(existing)],
                    }]
                ),
                encoding="utf-8",
            )
            args = self._args(root, full_info, root / "run" / "videos.json")
            with mock.patch.object(run_i2v, "_request_video") as request:
                self.assertEqual(run_i2v.generate(args), 0)
            request.assert_not_called()
            metrics = json.loads(
                (root / "generation_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["resumed"], 1)
            self.assertEqual(metrics["generated"], 0)

    def test_default_output_video_is_resumed_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            existing = root / "videos" / "00000.mp4"
            existing.parent.mkdir()
            existing.write_bytes(b"old")
            full_info = root / "full_info.json"
            full_info.write_text(
                json.dumps([{"image_name": "unused.png", "prompt_en": "scene"}]),
                encoding="utf-8",
            )
            args = self._args(root, full_info, root / "run" / "videos.json")
            with mock.patch.object(run_i2v, "_request_video") as request:
                self.assertEqual(run_i2v.generate(args), 0)
            request.assert_not_called()
            metrics = json.loads(
                (root / "generation_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["resumed"], 1)
            self.assertEqual(metrics["generated"], 0)


if __name__ == "__main__":
    unittest.main()
