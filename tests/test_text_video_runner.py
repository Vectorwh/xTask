"""Fresh request, embedding, media and archived result tests."""
import base64
import hashlib
import json
import io
import urllib.error
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from xtask.benchmarks.vbench.text_video import run as runner
from xtask.benchmarks.vbench.text_video.config import normalize_config
from xtask.benchmarks.vbench.text_video.embedding import load_embedding, validate_cfg
from xtask.benchmarks.vbench.text_video.media import validate_media
from test_text_video_config import config


def manifest(root, records=None):
    path = root / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (records or [{"id": "one", "prompt": "A dog running"}])))
    return path


def tensor(root, dimension=2, seq=2):
    path = root / "tensor.json"
    data = {"datatype": "FP32", "shape": [seq, dimension], "contents": {"fp32_contents": [.1] * (seq * dimension)}}
    path.write_text(json.dumps(data))
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


ENCODER = {"checkpoint": "frozen_encoder_hash", "tokenizer": "frozen_tokenizer_hash", "template_sha256": "a" * 64,
           "feature": "qwen3_vl_pre_norm_residual", "dtype": "FP32"}


class ManifestTests(unittest.TestCase):
    def test_images_and_unknown_context_cannot_silently_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            for key in ("image", "media_sources", "audio", "video", "context"):
                path = manifest(Path(directory), [{"id": "x", "prompt": "dog", key: "foo"}])
                with self.assertRaises(ValueError):
                    runner.read_manifest(path)

    def test_duplicate_ids_empty_prompt_and_empty_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = manifest(Path(directory), [{"id": "x", "prompt": "dog"}] * 2)
            with self.assertRaises(ValueError):
                runner.read_manifest(path)

    def test_text_cfg_requires_negative_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = config(manifest=str(manifest(root)))
            value["generation"]["guidance_scale"] = 4
            with self.assertRaises(ValueError):
                runner.prepare_requests(normalize_config(value), root)

    def test_cfg_pairs_keep_same_seed_and_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = config(scenario="cfg_pair", model="Joy", endpoint="http://localhost:8018/v1/video/generation",
                           manifest=str(manifest(root, [{"id": "x", "prompt": "Person walking", "negative_prompt": "vehicles", "neutral_negative_prompt": "artifacts"}])))
            value["generation"]["guidance_scale"] = 4
            requests = runner.prepare_requests(normalize_config(value), root)
            self.assertEqual([row["arm"] for row in requests], ["reference", "target"])
            self.assertEqual(requests[0]["parameters"], requests[1]["parameters"])
            self.assertNotEqual(requests[0]["payload"]["input"], requests[1]["payload"]["input"])


class EmbeddingTests(unittest.TestCase):
    def test_hash_shape_dtype_and_feature_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor = tensor(root)
            value, provenance = load_embedding(descriptor, root, 2, ENCODER)
            self.assertEqual(value["shape"], [2, 2])
            self.assertFalse(provenance["encoder_provenance_verified"])
            with self.assertRaises(ValueError):
                load_embedding(descriptor, root, 5120, ENCODER)
            with self.assertRaises(ValueError):
                load_embedding({**descriptor, "sha256": "0" * 64}, root, 2, ENCODER)
            with self.assertRaises(ValueError):
                load_embedding(descriptor, root, 2, {**ENCODER, "feature": "H3"})

    def test_cfg_positive_only_features_never_silently_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value, _ = load_embedding(tensor(root), root, 2, ENCODER)
            with self.assertRaises(ValueError):
                validate_cfg(value, None, 4)
            value["parameters"] = {"joyo_text_seqlens": {"int64_tensor_param": [1, 1]}}
            validate_cfg(value, None, 4)
            with self.assertRaises(ValueError):
                validate_cfg(value, value, 4)

    def test_external_inputs_snapshot_and_no_prompt_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = manifest(root, [{"id": "x", "prompt": "dog", "embedding": tensor(root), "encoder": ENCODER}])
            value = normalize_config(config(scenario="embedding", manifest=str(path), embedding_dim=2, model="Joy", endpoint="http://local"))
            prepared = runner.prepare_requests(value, root)
            self.assertEqual(set(prepared[0]["payload"]["input"]), {"prompt_embed"})
            self.assertTrue((root / "embeddings/00000_prompt_embed.json").is_file())


class MediaTests(unittest.TestCase):
    def _probe(self, audio=True):
        streams = [{"codec_type": "video", "width": 512, "height": 288, "nb_read_frames": "81",
                    "avg_frame_rate": "16/1", "duration": str(81 / 16)}]
        if audio:
            streams.append({"codec_type": "audio", "sample_rate": "44100", "channels": 2, "duration": str(81 / 16)})
        return {"streams": streams}

    def test_audio_required_and_full_decode(self):
        value = config(task="t2av")
        value["generation"]["sampling_rate"] = 44100
        value = normalize_config(value)
        with mock.patch("subprocess.run", side_effect=[SimpleNamespace(stdout=json.dumps(self._probe())), SimpleNamespace()]) as command:
            result = validate_media(Path("clip.mp4"), value["generation"], value["media"])
        self.assertTrue(result["fully_decoded"])
        self.assertIn("-xerror", command.call_args.args[0])
        with mock.patch("subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(self._probe(False)))):
            with self.assertRaises(ValueError):
                validate_media(Path("clip.mp4"), value["generation"], value["media"])

    def test_wrong_frames_and_unexpected_audio_fail(self):
        value = normalize_config(config())
        with mock.patch("subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(self._probe()))):
            with self.assertRaises(ValueError):
                validate_media(Path("clip.mp4"), value["generation"], value["media"])
        probe = self._probe(False)
        probe["streams"][0]["nb_read_frames"] = "80"
        with mock.patch("subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(probe))):
            with self.assertRaises(ValueError):
                validate_media(Path("clip.mp4"), value["generation"], value["media"])


class RunArchiveTests(unittest.TestCase):
    def _request(self, prepared, settings, directory, index):
        path = directory / "videos" / prepared["filename"]
        path.write_bytes(b"fixture video")
        return {k: v for k, v in prepared.items() if k != "payload"} | {
            "success": True, "video_path": str(path.relative_to(directory)), "video_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "started": float(index + 1), "finished": float(index + 2), "e2el_ms": 1000., "request_index": index}

    def test_two_invocations_do_not_reuse_outputs_or_clobber_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = config(manifest=str(manifest(root)), model="Joy", endpoint="http://local")
            with mock.patch.object(runner, "_request", side_effect=self._request) as request, mock.patch.object(runner, "validate_media", return_value={"fully_decoded": True}):
                one, first = runner.run(settings, root / "out")
                two, second = runner.run(settings, root / "out")
            self.assertEqual(request.call_count, 2)
            self.assertNotEqual(first, second)
            self.assertTrue(one["complete"] and two["complete"])
            self.assertTrue((first / "requests.json").exists())
            self.assertNotIn("total_score", one)
            self.assertNotIn("output_tps_avg", one["performance"])

    def test_decode_failure_preserves_video_and_failed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = config(manifest=str(manifest(root)), model="Joy", endpoint="http://local")
            with mock.patch.object(runner, "_request", side_effect=self._request), mock.patch.object(runner, "validate_media", side_effect=ValueError("bad frames")):
                result, archived = runner.run(settings, root / "out")
            self.assertFalse(result["complete"])
            self.assertEqual(result["failed"], 1)
            self.assertTrue(list((archived / "videos").glob("*.mp4")))
            self.assertTrue(result["issues"])

    def test_no_evaluator_never_claims_cfg_capability_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = config(scenario="cfg_pair", manifest=str(manifest(root, [{"id": "x", "prompt": "dog", "negative_prompt": "car", "neutral_negative_prompt": "artifacts"}])), model="Joy", endpoint="http://local")
            settings["generation"]["guidance_scale"] = 4
            with mock.patch.object(runner, "_request", side_effect=self._request), mock.patch.object(runner, "validate_media", return_value={}):
                result, _ = runner.run(settings, root / "out")
            self.assertEqual(result["checks_status"]["negative_control"], "not_scored")
            self.assertNotIn("negative_control", result)

    def test_http_error_keeps_bounded_server_diagnostics_without_retry(self):
        prepared = {"payload": {"model": "Joy", "input": {"prompt": "dog"}},
                    "endpoint": "http://local", "filename": "x.mp4"}
        settings = normalize_config(config())
        for body, truncated in ((b'{"message": "Model not supported"}', False), (b"x" * 10000, True)):
            error = urllib.error.HTTPError("http://local", 404, "Not Found", {}, io.BytesIO(body))
            opener = mock.MagicMock()
            opener.open.side_effect = error
            with self.subTest(truncated=truncated), tempfile.TemporaryDirectory() as directory, mock.patch(
                "urllib.request.build_opener", return_value=opener
            ):
                record = runner._request(prepared, settings, Path(directory), 0)
            self.assertFalse(record["success"])
            self.assertEqual(record["http_status"], 404)
            expected = body[:8192].decode() if truncated else json.dumps(json.loads(body))
            self.assertEqual(record["http_error_body"], expected)
            self.assertEqual(record["http_error_body_truncated"], truncated)
            opener.open.assert_called_once()

    def test_http_error_redacts_credentials(self):
        prepared = {"payload": {"model": "Joy", "input": {"prompt": "dog"}},
                    "endpoint": "http://local", "filename": "x.mp4"}
        settings = normalize_config(config())
        settings["api_key"] = "private-client-key"
        body = b'{"api_key": "server-secret", "message": "private-client-key"}'
        error = urllib.error.HTTPError("http://local", 401, "Unauthorized", {}, io.BytesIO(body))
        opener = mock.MagicMock()
        opener.open.side_effect = error
        with tempfile.TemporaryDirectory() as directory, mock.patch("urllib.request.build_opener", return_value=opener):
            record = runner._request(prepared, settings, Path(directory), 0)
        self.assertNotIn("server-secret", record["http_error_body"])
        self.assertNotIn("private-client-key", record["http_error_body"])
        self.assertEqual(record["http_status"], 401)

    def test_http_transport_native_schema(self):
        prepared = {"payload": {"model": "Joy", "input": {"prompt": "dog"}}, "endpoint": "http://local", "filename": "x.mp4"}
        settings = normalize_config(config())
        response = mock.MagicMock()
        response.read.return_value = json.dumps({"id": "x", "output": {"results": [{"video": {"type": "base64", "base64": base64.b64encode(b"video").decode()}}]}}).encode()
        opener = mock.MagicMock()
        opener.open.return_value.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory, mock.patch("urllib.request.build_opener", return_value=opener):
            root = Path(directory)
            (root / "videos").mkdir()
            record = runner._request(prepared, settings, root, 0)
        self.assertTrue(record["success"])
        self.assertEqual(record["response_id"], "x")
