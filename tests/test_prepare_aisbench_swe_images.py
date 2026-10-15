"""One-command SWE image preparation uses only a chosen x86 runner."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from tools import prepare_aisbench_swe_images as prepare


class FakeRunner:
    def __init__(self, fail_upload=False):
        self.commands = []
        self.uploads = []
        self.fail_upload = fail_upload

    def run(self, command, timeout=300):
        self.commands.append((command, timeout))
        if command == 'printf %s "$HOME"':
            stdout = "/export/home/example"
        elif command.startswith("docker info"):
            stdout = "/var/lib/docker"
        elif command.startswith("flock "):
            stdout = "[1/2] cached instance\n[2/2] cached instance\n"
        else:
            stdout = ""
        return SimpleNamespace(success=True, exit_code=0, stdout=stdout, stderr="")

    def upload(self, source, destination):
        self.uploads.append((destination, Path(source).read_text()))
        return not self.fail_upload


class PrepareSweImagesTests(unittest.TestCase):
    IMAGES = [
        "swebench/sweb.eval.x86_64.django_1776_django-11790:latest",
        "swebench/sweb.eval.x86_64.django_1776_django-11815:latest",
    ]

    def task(self):
        return {
            "case_name": "swebench_test",
            "metrics": "swebench",
            "benchmark": {
                "benchmark_type": "aisbench",
                "swebench": {"runner": {
                    "host": "11.87.48.253",
                    "runtime_container": {"proxy": {
                        "https": "http://example-proxy:80",
                        "no_proxy": "localhost,127.0.0.1",
                    }},
                }},
            },
        }

    def test_single_entrypoint_uploads_list_and_runs_in_order_on_x86_only(self):
        runner = FakeRunner()
        images_path, log = prepare.prepare_images(self.task(), self.IMAGES, runner)
        self.assertTrue(images_path.startswith("/export/home/example/xtask_swe/swe_image_prefetch/runs/"))
        self.assertTrue(images_path.endswith("/images.txt"))
        self.assertEqual(len(runner.uploads), 4)
        self.assertEqual(runner.uploads[0][1], "\n".join(self.IMAGES) + "\n")
        self.assertEqual(len(log.splitlines()), 2)
        commands = [entry[0] for entry in runner.commands]
        action = commands[-1]
        self.assertTrue(action.startswith("flock -n "))
        self.assertLess(action.index("prefill_aisbench_swe_layers.py"),
                        action.index("prefetch_aisbench_swe_images.py"))
        self.assertIn("resume_aisbench_swe_blob.py", action)
        self.assertIn("--images-file", action)
        self.assertIn("docker.1ms.run", action)
        self.assertIn("--docker-root /var/lib/docker", action)
        self.assertNotIn("system prune", action)
        self.assertNotIn("docker run", action)

    def test_failure_stops_before_image_operations(self):
        runner = FakeRunner(fail_upload=True)
        with self.assertRaisesRegex(RuntimeError, "Failed to upload"):
            prepare.prepare_images(self.task(), self.IMAGES, runner)
        self.assertFalse(any(command.startswith("flock ") for command, _ in runner.commands))

    def test_non_swe_and_invalid_proxy_fail_closed(self):
        runner = FakeRunner()
        task = self.task()
        task["metrics"] = "perf"
        with self.assertRaisesRegex(ValueError, "explicit AISBench SWE-bench"):
            prepare.prepare_images(task, self.IMAGES, runner)
        self.assertEqual(runner.commands, [])
        task = self.task()
        task["benchmark"]["swebench"]["runner"]["runtime_container"]["proxy"]["https"] = "malformed"
        with self.assertRaisesRegex(ValueError, "HTTP\\(S\\) URL"):
            prepare.prepare_images(task, self.IMAGES, runner)
        self.assertFalse(any(command.startswith("flock ") for command, _ in runner.commands))

    def test_case_yaml_is_passed_to_generator_without_handwritten_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.yaml"
            path.write_text(yaml.safe_dump(self.task()), encoding="utf-8")
            runner = FakeRunner()
            with mock.patch.object(prepare, "image_list_for_case", return_value=self.IMAGES) as generate, mock.patch.object(
                prepare, "create_host_executor", return_value=runner,
            ) as create:
                self.assertEqual(prepare.main(["--case-yaml", str(path)]), 0)
            generate.assert_called_once_with(path, None)
            create.assert_called_once_with("11.87.48.253", user=None, port=22)
            self.assertEqual(runner.uploads[0][1].splitlines(), self.IMAGES)


if __name__ == "__main__":
    unittest.main()
