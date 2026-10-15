"""The optional SWE image prefetch tool must not modify shared Docker resources."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import prefetch_aisbench_swe_images as prefetch


class PrefetchSweImagesTests(unittest.TestCase):
    IMAGE = "swebench/sweb.eval.x86_64.django_1776_django-11880:latest"

    def test_image_list_is_validated_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.txt"
            path.write_text("# dataset\n" + self.IMAGE + "\n" + self.IMAGE + "\n")
            self.assertEqual(prefetch.read_images(path), [self.IMAGE])
            path.write_text(self.IMAGE + "\n; docker system prune -af\n")
            with self.assertRaisesRegex(ValueError, "Invalid SWE-bench image"):
                prefetch.read_images(path)

    def test_only_zero_byte_digest_cache_files_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = root / ("sha256:" + "a" * 64)
            valid = root / ("sha256:" + "b" * 64)
            unrelated = root / "other-empty-file"
            incomplete.write_bytes(b"")
            valid.write_bytes(b"valid")
            unrelated.write_bytes(b"")
            self.assertEqual(prefetch.remove_incomplete_cache_entries(root),
                             [incomplete.name])
            self.assertFalse(incomplete.exists())
            self.assertEqual(valid.read_bytes(), b"valid")
            self.assertTrue(unrelated.exists())

    def test_cached_image_never_pulls_or_removes_anything(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            prefetch, "run", return_value=True
        ) as run:
            prefetch.prefetch([self.IMAGE], crane=Path("/usr/bin/crane"),
                              mirror="docker.1ms.run", root=Path(tmp) / "prefetch",
                              docker_root=Path(tmp), min_free_gb=0, timeout=1, retries=1)
        run.assert_called_once_with(["docker", "image", "inspect", self.IMAGE], 30)

    def test_one_uncached_image_is_loaded_and_only_tagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crane = root / "crane"
            crane.write_bytes(b"binary")
            crane.chmod(0o700)
            present = set()
            calls = []
            mirror = "docker.1ms.run/" + self.IMAGE

            def command(argv, timeout, log_path=None):
                calls.append(argv)
                if argv[:3] == ["docker", "image", "inspect"]:
                    return argv[3] in present
                if argv[0] == str(crane):
                    Path(argv[-1]).write_bytes(b"archive")
                    return True
                if argv[:2] == ["docker", "load"]:
                    present.add(mirror)
                    return True
                if argv[:2] == ["docker", "tag"]:
                    present.add(self.IMAGE)
                    return True
                raise AssertionError("Unexpected shared daemon operation: {!r}".format(argv))

            with mock.patch.object(prefetch, "run", side_effect=command):
                prefetch.prefetch([self.IMAGE], crane=crane, mirror="docker.1ms.run",
                                  root=root / "prefetch", docker_root=root,
                                  min_free_gb=0, timeout=30, retries=1)
            self.assertIn(self.IMAGE, present)
            self.assertFalse(list((root / "prefetch/staging").glob("**/*.tar")))
            self.assertEqual([argv[:2] for argv in calls if argv[:2] == ["docker", "tag"]],
                             [["docker", "tag"]])
            self.assertFalse(any("prune" in " ".join(argv) or "rm" in argv for argv in calls))

    def test_low_disk_space_aborts_before_downloading(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            prefetch, "run", return_value=False
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "refusing to fill shared Docker storage"):
                prefetch.prefetch([self.IMAGE], crane=Path("/usr/bin/crane"),
                                  mirror="docker.1ms.run", root=Path(tmp) / "prefetch",
                                  docker_root=Path(tmp), min_free_gb=10 ** 9,
                                  timeout=1, retries=1)
            self.assertEqual(run.call_count, 2)  # Only cache probes, no pull.


if __name__ == "__main__":
    unittest.main()
