"""The optional SWE image list generator is case-scoped and offline."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tools import generate_aisbench_swe_images as generator
from tools.prefetch_aisbench_swe_images import read_images


class GenerateSweImageListTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "data").mkdir(parents=True)
        self.yaml = self.root / "case.yaml"

    def write_case(self, **swe):
        config = {
            "case_name": "swe_image_list_test",
            "metrics": "swebench",
            "benchmark": {"benchmark_type": "aisbench", "swebench": {
                "dataset": "verified_mini", "split": "test", **swe,
            }},
        }
        self.yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config

    def write_shard(self, split, number, instance_ids):
        pq.write_table(pa.table({"instance_id": instance_ids}),
                       self.dataset / "data" / f"{split}-{number:05d}.parquet")

    def test_filter_and_multiple_shards_match_aisbench_order(self):
        self.write_case(filter_spec=r"^(?:django__django-11790|django__django-11815)$")
        self.write_shard("test", 0, ["django__django-11790", "django__django-11848"])
        self.write_shard("test", 1, ["django__django-11815", "django__django-11790"])
        self.write_shard("train", 0, ["django__django-99999"])
        images = generator.image_list_for_case(self.yaml, self.dataset)
        self.assertEqual(images, [
            "swebench/sweb.eval.x86_64.django_1776_django-11790:latest",
            "swebench/sweb.eval.x86_64.django_1776_django-11815:latest",
        ])
        output = self.root / "chosen" / "images.txt"
        generator.write_image_list(output, images)
        self.assertEqual(read_images(output), images)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_default_dataset_resolution_and_cli(self):
        self.write_case()
        self.write_shard("test", 0, ["sphinx-doc__sphinx-10323"])
        with mock.patch.object(generator, "default_swebench_dataset_path",
                               return_value=self.dataset):
            output = self.root / "generated.txt"
            self.assertEqual(generator.main([
                "--case-yaml", str(self.yaml), "--output", str(output),
            ]), 0)
        self.assertEqual(output.read_text(),
                         "swebench/sweb.eval.x86_64.sphinx-doc_1776_sphinx-10323:latest\n")

    def test_explicit_remote_path_requires_local_override(self):
        self.write_case(dataset_path=str(self.root / "not_on_this_host"))
        self.write_shard("test", 0, ["django__django-11790"])
        with self.assertRaisesRegex(FileNotFoundError, "--dataset-path"):
            generator.image_list_for_case(self.yaml)
        self.assertEqual(len(generator.image_list_for_case(self.yaml, self.dataset)), 1)

    def test_empty_or_invalid_filter_refuses_to_write(self):
        self.write_shard("test", 0, ["django__django-11790"])
        self.write_case(filter_spec=r"^missing$")
        with self.assertRaisesRegex(ValueError, "selected no"):
            generator.image_list_for_case(self.yaml, self.dataset)
        self.write_case(filter_spec="[")
        with self.assertRaisesRegex(ValueError, "Invalid SWE-bench filter_spec"):
            generator.image_list_for_case(self.yaml, self.dataset)

    def test_rejects_non_swe_yaml_and_unsupported_image(self):
        task = self.write_case()
        task["metrics"] = "perf"
        self.yaml.write_text(yaml.safe_dump(task))
        with self.assertRaisesRegex(ValueError, "explicit AISBench SWE-bench"):
            generator.image_list_for_case(self.yaml, self.dataset)
        self.write_case()
        self.write_shard("test", 0, ["bad/id"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            generator.image_list_for_case(self.yaml, self.dataset)

    def test_split_is_honored_and_missing_split_fails(self):
        self.write_case(split="train")
        self.write_shard("test", 0, ["django__django-11790"])
        with self.assertRaisesRegex(FileNotFoundError, "train"):
            generator.image_list_for_case(self.yaml, self.dataset)
        self.write_shard("train", 0, ["django__django-11815"])
        self.assertEqual(generator.image_list_for_case(self.yaml, self.dataset), [
            "swebench/sweb.eval.x86_64.django_1776_django-11815:latest"
        ])


if __name__ == "__main__":
    unittest.main()
