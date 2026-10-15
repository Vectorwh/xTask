#!/usr/bin/env python3
"""Generate x86 SWE-bench instance images for one xTask YAML.

Run on the control host before copying the list to the x86 Docker host. This
only reads a local parquet snapshot; it never pulls images or starts xLLM.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import yaml

from tools.prefetch_aisbench_swe_images import IMAGE_NAME
from xtask.benchmarks.aisbench.swebench import (
    is_aisbench_swebench_task,
    swebench_config,
)
from xtask.benchmarks.aisbench.swebench_x86_setup import (
    default_swebench_dataset_path,
)


def _dataset_path(task: Mapping, override: Path | None) -> Path:
    raw = str(override) if override is not None else str(
        swebench_config(task).get("dataset_path") or ""
    ).strip()
    if raw:
        path = Path(os.path.expandvars(raw)).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"SWE-bench dataset is not on this host: {path}; "
                "specify a local snapshot with --dataset-path"
            )
        return path
    default = default_swebench_dataset_path(task)
    if default is None:
        raise FileNotFoundError(
            "No local SWE-bench dataset snapshot found; "
            "specify --dataset-path (online datasets cannot be prelisted)"
        )
    return default


def _parquet_files(root: Path, split: str) -> list[Path]:
    if root.is_file():
        files = [root] if root.suffix == ".parquet" else []
    else:
        files = sorted((root / "data").glob(f"{split}-*.parquet"))
        if not files:
            files = sorted(root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split!r} parquet shards found under {root}")
    return files


def image_list_for_case(case_yaml: Path, dataset_path: Path | None = None) -> list[str]:
    task = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    if not is_aisbench_swebench_task(task):
        raise ValueError("YAML must be an explicit AISBench SWE-bench case")
    config = swebench_config(task)
    split = str(config.get("split") or "test")
    if not re.fullmatch(r"[A-Za-z0-9_]+", split):
        raise ValueError(f"Invalid SWE-bench dataset split: {split!r}")
    spec = str(config.get("filter_spec") or "")
    try:
        filter_instance = re.compile(spec).match  # Same filtering semantics as AISBench.
    except re.error as exc:
        raise ValueError(f"Invalid SWE-bench filter_spec: {spec!r}") from exc
    root = _dataset_path(task, dataset_path)

    import pyarrow.parquet as pq

    instances = []
    for file in _parquet_files(root, split):
        instances.extend(pq.read_table(file, columns=["instance_id"])
                         .column("instance_id").to_pylist())
    if config.get("shuffle"):
        instances.sort()
        random.Random(42).shuffle(instances)

    images = []
    seen = set()
    for instance_id in instances:
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"Invalid SWE-bench instance_id: {instance_id!r}")
        if not filter_instance(instance_id):
            continue
        image = (
            "swebench/sweb.eval.x86_64."
            + instance_id.replace("__", "_1776_").lower()
            + ":latest"
        )
        if not IMAGE_NAME.fullmatch(image):
            raise ValueError(f"Unsupported SWE-bench x86 image for {instance_id!r}")
        if image not in seen:
            seen.add(image)
            images.append(image)
    if not images:
        raise ValueError("The case filter selected no SWE-bench instance images")
    return images


def write_image_list(output: Path, images: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix=f".{output.name}.",
            dir=output.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write("\n".join(images) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-yaml", required=True, type=Path)
    parser.add_argument("--dataset-path", type=Path,
                        help="Local parquet file or directory, overriding the YAML/default")
    parser.add_argument("--output", required=True, type=Path,
                        help="Explicit destination; no case-specific output path is assumed")
    args = parser.parse_args(argv)
    images = image_list_for_case(args.case_yaml, args.dataset_path)
    write_image_list(args.output, images)
    print(f"Wrote {len(images)} SWE-bench images to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"SWE-bench image list generation failed: {exc}", file=sys.stderr)
        sys.exit(1)
