#!/usr/bin/env python3
"""Prefetch SWE-bench instance images via a user-scoped registry proxy.

Run on the x86 Docker host, before starting the xLLM/NPU benchmark. This tool
never changes the daemon configuration or deletes any existing Docker images.
Supply a verified crane binary and an image list in AISBench dataset order.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

IMAGE_NAME = re.compile(r"swebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+:latest\Z")
MIRROR_NAME = re.compile(r"[a-zA-Z0-9.-]+(?::[0-9]+)?\Z")
LAYER_CACHE_NAME = re.compile(r"sha256:[0-9a-f]{64}\Z")


def read_images(path: Path):
    images = []
    seen = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        image = raw.strip()
        if not image or image.startswith("#"):
            continue
        if not IMAGE_NAME.fullmatch(image):
            raise ValueError("Invalid SWE-bench image reference: {!r}".format(image))
        if image not in seen:
            images.append(image)
            seen.add(image)
    if not images:
        raise ValueError("The SWE-bench image list is empty")
    return images


def run(command, timeout, log_path=None):
    if log_path is None:
        return subprocess.run(command, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout).returncode == 0
    with log_path.open("w", encoding="utf-8") as stream:
        os.chmod(log_path, 0o600)
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=timeout)
        except subprocess.TimeoutExpired:
            stream.write("\nImage operation exceeded {} seconds.\n".format(timeout))
            return False
    return result.returncode == 0


def remove_incomplete_cache_entries(cache: Path):
    """Discard only zero-byte digest files left by an interrupted crane pull."""
    removed = []
    for item in cache.iterdir():
        if (LAYER_CACHE_NAME.fullmatch(item.name) and item.is_file()
                and not item.is_symlink() and item.stat().st_size == 0):
            item.unlink()
            removed.append(item.name)
    return removed


def prefetch(images, *, crane: Path, mirror: str, root: Path,
             docker_root: Path, min_free_gb: float, timeout: int, retries: int):
    cache = root / "cache"
    staging = root / "staging"
    logs = root / "logs"
    for directory in (root, cache, staging, logs):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
    remove_incomplete_cache_entries(cache)
    for index, canonical in enumerate(images, 1):
        if run(["docker", "image", "inspect", canonical], 30):
            print("[{}/{}] cached {}".format(index, len(images), canonical), flush=True)
            continue
        mirrored = mirror + "/" + canonical
        if run(["docker", "image", "inspect", mirrored], 30):
            if not run(["docker", "tag", mirrored, canonical], 30):
                raise RuntimeError("Cannot tag cached image: {}".format(canonical))
            print("[{}/{}] tagged {}".format(index, len(images), canonical), flush=True)
            continue
        free_gb = shutil.disk_usage(str(docker_root)).free / (1024 ** 3)
        if free_gb < min_free_gb:
            raise RuntimeError(
                "Only {:.1f} GiB remains; refusing to fill shared Docker storage (minimum: {:.1f} GiB)".format(
                    free_gb, min_free_gb
                )
            )
        name = canonical.rsplit("/", 1)[-1].removesuffix(":latest")
        log = logs / (name + ".log")
        succeeded = False
        for attempt in range(1, retries + 1):
            print("[{}/{}] pulling {} via {} (attempt {}/{}, free {:.1f} GiB)".format(
                index, len(images), canonical, mirror, attempt, retries, free_gb
            ), flush=True)
            with tempfile.TemporaryDirectory(prefix="swe_", dir=str(staging)) as tmp:
                archive = Path(tmp) / "image.tar"
                if not run([str(crane), "pull", "--platform", "linux/amd64", "--cache_path",
                            str(cache), mirrored, str(archive)], timeout, log):
                    removed = remove_incomplete_cache_entries(cache)
                    if removed:
                        print("Removed {} incomplete user-cache layer(s)".format(len(removed)),
                              flush=True)
                    continue
                if not run(["docker", "load", "-i", str(archive)], timeout, log):
                    continue
            if not run(["docker", "image", "inspect", mirrored], 30):
                continue
            if not run(["docker", "tag", mirrored, canonical], 30):
                continue
            if run(["docker", "image", "inspect", canonical], 30):
                succeeded = True
                break
            time.sleep(min(2 * attempt, 10))
        if not succeeded:
            raise RuntimeError("Cannot cache {}; see {}".format(canonical, log))
        print("[{}/{}] ready {}".format(index, len(images), canonical), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-file", required=True, type=Path)
    parser.add_argument("--crane", required=True, type=Path)
    parser.add_argument("--mirror", default="docker.1ms.run")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--docker-root", required=True, type=Path)
    parser.add_argument("--min-free-gb", type=float, default=80.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if not MIRROR_NAME.fullmatch(args.mirror):
        parser.error("--mirror must be a registry hostname, without a scheme or path")
    if not args.crane.is_file() or not os.access(args.crane, os.X_OK):
        parser.error("--crane must be an executable file")
    if args.timeout <= 0 or args.retries <= 0 or args.min_free_gb < 0:
        parser.error("timeout/retries must be positive and min-free-gb non-negative")
    prefetch(read_images(args.images_file), crane=args.crane, mirror=args.mirror,
             root=args.root, docker_root=args.docker_root,
             min_free_gb=args.min_free_gb, timeout=args.timeout, retries=args.retries)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print("SWE-bench image prefetch failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
