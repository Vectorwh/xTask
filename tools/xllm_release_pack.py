#!/usr/bin/env python3
"""Validate/stage new xLLM release packs; legacy installation stays separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import uuid
import shutil
import subprocess
import sys
import tarfile
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_file():
        fields = sidecar.read_text().split()
        if not fields or fields[0] != sha256(path):
            raise ValueError("Checksum mismatch: %s" % path.name)


def extract(archive, target):
    """Reject traversal, special files and links outside the staging directory."""
    target = Path(target).resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe archive path: %s" % member.name)
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError("Unsupported archive member: %s" % member.name)
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                base = target / name.parent if member.issym() else target
                resolved = (base / str(link)).resolve()
                if link.is_absolute() or (resolved != target and target not in resolved.parents):
                    raise ValueError("Unsafe archive link: %s" % member.name)
        # Validate against earlier symlinks as each member is extracted.
        for member in members:
            destination = (target / member.name).resolve()
            if destination != target and target not in destination.parents:
                raise ValueError("Archive member escapes staging: %s" % member.name)
            bundle.extract(member, str(target))


def _one(root, pattern):
    matches = list(root.glob(pattern))
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError("Expected exactly one %s in release pack" % pattern)
    return matches[0]


def prepare(archive, destination, validate_environment=True):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    verify_sidecar(archive)
    if destination.exists():
        raise ValueError("Release staging destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".release-pack-", dir=str(destination.parent)))
    try:
        extract(archive, stage)
        root = stage / "xllm_pack"
        if not root.is_dir():
            raise ValueError("Missing xllm_pack directory")
        if list(root.glob("xllm-lib-*.tar.gz")):
            raise ValueError("Legacy pack: use the existing installation path")
        wheel = _one(root, "xllm*.whl")
        vendors = _one(root, "vendors-*.tar.gz")
        runtime = _one(root, "mooncake-runtime-v*.tar.gz")
        for path in root.glob("*.tar.gz"):
            verify_sidecar(path)
        extract(vendors, root)
        extract(runtime, root)
        vendor_dir = root / "vendors"
        if not (vendor_dir / "config.ini").is_file() or any(
            not (vendor_dir / name).is_dir() for name in ("custom_xllm_math", "custom_transformer")
        ):
            raise ValueError("Incomplete vendors archive")
        mooncake = _one(root, "mooncake-v*/MANIFEST.txt").parent
        manifest = json.loads((mooncake / "MANIFEST.txt").read_text())
        if validate_environment and manifest.get("architecture") != platform.machine():
            raise ValueError("Mooncake architecture does not match this host")
        if manifest.get("source_wheel_sha256") != sha256(wheel):
            raise ValueError("Mooncake runtime does not match the xLLM wheel")
        for name in ("mooncake_master", "mooncake_client"):
            if not os.access(mooncake / "bin" / name, os.X_OK):
                raise ValueError("Missing executable Mooncake runtime binary")
        if not (mooncake / "lib").is_dir() or not list((mooncake / "lib").glob("*.so*")):
            raise ValueError("Missing Mooncake runtime libraries")
        checksums = mooncake / "SHA256SUMS"
        if not checksums.is_file():
            raise ValueError("Missing Mooncake SHA256SUMS")
        checksum_lines = checksums.read_text().splitlines()
        if not checksum_lines:
            raise ValueError("Empty Mooncake SHA256SUMS")
        for line in checksum_lines:
            digest, name = line.split(maxsplit=1)
            path = (mooncake / name.lstrip("*")).resolve()
            if mooncake not in path.parents or sha256(path) != digest:
                raise ValueError("Invalid Mooncake member checksum")
        # Let pip perform the final ABI check; reject clear Python/CPU/Torch
        # mismatches before modifying installed packages or vendors.
        filename = wheel.name
        cp = "cp%s%s" % sys.version_info[:2]
        if validate_environment and "-cp" in filename and "-" + cp + "-" not in filename:
            raise ValueError("xLLM wheel Python ABI mismatch")
        if validate_environment and "linux_" + platform.machine() not in filename and not filename.endswith("any.whl"):
            raise ValueError("xLLM wheel CPU architecture mismatch")
        if validate_environment and "torch" in filename:
            match = re.search(r"torch(\d+)_(\d+)_", filename)
            if match:
                version = subprocess.check_output(
                    [sys.executable, "-c", "import torch; print(torch.__version__)"], text=True).strip()
                if version.split(".")[:2] != list(match.groups()):
                    raise ValueError("xLLM wheel Torch version mismatch")
        # Compare the exact CANN version tag emitted by upstream when the
        # receiving image publishes Version=. Never guess from an image name.
        version_file = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")) / "compiler/version.info"
        if validate_environment and version_file.is_file():
            match = re.search(r"^Version=(.+)$", version_file.read_text(), re.M)
            tag = re.search(r"-(cann[A-Za-z0-9.]+)-", vendors.name)
            if match and tag and "cann" + re.sub(r"[^A-Za-z0-9.]", "", match.group(1).strip()) != tag.group(1):
                raise ValueError("Vendors CANN version mismatch")
        third = root / "third_whl"
        third.mkdir(exist_ok=True)
        for path in root.glob("*.whl"):
            if path != wheel:
                shutil.move(str(path), str(third / path.name))
        service = root / "service_binary"
        if (root / "xllm_master_serving").is_file():
            dynamic_info = subprocess.check_output(
                ["readelf", "-d", str(root / "xllm_master_serving")], text=True, stderr=subprocess.STDOUT)
            if "Shared library: [librust_tokenizers.so]" in dynamic_info and not (root / "librust_tokenizers.so").is_file():
                raise ValueError("Dynamic Service tokenizer is missing from new pack")
            service.mkdir(exist_ok=True)
            shutil.copy2(root / "xllm_master_serving", service / "xllm_master_serving")
            if (root / "librust_tokenizers.so").is_file():
                shutil.copy2(root / "librust_tokenizers.so", service / "librust_tokenizers.so")
        provenance = {
            "layout": "release-v1", "archive_sha256": sha256(archive),
            "wheel": wheel.name, "mooncake": mooncake.name,
            "xllm_commit": manifest.get("xllm_commit"),
            "service_version": (root / "xllm-service_version.txt").read_text()
                if (root / "xllm-service_version.txt").is_file() else None,
        }
        (root / "xtask-pack-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
        os.replace(stage, destination)
        return provenance
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def publish(archive, directory, service=None):
    """Archive atomically; optional append always regenerates the outer checksum."""
    archive, directory = Path(archive), Path(directory)
    verify_sidecar(archive)
    directory.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".publish-", dir=str(directory)))
    try:
        output = stage / "xllm_pack.tar.gz"
        if service:
            service = Path(service)
            binary = service / "xllm_master_serving"
            if not binary.is_file():
                raise ValueError("Missing archived Service executable")
            with tarfile.open(archive, "r:gz") as old, tarfile.open(output, "w:gz") as new:
                for member in old:
                    if member.name in ("xllm_pack/xllm_master_serving", "xllm_pack/xllm-service_version.txt", "xllm_pack/librust_tokenizers.so"):
                        continue
                    new.addfile(member, old.extractfile(member) if member.isfile() else None)
                new.add(binary, arcname="xllm_pack/xllm_master_serving")
                new.add(service / "manifest.txt", arcname="xllm_pack/xllm-service_version.txt")
                if (service / "librust_tokenizers.so").is_file():
                    new.add(service / "librust_tokenizers.so", arcname="xllm_pack/librust_tokenizers.so")
        else:
            shutil.copy2(archive, output)
        # Validate the complete package before changing the published pointer.
        checked = stage / "validated"
        # Publishing can run on the host's Python while the wheel targets the
        # container's Python/Torch/CANN; ABI checks belong to installation.
        prepare(output, checked, validate_environment=False)
        shutil.rmtree(checked)
        digest = sha256(output)
        (stage / "xllm_pack.tar.gz.sha256").write_text(digest + "  xllm_pack.tar.gz\n")
        release = directory / digest
        if release.exists():
            verify_sidecar(release / "xllm_pack.tar.gz")
            if sha256(release / "xllm_pack.tar.gz") != digest:
                raise ValueError("Existing content-addressed release is corrupt")
            shutil.rmtree(stage)
        else:
            os.replace(stage, release)
        link = directory / (".current-" + uuid.uuid4().hex)
        link.symlink_to(release.name)
        os.replace(link, directory / "current")
        return str(release)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    stage = sub.add_parser("prepare")
    stage.add_argument("archive")
    stage.add_argument("destination")
    pub = sub.add_parser("publish")
    pub.add_argument("archive")
    pub.add_argument("directory")
    pub.add_argument("--service")
    args = parser.parse_args()
    try:
        result = prepare(args.archive, args.destination) if args.action == "prepare" else publish(args.archive, args.directory, args.service)
        print(json.dumps(result))
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as exc:
        parser.exit(1, "Release pack error: %s\n" % exc)


if __name__ == "__main__":
    main()
