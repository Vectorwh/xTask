# Copyright 2026 The xTask Authors.
"""Atomic, private, portable text-video artifacts and date directories."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_date_dir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    time = datetime.now()
    for offset in range(10000):
        candidate = root / (time + timedelta(seconds=offset)).strftime("%Y%m%d_%H%M%S")
        try:
            candidate.mkdir(mode=0o700)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Unable to allocate a unique date archive")


def redact(value):
    if isinstance(value, dict):
        return {k: ("***" if k.lower() in {"api_key", "authorization", "token", "password"} else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value
