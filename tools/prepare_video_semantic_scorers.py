#!/usr/bin/env python3
# Copyright 2026 The xTask Authors.
"""Explicitly prepare an isolated CPU semantic scorer; never run from benchmarks.

Run inside the same Python/OS runner image used by xTask so the venv interpreter
and compiled wheels remain compatible. The output root must be a dedicated path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

REQUIREMENTS = [
    "torch==2.9.0", "torchvision==0.24.0", "transformers==4.57.1",
    "huggingface-hub==0.36.2", "tokenizers==0.22.1", "safetensors>=0.4.3,<1",
    "numpy==1.26.4", "Pillow>=10,<12", "opencv-python-headless==4.11.0.86",
    "scipy>=1.10,<2", "librosa==0.11.0", "soundfile>=0.12,<1",
    "timm==1.0.12",
]
REPOSITORIES = {"vision": ("IDEA-Research/grounding-dino-tiny", "grounding-dino-tiny"),
                "audio": ("laion/clap-htsat-unfused", "clap-htsat-unfused")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Dedicated output path mounted in the scorer container")
    parser.add_argument("--wheelhouse", help="Prepared local wheels: pip uses --no-index")
    parser.add_argument("--download-models", action="store_true", help="Explicit opt-in to download neural checkpoints")
    parser.add_argument("--model-endpoint", help="Explicit HTTPS Hugging Face API mirror for resource preparation only")
    parser.add_argument("--vision-revision", help="Pinned HF commit SHA; required with --download-models")
    parser.add_argument("--audio-revision", help="Pinned HF commit SHA; required with --download-models")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Prepare with Python >=3.10 inside the xTask scoring image, not controller Python3.9")
    root = Path(args.root).expanduser().resolve()
    if root == Path("/") or root == Path.home():
        parser.error("root must be a dedicated scorer path")
    environment = root / "semantic_env"
    if environment.exists():
        parser.error(f"Refusing to overwrite an existing scoring environment: {environment}")
    if args.download_models:
        import re
        if any(not revision or not re.fullmatch(r"[0-9a-f]{40}", revision)
               for revision in (args.vision_revision, args.audio_revision)):
            parser.error("Download requires pinned 40-character vision/audio commit revisions; no implicit main")
    if args.model_endpoint:
        from urllib.parse import urlsplit
        parsed = urlsplit(args.model_endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            parser.error("model-endpoint must be an HTTPS URL without credentials/query/fragment")
    root.mkdir(parents=True, exist_ok=True)
    (root / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n")
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
    python = environment / "bin/python"
    command = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    if args.wheelhouse:
        command += ["--no-index", "--find-links", str(Path(args.wheelhouse).resolve())]
    command += ["-r", str(root / "requirements.txt")]
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONPATH="")
    subprocess.run(command, check=True, env=env)
    if args.model_endpoint:
        env["HF_ENDPOINT"] = args.model_endpoint.rstrip("/")
    subprocess.run([str(python), "-c", "import importlib.util; assert importlib.util.find_spec('torch_npu') is None; from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, ClapModel, ClapProcessor; import torch, cv2; print('CPU scorer environment ready')"], check=True, env=env)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True, env=env)
    (root / "requirements.lock.txt").write_text(freeze)
    preparation = {"python": sys.version, "python_exec": str(python), "pip_freeze": freeze,
                   "models": {}, "note": "Dedicated setup only; never mutates WAN/VBench engine environments"}
    if args.download_models:
        for modality, (repository, dirname) in REPOSITORIES.items():
            revision = getattr(args, modality + "_revision")
            target = root / dirname
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite scorer checkpoint: {target}")
            code = "from huggingface_hub import snapshot_download; snapshot_download(repo_id=" + repr(repository) + ", revision=" + repr(revision) + ", local_dir=" + repr(str(target)) + ", allow_patterns=['*.json','*.safetensors','*.bin','*.txt','*.model'])"
            subprocess.run([str(python), "-c", code], check=True, env=env)
            preparation["models"][modality] = {"repo_id": repository, "revision": revision, "path": str(target)}
    (root / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    print("Prepared scorer Python:", python)
    print("Checkpoints must be prepared separately unless --download-models was selected.")


if __name__ == "__main__":
    main()
