# Copyright 2026 The xTask Authors.
"""Offline-only GroundingDINO/CLAP scorers, loaded solely in scoring context."""
from __future__ import annotations

import array
import importlib.metadata
import math
import os
import subprocess
import sys
from pathlib import Path

from .artifacts import sha256


def preflight_checkpoints(spec):
    for modality, expected_type in (("vision", "grounding-dino"), ("audio", "clap")):
        if modality not in spec:
            continue
        path = Path(spec[modality]["model_path"])
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"Prepare local {modality} scorer checkpoint first: {path}")
        import json
        configuration = json.loads((path / "config.json").read_text())
        if configuration.get("model_type") != expected_type:
            raise ValueError(f"{modality} model_type must be {expected_type}")
        weights = [p for p in path.rglob("*") if p.is_file() and p.suffix in {".safetensors", ".bin"}]
        if not weights or any(p.stat().st_size == 0 for p in weights):
            raise FileNotFoundError(f"No nonempty {modality} scorer weights in {path}")


def preflight_environment(spec, directory: Path):
    """Check the selected interpreter/dependencies before expensive generation."""
    command = [spec.get("python_exec", sys.executable), "-c",
               "import importlib.util,importlib.metadata,json; "
               "assert importlib.util.find_spec('torch_npu') is None, 'Use an isolated scorer environment without torch_npu'; "
               "assert importlib.metadata.version('transformers') == '4.57.1', 'Prepare Transformers 4.57.1 separately; do not downgrade WAN'; "
               "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, ClapModel, ClapProcessor; "
               "import torch,cv2; print(json.dumps({'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'scoring_only':True}))"]
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONPATH="")
    result = subprocess.run(command, capture_output=True, text=True, timeout=min(spec["timeout"], 120), env=environment)
    (directory / "negative_environment_stdout.log").write_text(result.stdout)
    (directory / "negative_environment_stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError("Semantic scorer environment invalid; prepare negative_control.python_exec; see negative_environment_stderr.log")


def checkpoint_provenance(path: Path):
    files = [p for p in path.rglob("*") if p.is_file()
             and p.suffix in {".json", ".safetensors", ".bin", ".txt"}]
    return {"model_path": str(path), "files_sha256": {str(p.relative_to(path)): sha256(p) for p in sorted(files)},
            "transformers_version": importlib.metadata.version("transformers"),
            "torch_version": importlib.metadata.version("torch"),
            "scorer_code_sha256": sha256(Path(__file__)),
            "policy_code_sha256": sha256(Path(__file__).with_name("negative.py")), "network_access": "disabled/local_files_only"}


def sampled_frames(path: Path, count, max_side):
    import cv2
    from PIL import Image
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video for semantic scoring: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise ValueError("Video has no frame metadata")
    indices = sorted({min(total - 1, int((i + .5) * total / min(total, count))) for i in range(min(total, count))})
    images = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Unable to decode sampled frame {index}")
            height, width = frame.shape[:2]
            if max(height, width) > max_side:
                scale = max_side / max(height, width)
                frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return images, indices


class GroundingDinoScorer:
    def __init__(self, spec, device):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.spec = spec
        path = Path(spec["model_path"])
        self.provenance = {**checkpoint_provenance(path), "method": "grounding_dino_object_presence", "device": device,
                           "box_threshold": spec["box_threshold"], "text_threshold": spec["text_threshold"]}
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(str(path), local_files_only=True, trust_remote_code=False).to(device).eval()
        if getattr(self.model.config, "model_type", None) != "grounding-dino":
            raise ValueError("Vision checkpoint must be GroundingDINO; model-type substitutions change scoring")
        self.device = device
        self.torch = torch

    def score(self, path, required, forbidden, media):
        images, indices = sampled_frames(path, self.spec["frames"], self.spec["max_side"])
        masks, evidence = {}, {}
        for label in required + forbidden:
            # Query labels separately; accepting any box from a multi-class
            # query would conflate required and forbidden classes.
            mask, frames = [], []
            for image in images:
                inputs = self.processor(images=image, text=label.strip() + ".", return_tensors="pt").to(self.device)
                with self.torch.inference_mode():
                    outputs = self.model(**inputs)
                results = self.processor.post_process_grounded_object_detection(
                    outputs, input_ids=inputs["input_ids"], threshold=self.spec["box_threshold"],
                    text_threshold=self.spec["text_threshold"], target_sizes=[image.size[::-1]],
                )[0]
                scores = [float(value) for value in results["scores"].detach().cpu().tolist()]
                if any(not math.isfinite(value) for value in scores):
                    raise ValueError("Detector returned non-finite confidence")
                mask.append(bool(scores))
                frames.append({"scores": scores, "boxes": results["boxes"].detach().cpu().tolist(),
                               "labels": results.get("text_labels", results.get("labels", []))})
            masks[label], evidence[label] = mask, frames
        return {"required": {label: masks[label] for label in required},
                "forbidden": {label: masks[label] for label in forbidden},
                "frame_indices": indices, "detections": evidence,
                "interpretation": "Sampled-frame object detector output; not calibrated probability or action verification"}


def read_pcm(path, sample_rate, media, timeout):
    result = subprocess.run(
        [media.get("ffmpeg", "ffmpeg"), "-nostdin", "-v", "error", "-xerror", "-i", str(path),
         "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
        check=True, capture_output=True, timeout=timeout,
    )
    samples = array.array("f")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples or any(not math.isfinite(value) for value in samples):
        raise ValueError("Missing/non-finite decoded audio samples")
    return samples


def window_starts(length, window, stride):
    if length <= 0 or window <= 0 or stride <= 0 or stride > window:
        raise ValueError("Invalid audio sampling window")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


class ClapScorer:
    def __init__(self, spec, device, timeout):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.spec, self.timeout, self.device, self.torch = spec, timeout, device, torch
        path = Path(spec["model_path"])
        self.provenance = {**checkpoint_provenance(path), "method": "clap_annotated_contrastive_audio_presence", "device": device}
        self.processor = ClapProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = ClapModel.from_pretrained(str(path), local_files_only=True).to(device).eval()
        if getattr(self.model.config, "model_type", None) != "clap":
            raise ValueError("Audio scorer checkpoint must be CLAP")
        extractor = self.processor.feature_extractor
        self.sample_rate = extractor.sampling_rate
        self.audio_truncation = "fusion" if self.model.config.audio_config.enable_fusion else "rand_trunc"
        if round(spec["window_seconds"] * self.sample_rate) > extractor.nb_max_samples:
            raise ValueError("Audio scoring window exceeds CLAP checkpoint maximum; random crops are forbidden")
        self.provenance["scoring_sample_rate"] = self.sample_rate
        self.provenance["feature_truncation"] = self.audio_truncation
        self.provenance["feature_padding"] = "pad"

    def score(self, path, required, forbidden, media):
        import numpy as np
        torch = self.torch
        samples = np.asarray(read_pcm(path, self.sample_rate, media, self.timeout), dtype=np.float32)
        window = round(self.spec["window_seconds"] * self.sample_rate)
        starts = window_starts(len(samples), window, round(self.spec["stride_seconds"] * self.sample_rate))
        # Build deterministic windows; no random CLAP crop over a long input.
        audio_features = []
        for start in starts:
            values = samples[start:start + window]
            if len(values) < window:
                values = np.pad(values, (0, window - len(values)))
            inputs = self.processor(audio=values, sampling_rate=self.sample_rate, padding="pad", truncation=self.audio_truncation, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                feature = self.model.get_audio_features(**inputs)
            audio_features.append(torch.nn.functional.normalize(feature.float(), dim=-1))
        features = torch.cat(audio_features)
        masks, evidence = {}, {}
        for event in required + forbidden:
            descriptions = [event["text"]] + event["alternatives"]
            tokens = self.processor(text=descriptions, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                text = self.model.get_text_features(**tokens)
            similarity = (features @ torch.nn.functional.normalize(text.float(), dim=-1).T).detach().cpu().numpy()
            if not np.isfinite(similarity).all():
                raise ValueError("CLAP returned NaN/Inf")
            gap = similarity[:, 0] - similarity[:, 1:].max(axis=1)
            masks[event["id"]] = ((similarity[:, 0] >= self.spec["similarity_min"]) & (gap >= self.spec["margin"])).tolist()
            evidence[event["id"]] = {"descriptions": descriptions, "cosine_similarities": similarity.tolist(),
                                     "target_minus_best_alternative": gap.tolist()}
        return {"required": {event["id"]: masks[event["id"]] for event in required},
                "forbidden": {event["id"]: masks[event["id"]] for event in forbidden},
                "window_start_s": [start / self.sample_rate for start in starts], "similarity": evidence,
                "interpretation": "CLAP window-level contrastive surrogate; not calibrated sound-event probability, ASR, or audio/video synchronization"}


def load_scorers(spec):
    # This module is never imported during the model-generation/perf phase.
    preflight_checkpoints(spec)
    installed = importlib.metadata.version("transformers")
    if installed != "4.57.1":
        raise RuntimeError(f"xtask_negative_v1 requires isolated Transformers 4.57.1, found {installed}; use negative_control.python_exec, do not downgrade WAN/VBench environments")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    scorers = {}
    if "vision" in spec:
        scorers["vision"] = GroundingDinoScorer(spec["vision"], spec["device"])
    if "audio" in spec:
        scorers["audio"] = ClapScorer(spec["audio"], spec["device"], spec["timeout"])
    return scorers
