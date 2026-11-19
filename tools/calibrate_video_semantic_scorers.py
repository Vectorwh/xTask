#!/usr/bin/env python3
# Copyright 2026 The xTask Authors.
"""Offline operational calibration of object/audio scorers, not JoyAI acceptance.

Run in semantic_env with explicit local models and datasets. Threshold selection
uses calibration rows only; held-out recording/image groups are never tuned on.
No NPU, network, implicit downloads, model substitutions or benchmark mutations.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtask.benchmarks.vbench.text_video.artifacts import sha256, write_json

EVENTS = {
    "rain": {"text": "The sound of rain falling on the ground.", "alternatives": ["The sound of music playing.", "The sound of a person speaking.", "Silence."]},
    "crackling_fire": {"text": "The sound of a crackling fire.", "alternatives": ["The sound of a person speaking.", "The sound of music.", "Silence."]},
    "dog": {"text": "The sound of a dog barking.", "alternatives": ["The sound of a person speaking.", "The sound of music.", "Silence."]},
    "sea_waves": {"text": "The sound of ocean waves breaking on a beach.", "alternatives": ["The sound of music playing.", "The sound of a person speaking.", "Silence."]},
    "helicopter": {"text": "The sound of a helicopter rotor running.", "alternatives": ["The sound of music playing.", "The sound of a person speaking.", "Silence."]},
    "chainsaw": {"text": "The sound of a chainsaw cutting wood.", "alternatives": ["The sound of music playing.", "The sound of a person speaking.", "Silence."]},
    "music": {"text": "The sound of music playing.", "alternatives": ["The sound of rain falling.", "The sound of a dog barking.", "The sound of a person speaking.", "Silence."]},
    "speech": {"text": "The sound of a person speaking.", "alternatives": ["The sound of music playing.", "The sound of rain falling.", "The sound of a dog barking.", "Silence."]},
}


def validate_samples(rows: list[dict[str, Any]], root: Path, group_key: str, require_unique_groups: bool = False) -> None:
    if not rows:
        raise ValueError("Calibration dataset must not be empty")
    groups: dict[str, str] = {}
    identities: set[str] = set()
    for row in rows:
        if row["id"] in identities or row["split"] not in {"calibration", "validation"}:
            raise ValueError("Duplicate ID or invalid calibration split")
        identities.add(row["id"])
        group = str(row[group_key])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Calibration/validation leakage through shared source")
        if require_unique_groups and group in groups:
            raise ValueError("Repeated source recordings cannot increase independent calibration sample counts")
        groups[group] = row["split"]
        path = (root / row["path"]).resolve()
        if root.resolve() not in path.parents or sha256(path) != row["sha256"]:
            raise ValueError("Calibration input escaped root or checksum changed")
    if set(groups.values()) != {"calibration", "validation"}:
        raise ValueError("Separate calibration and validation source groups required")


def confusion(rows: list[dict[str, Any]], predicted: list[bool]) -> dict[str, Any]:
    if len(rows) != len(predicted) or not rows:
        raise ValueError("Missing calibration predictions")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for row, result in zip(rows, predicted):
        if type(row["present"]) is not bool or type(result) is not bool:
            raise ValueError("Presence observations must be Boolean")
        counts[("t" if result == row["present"] else "f") + ("p" if result else "n")] += 1
    positive = counts["tp"] + counts["fn"]
    negative = counts["tn"] + counts["fp"]
    sensitivity = counts["tp"] / positive if positive else None
    specificity = counts["tn"] / negative if negative else None
    return {**counts, "samples": len(rows), "positive_samples": positive, "negative_samples": negative,
            "recall": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2 if positive and negative else None}


def predict(row: dict[str, Any], setting: dict[str, float], modality: str) -> bool:
    if modality == "vision":
        value = row["scores_by_text_threshold"][str(setting["text_threshold"])]
        return value >= setting["box_threshold"]
    return row["similarity"] >= setting["similarity_min"] and row["margin"] >= setting["margin"]


def select_threshold(rows: list[dict[str, Any]], modality: str) -> dict[str, Any]:
    training = [r for r in rows if r["split"] == "calibration"]
    validation = [r for r in rows if r["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Both dataset splits must have observations")
    grid = ([{"box_threshold": b, "text_threshold": t} for b, t in itertools.product((.2, .25, .3, .35, .4, .45, .5), (.2, .3))]
            if modality == "vision" else
            [{"similarity_min": s, "margin": m} for s, m in itertools.product((.10, .15, .20, .25, .30, .35), (.0, .02, .05, .08, .10))])
    trials = []
    for setting in grid:
        metric = confusion(training, [predict(r, setting, modality) for r in training])
        trials.append({"setting": setting, "calibration": metric})
    eligible = [r for r in trials if r["calibration"]["specificity"] >= .9]
    candidates = eligible or trials
    best = max(candidates, key=lambda r: (r["calibration"]["balanced_accuracy"], r["calibration"]["specificity"]))
    setting = best["setting"]
    metrics = confusion(validation, [predict(r, setting, modality) for r in validation])
    per_label = {}
    for label in sorted({r["label"] for r in validation}):
        subset = [r for r in validation if r["label"] == label]
        per_label[label] = confusion(subset, [predict(r, setting, modality) for r in subset])
    usable = [label for label, item in per_label.items()
              if item["positive_samples"] >= 4 and item["negative_samples"] >= 4
              and item["recall"] >= .7 and item["specificity"] >= .9]
    return {"thresholds": setting, "calibration": best["calibration"], "validation": metrics,
            "per_label_validation": per_label, "threshold_grid": trials,
            "operationally_supported_labels": usable,
            "operational_readiness": bool(eligible and metrics["recall"] >= .7 and metrics["specificity"] >= .9
                                          and len(usable) == len(per_label)),
            "readiness_policy": "Provisional engineering check: each held-out label >=4 positive/negative samples, recall>=0.7, specificity>=0.9; not benchmark certification",
            "formal_acceptance_baseline": False}


def vision_observations(root: Path, output: Path) -> list[dict[str, Any]]:
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    rows = json.loads((root / "calibration/coco/queries.json").read_text())
    validate_samples(rows, root, "image_id")
    path = root / "grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True, trust_remote_code=False).eval()
    result = []
    for i, row in enumerate(rows):
        with Image.open(root / row["path"]) as im:
            image = im.convert("RGB")
        if max(image.size) > 768:
            image.thumbnail((768, 768))
        inputs = processor(images=image, text=row["label"] + ".", return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**inputs)
        values = {}
        for threshold in (.2, .3):
            detections = processor.post_process_grounded_object_detection(outputs, input_ids=inputs["input_ids"], threshold=0., text_threshold=threshold, target_sizes=[image.size[::-1]])[0]
            scores = detections["scores"].tolist()
            if any(not math.isfinite(v) for v in scores):
                raise ValueError("Detector returned non-finite confidences")
            values[str(threshold)] = max(scores, default=0.)
        result.append({**row, "scores_by_text_threshold": values})
        write_json(output / "vision_observations.partial.json", result)
        print(f"vision {i + 1}/{len(rows)}", flush=True)
    return result


def audio_observations(root: Path, output: Path, manifest: str = "calibration/audio/samples_v2.json") -> list[dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from transformers import ClapModel, ClapProcessor
    manifest_path = (root / manifest).resolve()
    if root.resolve() not in manifest_path.parents:
        raise ValueError("Audio manifest must reside within the explicit resource root")
    samples = json.loads(manifest_path.read_text())
    validate_samples(samples, root, "source_group", require_unique_groups=True)
    model = ClapModel.from_pretrained(root / "clap-htsat-unfused", local_files_only=True).eval()
    processor = ClapProcessor.from_pretrained(root / "clap-htsat-unfused", local_files_only=True)
    rate = processor.feature_extractor.sampling_rate
    texts = {}
    for label, event in EVENTS.items():
        tokens = processor(text=[event["text"]] + event["alternatives"], padding=True, return_tensors="pt")
        with torch.inference_mode():
            text = model.get_text_features(**tokens)
        texts[label] = torch.nn.functional.normalize(text.float(), dim=-1)
    result = []
    for i, sample in enumerate(samples):
        audio, source_rate = sf.read(root / sample["path"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if source_rate != rate:
            audio = soxr.resample(audio, source_rate, rate)
        # One fixed 5s initial window per independent recording; additional
        # windows are not falsely counted as independent validation samples.
        audio = audio[:5 * rate]
        audio = np.pad(audio, (0, max(0, 5 * rate - len(audio))))
        inputs = processor(audio=audio, sampling_rate=rate, padding="pad", truncation="rand_trunc", return_tensors="pt")
        with torch.inference_mode():
            embedding = model.get_audio_features(**inputs)
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        for label, text in texts.items():
            similarity = (embedding @ text.T).detach().cpu().numpy()[0]
            if not np.isfinite(similarity).all():
                raise ValueError("CLAP returned non-finite similarities")
            result.append({**sample, "id": sample["id"] + "_" + label, "label": label,
                           "present": sample["event"] == label, "similarity": float(similarity[0]),
                           "margin": float(similarity[0] - max(similarity[1:])), "descriptions": EVENTS[label]})
        write_json(output / "audio_observations.partial.json", result)
        print(f"audio {i + 1}/{len(samples)}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modality", choices=("vision", "audio", "both"), default="both")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--audio-manifest", default="calibration/audio/samples_v2.json", help="Relative unique-source audio manifest within root")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if any((output / name).exists() for name in ("calibration_result.json", "vision_observations.json", "audio_observations.json")):
        raise FileExistsError("Refusing to overwrite a completed calibration archive; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    import importlib.util
    import importlib.metadata
    if importlib.util.find_spec("torch_npu") is not None or importlib.metadata.version("transformers") != "4.57.1":
        raise RuntimeError("Use isolated semantic_env without torch_npu and with Transformers 4.57.1")
    import torch
    torch.set_num_threads(args.threads)
    summary = {"schema_version": 1, "purpose": "operational_scorer_calibration", "formal_acceptance_baseline": False,
               "torch": torch.__version__, "script_sha256": sha256(Path(__file__)), "modalities": {}}
    for modality in (("vision", "audio") if args.modality == "both" else (args.modality,)):
        observe = vision_observations if modality == "vision" else audio_observations
        rows = observe(root, output) if modality == "vision" else observe(root, output, args.audio_manifest)
        write_json(output / (modality + "_observations.json"), rows)
        summary["modalities"][modality] = select_threshold(rows, modality)
        write_json(output / "calibration_result.json", summary)
    print(json.dumps({k: {name: value for name, value in v.items() if name != "threshold_grid"} for k, v in summary["modalities"].items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
