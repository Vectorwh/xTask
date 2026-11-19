# Copyright 2026 The xTask Authors.
"""Paired negative-control evaluation using object/audio semantic evidence.

This is an xTask regression profile, not an official VBench/VABench total or a
universal calibrated acceptance standard. No neural network runs in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from xtask.benchmarks.vbench.text_video.artifacts import write_json

PROFILE = "xtask_negative_v1"
# Explicit engineering defaults, NOT thresholds published by benchmark authors.
DEFAULT_POLICY = {
    "min_conditions": 6, "min_informative_conditions": 3,
    "reference_presence_min": 0.10, "target_forbidden_max": 0.10,
    "relative_reduction_min": 0.25, "improvement_rate_min": 0.60,
    "required_presence_min": 0.50, "positive_drop_max": 0.05,
    "improvement_epsilon": 0.02,
}
VISUAL_KEYS = ("required_video", "forbidden_video")
AUDIO_KEYS = ("required_audio", "forbidden_audio")
METRICS = {f"{modality}_{name}" for modality in ("video", "audio") for name in (
    "reference_forbidden_rate", "target_forbidden_rate", "relative_reduction",
    "improvement_rate", "reference_required_rate", "target_required_rate",
    "positive_drop", "informative_conditions", "conditions", "informative_target_forbidden_rate",
)} | {"passed", "conditions", "scored_pairs"}


def _finite(value, name, low=0, high=1):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return value


def normalize_spec(raw):
    if not isinstance(raw, Mapping) or raw.get("backend") != "benchmark_semantics":
        raise ValueError("negative_control requires backend: benchmark_semantics")
    spec = json.loads(json.dumps(raw, allow_nan=False))
    if set(spec) - {"backend", "profile", "device", "vision", "audio", "policy", "timeout", "python_exec"}:
        raise ValueError("Unknown benchmark_semantics fields")
    if spec.setdefault("profile", PROFILE) != PROFILE:
        raise ValueError("Unknown negative-control policy profile")
    if spec.setdefault("device", "cpu") != "cpu" and not str(spec["device"]).startswith("cuda"):
        raise ValueError("Semantic scorers support CPU/CUDA, not the NPU generator context")
    if "python_exec" in spec and (not isinstance(spec["python_exec"], str) or not spec["python_exec"]):
        raise ValueError("negative_control.python_exec must be a prepared scoring Python executable")
    modalities = []
    for name in ("vision", "audio"):
        if name not in spec:
            continue
        block = spec[name]
        if not isinstance(block, dict) or not isinstance(block.get("model_path"), str) or not block["model_path"]:
            raise ValueError(f"{name}.model_path must point to a prepared local checkpoint")
        common = {"model_path"}
        extra = {"frames", "max_side", "box_threshold", "text_threshold"} if name == "vision" else {"window_seconds", "stride_seconds", "margin", "similarity_min"}
        if set(block) - common - extra:
            raise ValueError(f"Unknown {name} scorer options")
        if name == "vision":
            for key, default in (("frames", 16), ("max_side", 768)):
                value = block.setdefault(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= (128 if key == "frames" else 4096):
                    raise ValueError(f"vision.{key} out of supported range")
            _finite(block.setdefault("box_threshold", .4), "vision.box_threshold")
            _finite(block.setdefault("text_threshold", .3), "vision.text_threshold")
        else:
            for key, default in (("window_seconds", 5.0), ("stride_seconds", 2.5)):
                _finite(block.setdefault(key, default), "audio." + key, .1, 10)
            if block["stride_seconds"] > block["window_seconds"]:
                raise ValueError("Audio windows must not leave unobserved gaps")
            _finite(block.setdefault("margin", .05), "audio.margin")
            _finite(block.setdefault("similarity_min", .15), "audio.similarity_min", -1, 1)
        modalities.append(name)
    if not modalities:
        raise ValueError("Enable at least one semantic scorer: vision or audio")
    policy = spec.setdefault("policy", {})
    if not isinstance(policy, dict) or set(policy) - set(DEFAULT_POLICY):
        raise ValueError("Unknown negative-control acceptance fields")
    for key, default in DEFAULT_POLICY.items():
        value = policy.setdefault(key, default)
        if key.startswith("min_"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"policy.{key} must be a positive integer")
        else:
            _finite(value, "policy." + key)
    if policy["reference_presence_min"] <= 0 or policy["improvement_epsilon"] <= 0:
        raise ValueError("Reference presence and improvement epsilon must be positive; identical videos are not an improvement")
    if policy["min_informative_conditions"] > policy["min_conditions"]:
        raise ValueError("min_informative_conditions must not exceed min_conditions")
    _finite(spec.setdefault("timeout", 3600), "timeout", .1, 86400)
    return spec


def validate_constraints(raw, spec):
    if not isinstance(raw, dict) or set(raw) - set(VISUAL_KEYS + AUDIO_KEYS):
        raise ValueError("Semantic checks require required_video/forbidden_video/required_audio/forbidden_audio lists; free-form sentences are not object detectors")
    if not raw:
        raise ValueError("Every sample needs explicit semantic constraints")
    for modality, keys in (("vision", VISUAL_KEYS), ("audio", AUDIO_KEYS)):
        active = any(key in raw for key in keys)
        if active and modality not in spec:
            raise ValueError(f"Sample {modality} constraints have no enabled scorer")
        if not active:
            continue
        for key in keys:
            entries = raw.get(key)
            if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
                raise ValueError(f"{key} needs 1..16 targets; protect positive targets as well")
            if modality == "vision":
                if (any(not isinstance(label, str) or not label.strip() or "." in label or len(label) > 80 for label in entries)
                        or len(set(entries)) != len(entries)):
                    raise ValueError("Visual constraints must be distinct short object labels, not action/negation sentences")
            else:
                for item in entries:
                    if not isinstance(item, dict) or set(item) != {"id", "text", "alternatives"}:
                        raise ValueError("Audio event needs id, positive text description and alternatives")
                    if any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("id", "text")):
                        raise ValueError("Audio event id/text must not be empty")
                    alternatives = item["alternatives"]
                    if (not isinstance(alternatives, list) or not alternatives
                            or any(not isinstance(t, str) or not t.strip() for t in alternatives)
                            or item["text"] in alternatives or len(set(alternatives)) != len(alternatives)):
                        raise ValueError("CLAP event detection requires distinct explicit contrasting descriptions")
                if len({item["id"] for item in entries}) != len(entries):
                    raise ValueError("Audio event IDs must be unique")
        identifiers = lambda values: {v if isinstance(v, str) else v["id"] for v in values}
        if identifiers(raw[keys[0]]) & identifiers(raw[keys[1]]):
            raise ValueError("A content target cannot be both required and forbidden")
    return raw


def presence_summary(required, forbidden):
    """Required labels use conjunction; forbidden labels use sample-wise union."""
    arrays = list(required.values()) + list(forbidden.values())
    if not arrays or not required or not forbidden or not arrays[0]:
        raise ValueError("Missing semantic per-frame/window evidence")
    n = len(arrays[0])
    if any(len(values) != n or any(type(v) is not bool for v in values) for values in arrays):
        raise ValueError("Semantic evidence masks must be complete equal-length boolean lists")
    required_rate = sum(all(values[i] for values in required.values()) for i in range(n)) / n
    forbidden_rate = sum(any(values[i] for values in forbidden.values()) for i in range(n)) / n
    return {"required_rate": required_rate, "forbidden_rate": forbidden_rate, "samples": n}


def summarize(rows, spec):
    """Average repeated identical conditions before judging independent evidence."""
    units = {}
    for row in rows:
        units.setdefault(row["condition_key"], []).append(row)
    metrics = {"conditions": len(units)}
    violations = []
    policy = spec["policy"]
    for modality in ("video", "audio"):
        relevant = [[row[modality] for row in group if modality in row] for group in units.values()]
        relevant = [group for group in relevant if group]
        if not relevant:
            continue
        # A repeated seed is a repeated observation, not independent benchmark evidence.
        aggregates = [{arm: {name: sum(r[arm][name] for r in group) / len(group)
                              for name in ("required_rate", "forbidden_rate")}
                       for arm in ("reference", "target")} for group in relevant]
        ref = [row["reference"]["forbidden_rate"] for row in aggregates]
        target = [row["target"]["forbidden_rate"] for row in aggregates]
        informative = [i for i, val in enumerate(ref) if val >= policy["reference_presence_min"]]
        mean = lambda values: sum(values) / len(values)
        ref_rate, target_rate = mean(ref), mean(target)
        prefix = modality + "_"
        metrics.update({prefix + "conditions": len(aggregates), prefix + "informative_conditions": len(informative),
                        prefix + "reference_forbidden_rate": ref_rate, prefix + "target_forbidden_rate": target_rate,
                        prefix + "reference_required_rate": mean([row["reference"]["required_rate"] for row in aggregates]),
                        prefix + "target_required_rate": mean([row["target"]["required_rate"] for row in aggregates])})
        if informative:
            ref_present = mean([ref[i] for i in informative])
            metrics[prefix + "informative_target_forbidden_rate"] = mean([target[i] for i in informative])
            metrics[prefix + "relative_reduction"] = (ref_present - metrics[prefix + "informative_target_forbidden_rate"]) / ref_present
            metrics[prefix + "improvement_rate"] = mean([float(ref[i] - target[i] >= policy["improvement_epsilon"]) for i in informative])
        metrics[prefix + "positive_drop"] = metrics[prefix + "reference_required_rate"] - metrics[prefix + "target_required_rate"]
        insufficient = len(aggregates) < policy["min_conditions"] or len(informative) < policy["min_informative_conditions"]
        if insufficient:
            violations.append({"category": "processing", "reason": f"{modality}: inconclusive, {len(aggregates)} independent conditions/{len(informative)} informative controls; need {policy['min_conditions']}/{policy['min_informative_conditions']}"})
        rules = (("target_forbidden_rate", "lte", "target_forbidden_max"),
                 ("reference_required_rate", "gte", "required_presence_min"),
                 ("target_required_rate", "gte", "required_presence_min"),
                 ("positive_drop", "lte", "positive_drop_max"))
        if not insufficient:
            rules += (("informative_target_forbidden_rate", "lte", "target_forbidden_max"),
                      ("relative_reduction", "gte", "relative_reduction_min"), ("improvement_rate", "gte", "improvement_rate_min"))
        for name, compare, policy_key in rules:
            actual, bound = metrics[prefix + name], policy[policy_key]
            if actual > bound if compare == "lte" else actual < bound:
                violations.append({"category": "baseline", "reason": f"{prefix + name}={actual:.6f} violates provisional {compare} {bound}"})
    for modality, block in (("video", "vision"), ("audio", "audio")):
        if block in spec and not any(modality in row for row in rows):
            violations.append({"category": "processing", "reason": f"Enabled {modality} scorer has no annotated conditions"})
    status = "inconclusive" if any(v["category"] == "processing" for v in violations) else "fail" if violations else "pass"
    metrics["passed"] = int(status == "pass")
    return metrics, {"status": status, "passed": status == "pass", "provisional": True,
                     "policy_profile": PROFILE, "violations": violations,
                     "note": "Surrogate detector/contrastive scores on a custom paired dataset; not official benchmark certification. Repeated identical seeds do not increase condition counts."}


def score_pairs(directory: Path, pairs, spec, config, scorers=None):
    spec = normalize_spec(spec)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Semantic scoring pairs must be a nonempty list")
    constraints = []
    for pair in pairs:
        if not isinstance(pair, Mapping) or not isinstance(pair.get("reference"), Mapping) or not isinstance(pair.get("target"), Mapping):
            raise ValueError("Scoring pair must contain reference/target request objects")
        a, b = pair["reference"], pair["target"]
        if a.get("conditioning") != "text" or b.get("conditioning") != "text":
            raise ValueError("Negative semantic control prioritizes direct text; embedding is not required")
        if a.get("parameters") != b.get("parameters") or a.get("prompt") != b.get("prompt"):
            raise ValueError("Paired prompt, seed and generation parameters must be identical")
        if a.get("checks") != b.get("checks"):
            raise ValueError("Paired semantic annotations differ")
        parameters = a.get("parameters") or {}
        if (not isinstance(parameters.get("seed"), int) or isinstance(parameters["seed"], bool)
                or parameters["seed"] < 0 or not isinstance(parameters.get("guidance_scale"), (int, float))
                or not math.isfinite(parameters["guidance_scale"]) or parameters["guidance_scale"] <= 1):
            raise ValueError("Paired semantics require an explicit seed and guidance_scale > 1")
        inputs = (a.get("request_input") or {}, b.get("request_input") or {})
        if inputs[0].get("prompt") != a["prompt"] or inputs[1].get("prompt") != a["prompt"]:
            raise ValueError("Archived request text differs from paired prompt")
        if (not inputs[0].get("negative_prompt") or not inputs[1].get("negative_prompt")
                or inputs[0]["negative_prompt"] == inputs[1]["negative_prompt"]):
            raise ValueError("Negative-control score requires distinct actual reference/target negative prompts")
        constraints.append(validate_constraints(b.get("checks"), spec))
    if not pairs:
        raise ValueError("No negative-control pairs")
    coverage = [(p.get("sample_id"), p.get("repeat")) for p in pairs]
    if any(not isinstance(sample, str) or not sample or type(repeat) is not int or repeat <= 0 for sample, repeat in coverage):
        raise ValueError("Pair sample_id/repeat must be a nonempty ID and positive repeat index")
    if len(set(coverage)) != len(coverage):
        raise ValueError("Duplicate semantic scoring pairs")
    if scorers is None:
        from xtask.benchmarks.vbench.text_video.semantic_models import load_scorers
        scorers = load_scorers(spec)
    rows = []
    for pair, annotation in zip(pairs, constraints):
        a, b = pair["reference"], pair["target"]
        inputs = (a.get("request_input") or {}, b.get("request_input") or {})
        key = hashlib.sha256(json.dumps([a["prompt"], a["parameters"], inputs, annotation], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        row = {"sample_id": pair["sample_id"], "repeat": pair["repeat"], "condition_key": key}
        for modality, keys, scorer_name in (("video", VISUAL_KEYS, "vision"), ("audio", AUDIO_KEYS, "audio")):
            if keys[0] not in annotation:
                continue
            detail = {}
            for arm in ("reference", "target"):
                record = pair[arm]
                path = (directory / record["video_path"]).resolve()
                evidence = scorers[scorer_name].score(path, annotation[keys[0]], annotation[keys[1]], config["media"])
                detail[arm] = {**presence_summary(evidence["required"], evidence["forbidden"]), "evidence": evidence}
            row[modality] = detail
        rows.append(row)
        write_json(directory / "negative_control_partial.json", rows)
    metrics, assessment = summarize(rows, spec)
    public_rows = [{k: v for k, v in row.items() if k != "condition_key"} for row in rows]
    provenance = {name: scorer.provenance for name, scorer in scorers.items()}
    result = {"complete": True, "profile": PROFILE, "metrics": metrics, "assessment": assessment,
              "pairs": public_rows, "provenance": provenance, "specification": spec,
              "sources": ["VBench/T2V-CompBench: sampled-frame object presence", "GroundingDINO: text-grounded detection", "CLAP: audio-text cosine similarity; xTask uses annotated competing descriptions"],
              "scope": "Object/audio-event exclusion and positive-content retention, not action/physics/lip-sync or universal semantic quality"}
    write_json(directory / "negative_control.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline paired semantic scoring; provisional xTask profile")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pairs_path = Path(args.pairs).resolve()
    configuration = json.loads(Path(args.config).read_text())
    pairs = json.loads(pairs_path.read_text())
    score_pairs(pairs_path.parent, pairs, configuration["specification"], configuration["task_config"])
    generated = pairs_path.parent / "negative_control.json"
    target = Path(args.output).resolve()
    if generated != target:
        write_json(target, json.loads(generated.read_text()))
    return 0  # Policy fail/inconclusive is data, not a broken evaluator process.


if __name__ == "__main__":
    raise SystemExit(main())
