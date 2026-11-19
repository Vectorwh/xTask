# Copyright 2026 The xTask Authors.
"""Explicit external paired-check contract: never infer success from differing files."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from .artifacts import write_json


def run_checks(directory: Path, records, config):
    result = {}
    pairs = []
    grouped = {}
    for record in records:
        arms = grouped.setdefault((record["sample_id"], record["repeat"]), {})
        if record["arm"] in arms:
            raise ValueError("Duplicate paired request evidence")
        arms[record["arm"]] = record
    for key, arms in grouped.items():
        if set(arms) != {"reference", "target"}:
            raise ValueError("Paired evaluator requires complete reference/target coverage")
        pairs.append({"sample_id": key[0], "repeat": key[1], **arms})
    manifest = directory / "pairs.json"
    write_json(manifest, pairs)
    for name, specification in config["checks"].items():
        output = directory / (name + ".json")
        if name == "negative_control" and specification.get("backend") == "benchmark_semantics":
            score_config = directory / "negative_scorer_config.json"
            write_json(score_config, {"specification": specification, "task_config": {"media": config["media"]}})
            command = [specification.get("python_exec", sys.executable), str(Path(__file__).with_name("negative.py").resolve()),
                       "--pairs", str(manifest.resolve()), "--config", str(score_config.resolve()),
                       "--output", str(output.resolve())]
            write_json(directory / "negative_control_command.json", command)
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONPATH="")
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=specification["timeout"],
                                           cwd=str(directory), env=environment)
            except subprocess.TimeoutExpired as exc:
                (directory / "negative_control_stderr.log").write_text(str(exc))
                raise RuntimeError("Negative-control semantic scorer timed out") from exc
            (directory / "negative_control_stdout.log").write_text(completed.stdout)
            (directory / "negative_control_stderr.log").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"Negative-control semantic evaluator failed: {completed.returncode}; see negative_control_stderr.log")
            payload = json.loads(output.read_text())
        elif name == "embedding_parity" and specification.get("backend") == "decoded_media":
            from .parity import compare_pairs

            payload = compare_pairs(directory, pairs, specification, config)
        else:
            command = specification["command"] + ["--pairs", str(manifest), "--output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=specification.get("timeout", 600), cwd=str(directory))
            (directory / (name + "_stdout.log")).write_text(completed.stdout)
            (directory / (name + "_stderr.log")).write_text(completed.stderr)
            write_json(directory / (name + "_command.json"), command)
            if completed.returncode:
                raise RuntimeError(f"{name} evaluator failed: {completed.returncode}")
            payload = json.loads(output.read_text())
        expected = {(row["sample_id"], row["repeat"]) for row in pairs}
        scored = payload.get("pairs", []) if isinstance(payload, dict) else []
        if not isinstance(scored, list) or any(not isinstance(row, dict) for row in scored):
            raise ValueError(f"{name} evaluator pairs must be a list of result mappings")
        actual = [(row.get("sample_id"), row.get("repeat")) for row in scored]
        if not isinstance(payload, dict) or payload.get("complete") is not True or set(actual) != expected or len(actual) != len(expected):
            raise ValueError(f"{name} evaluator returned incomplete/duplicate pair coverage")
        metrics = payload.get("metrics")
        if (not isinstance(metrics, dict) or not metrics
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values())):
            raise ValueError(f"{name} evaluator must return finite explicit metrics")
        if set(metrics) & {"complete", "scored_pairs", "pairs"}:
            raise ValueError(f"{name} evaluator metrics cannot override reserved status fields")
        result[name] = {"complete": True, **metrics, "scored_pairs": len(actual)}
        if name == "negative_control" and specification.get("backend") == "benchmark_semantics":
            result[name]["assessment"] = {key: payload["assessment"][key] for key in (
                "status", "passed", "provisional", "policy_profile", "violations",
            )}
            result[name]["profile"] = payload["profile"]
    result["checks_status"] = {name: "scored" for name in config["checks"]}
    return result
