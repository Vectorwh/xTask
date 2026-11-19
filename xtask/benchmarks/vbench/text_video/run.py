# Copyright 2026 The xTask Authors.
"""Fresh JoyOV2 generation with explicit task/conditioning and request archives."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import hashlib
import csv
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from xtask.benchmarks.vbench.text_video.artifacts import new_date_dir, redact, sha256, write_json
from xtask.benchmarks.vbench.text_video.config import normalize_config, result_issues
from xtask.benchmarks.vbench.text_video.embedding import load_embedding, validate_cfg
from xtask.benchmarks.vbench.text_video.media import validate_media


def read_manifest(path: Path, limit=0):
    text = path.read_text(encoding="utf-8")
    records = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must be a nonempty JSON list or JSONL")
    allowed = {"id", "prompt", "negative_prompt", "neutral_negative_prompt", "embedding",
               "negative_embedding", "encoder", "seed", "checks", "category"}
    ids = set()
    for item in records:
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError("Unknown manifest fields; JoyOV2 does not consume reference image/video/audio")
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError("manifest requires unique nonempty string IDs")
        ids.add(sample_id)
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError(f"Missing prompt: {sample_id}")
        for name in ("negative_prompt", "neutral_negative_prompt"):
            if name in item and (not isinstance(item[name], str) or not item[name].strip()):
                raise ValueError(f"{name} must be a nonempty string: {sample_id}")
        if "seed" in item and (isinstance(item["seed"], bool) or not isinstance(item["seed"], int) or item["seed"] < 0):
            raise ValueError("Sample seed must be nonnegative integer")
    return records[:limit] if limit else records


def prepare_requests(config, directory: Path):
    manifest = Path(config["manifest"])
    records = read_manifest(manifest, config["generation"]["limit"])
    archived_records = copy.deepcopy(records)
    assets = directory / "embeddings"
    generation = config["generation"]
    wire_fields = {"size", "num_frames", "fps", "num_inference_steps", "guidance_scale",
                   "flow_shift", "seed", "sampling_rate", "num_videos_per_prompt"}
    params = {k: v for k, v in generation.items() if k in wire_fields}
    prepared = []
    for index, item in enumerate(records):
        options = dict(params, seed=item.get("seed", params["seed"]), output_type="base64")
        if config["scenario"] in {"text", "cfg_pair"} and any(item.get(name) for name in ("embedding", "negative_embedding")):
            raise ValueError("Text scenario cannot silently ignore supplied embeddings")
        text_input = {"prompt": item["prompt"]}
        if item.get("negative_prompt"):
            text_input["negative_prompt"] = item["negative_prompt"]
        if config["scenario"] in {"text", "cfg_pair"} and options["guidance_scale"] > 1 and not item.get("negative_prompt"):
            raise ValueError(f"JoyOV2 CFG text path requires negative_prompt: {item['id']}")
        arms = [("main", text_input)]
        feature_provenance = {}
        if config["checks"].get("negative_control", {}).get("backend") == "benchmark_semantics":
            from xtask.benchmarks.vbench.text_video.negative import validate_constraints

            validate_constraints(item.get("checks"), config["checks"]["negative_control"])
        if config["scenario"] == "cfg_pair":
            neutral = item.get("neutral_negative_prompt")
            if not neutral or neutral == item["negative_prompt"]:
                raise ValueError("cfg_pair requires distinct explicit neutral_negative_prompt and negative_prompt")
            arms = [("reference", dict(text_input, negative_prompt=neutral)), ("target", text_input)]
        elif config["scenario"] in {"embedding", "embedding_pair"}:
            dimension = config.get("embedding_dim", 5120)
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise ValueError("embedding_dim must be a positive integer matching transformer config")
            tensor, source = load_embedding(item.get("embedding"), manifest.parent, dimension, item.get("encoder"))
            embedded_input = {"prompt_embed": tensor}
            feature_provenance["prompt_embed"] = source
            negative = None
            if item.get("negative_embedding"):
                negative, negative_source = load_embedding(item["negative_embedding"], manifest.parent, dimension, item.get("encoder"))
                embedded_input["negative_prompt_embed"] = negative
                feature_provenance["negative_prompt_embed"] = negative_source
            validate_cfg(tensor, negative, options["guidance_scale"])
            if config["scenario"] == "embedding_pair":
                if options["guidance_scale"] > 1 and (negative is None or not item.get("negative_prompt")):
                    raise ValueError("Embedding parity CFG requires separate negative_embedding and matching negative_prompt")
                for value in (tensor, negative):
                    if value is not None:
                        lengths = (value.get("parameters") or {}).get("joyo_text_seqlens", {}).get("int64_tensor_param")
                        if lengths is not None and len(lengths) != 1:
                            raise ValueError("Embedding parity needs unpacked positive/negative inputs")
            # Snapshot feature files before generating. Future trials cannot replace them.
            assets.mkdir(exist_ok=True)
            for field, feature in feature_provenance.items():
                target = assets / f"{index:05d}_{field}.json"
                shutil.copyfile(feature["source_path"], target)
                feature["archive_path"] = str(target.relative_to(directory))
                descriptor_key = "embedding" if field == "prompt_embed" else "negative_embedding"
                archived_records[index][descriptor_key]["path"] = feature["archive_path"]
                if sha256(target) != feature["sha256"]:
                    raise ValueError("Embedding changed during staging")
            arms = [("main", embedded_input)] if config["scenario"] == "embedding" else [("reference", text_input), ("target", embedded_input)]
        for repeat in range(1, generation["repeats"] + 1):
            for arm, input_data in arms:
                prepared.append({"sample_id": item["id"], "repeat": repeat, "arm": arm,
                                 "filename": f"{index:05d}_r{repeat}_{arm}.mp4", "parameters": options,
                                 "payload": {"model": config["model"], "input": input_data, "parameters": options},
                                 "endpoint": config.get("reference_endpoint", config["endpoint"]) if arm == "reference" else config["endpoint"],
                                 "embedding_provenance": feature_provenance if arm != "reference" else {},
                                 "conditioning": "embedding" if "prompt_embed" in input_data else "text",
                                 "prompt": item["prompt"], "checks": item.get("checks", {})})
    write_json(directory / "manifest.json", archived_records)
    return prepared


def _request(prepared, config, directory: Path, index):
    entry = {k: v for k, v in prepared.items() if k != "payload"}
    # Keep compact actual inputs; feature bodies reside in immutable files.
    entry["request_input"] = {k: ("<archived embedding>" if k.endswith("embed") else v)
                              for k, v in prepared["payload"]["input"].items()}
    data = json.dumps(prepared["payload"], ensure_ascii=False, allow_nan=False).encode("utf-8")
    entry["request_body_sha256"] = hashlib.sha256(data).hexdigest()
    entry["request_bytes"] = len(data)
    headers = {"Content-Type": "application/json"}
    if config.get("api_key"):
        headers["Authorization"] = "Bearer " + config["api_key"]
    entry["started_at_unix_s"] = time.time()
    entry["started"] = time.monotonic()
    try:
        request = urllib.request.Request(prepared["endpoint"], data=data, headers=headers, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=config["generation"]["request_timeout"]) as response:
            body = response.read(256 * 1024 * 1024 + 1)
        entry["finished"] = time.monotonic()
        if len(body) > 256 * 1024 * 1024:
            raise ValueError("Video response exceeds 256 MiB limit")
        payload = json.loads(body)
        results = (payload.get("output") or {}).get("results")
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError("Expected exactly one output.results video")
        media = results[0].get("video")
        if not isinstance(media, dict) or media.get("type") != "base64" or not isinstance(media.get("base64"), str):
            raise ValueError("Expected native xLLM Base64 MediaSource response")
        video = base64.b64decode(media["base64"], validate=True)
        if not video:
            raise ValueError("Empty generated video")
        target = directory / "videos" / prepared["filename"]
        target.write_bytes(video)
        entry.update(success=True, video_path=str(target.relative_to(directory)), video_sha256=sha256(target),
                     response_id=payload.get("id"), response_bytes=len(body))
    except urllib.error.HTTPError as exc:
        entry.setdefault("finished", time.monotonic())
        entry.update(success=False, error=f"{type(exc).__name__}: {exc}", http_status=exc.code)
        # Preserve bounded server diagnostics (e.g. model-ID admission errors)
        # rather than reporting only "404". Never retry or count this as success.
        try:
            diagnostic = exc.read(8193)
            text = diagnostic[:8192].decode("utf-8", errors="replace")
            try:
                text = json.dumps(redact(json.loads(text)), ensure_ascii=False)
            except ValueError:
                pass
            if config.get("api_key"):
                text = text.replace(config["api_key"], "***")
            entry["http_error_body"] = text
            entry["http_error_body_truncated"] = len(diagnostic) > 8192
        except Exception:
            entry["http_error_body"] = "<unavailable>"
        finally:
            exc.close()
    except Exception as exc:
        entry.setdefault("finished", time.monotonic())
        entry.update(success=False, error=f"{type(exc).__name__}: {exc}")
    entry["e2el_ms"] = (entry["finished"] - entry["started"]) * 1000
    entry["finished_at_unix_s"] = time.time()
    entry["request_index"] = index
    return entry


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def metrics(records):
    good = [row for row in records if row.get("success")]
    if not good:
        return {}
    elapsed = max(row["finished"] for row in records) - min(row["started"] for row in records)
    latency = [row["e2el_ms"] for row in good]
    return {"e2el_avg_ms": sum(latency) / len(latency), "e2el_p90_ms": percentile(latency, .9),
            "e2el_p99_ms": percentile(latency, .99), "request_throughput": len(good) / elapsed,
            "video_throughput": sum(row["parameters"]["num_frames"] / row["parameters"]["fps"] for row in good) / elapsed,
            "window_s": elapsed, "measured_requests": len(good),
            "timing_boundary": "HTTP send to complete response read; excludes offline embedding creation, media decode and quality evaluation",
            "small_sample_warning": len(good) < 100}


def run(config, output_root: Path):
    config = normalize_config(config)
    directory = new_date_dir(output_root)
    (directory / "videos").mkdir(mode=0o700)
    snapshot = redact(config)
    snapshot["manifest"] = "manifest.json"
    write_json(directory / "config.json", snapshot)
    write_json(directory / "runner_config.json", snapshot)
    payload = {"schema_version": 1, "adapter": "text_video", "state": "running",
               "task": config["task"], "scenario": config["scenario"], "complete": False,
               "failed": 0, "expected_requests": 0, "completed_requests": 0,
               "performance": {}, "video_quality": {}, "checks_status": {},
               "cache_condition": "one service lifetime; not a cold-cache guarantee"}
    write_json(directory / "text_video_result.json", payload)
    records = []
    try:
        payload["manifest_sha256"] = sha256(Path(config["manifest"]))
        prepared = prepare_requests(config, directory)
        semantic = config["checks"].get("negative_control", {})
        if semantic.get("backend") == "benchmark_semantics":
            from xtask.benchmarks.vbench.text_video.semantic_models import preflight_checkpoints, preflight_environment

            preflight_checkpoints(semantic)
            preflight_environment(semantic, directory)  # Isolated Python probe; no generator requests or model bodies loaded.
        payload["expected_requests"] = len(prepared)
        write_json(directory / "request_plan.json", [{k: v for k, v in p.items() if k != "payload"} for p in prepared])
        write_json(directory / "text_video_result.json", payload)
        # Explicit warmups are real requests, checked but never in measured results.
        warmups = []
        for index in range(config["generation"]["warmups"]):
            warmup_request = dict(prepared[0], filename=f"warmup_{index:05d}.mp4")
            entry = _request(warmup_request, config, directory, -index - 1)
            warmups.append(entry)
            write_json(directory / "warmups.json", warmups)
            if not entry["success"]:
                raise RuntimeError("Warmup request failed: " + entry["error"])
            validate_media(directory / entry["video_path"], entry["parameters"], config["media"])
        with (directory / "requests_http.jsonl").open("w", encoding="utf-8") as journal:
            with concurrent.futures.ThreadPoolExecutor(max_workers=config["generation"]["concurrency"]) as pool:
                futures = [pool.submit(_request, item, config, directory, index) for index, item in enumerate(prepared)]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    records.append(record)
                    journal.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    payload["completed_requests"] = len(records)
                    payload["failed"] = sum(not row["success"] for row in records)
                    write_json(directory / "text_video_result.json", payload)
        # Decode after all HTTP completions: validation must not inflate the perf window.
        for record in records:
            if not record["success"]:
                continue
            try:
                record["media"] = validate_media(directory / record["video_path"], record["parameters"], config["media"])
            except Exception as exc:
                record.update(success=False, error=f"Media validation: {exc}")
        records.sort(key=lambda row: row["request_index"])
        write_json(directory / "requests.json", records)
        with (directory / "requests.jsonl").open("w", encoding="utf-8") as journal:
            for row in records:
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        payload["completed_requests"] = len(records)
        payload["failed"] = sum(not row["success"] for row in records)
        payload["complete"] = len(records) == len(prepared) and payload["failed"] == 0
        payload["performance"] = metrics(records)
        payload["performance_by_arm"] = {arm: metrics([row for row in records if row["arm"] == arm]) for arm in {row["arm"] for row in records}}
        paired = config["scenario"] in {"cfg_pair", "embedding_pair"}
        if paired:
            payload["performance"]["paired_workload"] = True
            for value in payload["performance_by_arm"].values():
                value["paired_workload"] = True
                value["note"] = "Arm window may include other-arm executions; not an independent model throughput baseline"
            payload["checks_status"]["negative_control" if config["scenario"] == "cfg_pair" else "embedding_parity"] = "not_scored" if not config["checks"] else "pending"
        if payload["complete"] and config["evaluation"]["enabled"]:
            from xtask.benchmarks.vbench.text_video.evaluate import evaluate
            # Score target/main only. Reference videos are not allowed to contaminate aggregate quality.
            payload["video_quality"] = evaluate(directory, [row for row in records if row["arm"] != "reference"], config["evaluation"])
        if payload["complete"] and config["checks"]:
            from xtask.benchmarks.vbench.text_video.checks import run_checks
            payload.update(run_checks(directory, records, config))
    except BaseException as exc:
        error_key = "processing_error" if payload["complete"] else "execution_error"
        payload[error_key] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            payload["state"] = "interrupted"
        else:
            payload["state"] = "failed"
        if error_key == "execution_error":
            payload["complete"] = False
    finally:
        if payload["state"] == "running":
            payload["state"] = "completed"
        if records and not (directory / "requests.json").exists():
            write_json(directory / "requests.json", records)
        if (directory / "manifest.json").exists():
            snapshot["manifest_sha256"] = sha256(directory / "manifest.json")
            write_json(directory / "config.json", snapshot)
            write_json(directory / "runner_config.json", snapshot)
        payload["issues"] = result_issues(payload, config)
        write_json(directory / "text_video_result.json", payload)
        with (directory / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "repeat", "arm", "conditioning", "success", "e2el_ms", "video_path", "error"])
            writer.writeheader()
            writer.writerows({k: row.get(k, "") for k in writer.fieldnames} for row in records)
    print(json.dumps({"date_dir": str(directory), "complete": payload["complete"], "issues": payload["issues"]}, ensure_ascii=False))
    return payload, directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config.get("mode") == "evaluate":
        from xtask.benchmarks.vbench.text_video.evaluate import evaluate_archive
        payload, _ = evaluate_archive(config, Path(args.output_dir))
    else:
        payload, _ = run(config, Path(args.output_dir))
    return 1 if payload.get("issues") else 0


if __name__ == "__main__":
    raise SystemExit(main())
