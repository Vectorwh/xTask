# Copyright 2026 The xTask Authors.
"""Native video metrics/gates; never fake I2V total scores or LLM token rates."""
from __future__ import annotations

import json
import os
import socket
import shutil
from pathlib import Path

from .artifacts import new_date_dir, redact, write_json
from .config import normalize_config, result_issues


def task_config(task):
    raw = dict((task.get("benchmark") or {}).get("vbench") or {})
    raw["metrics"] = task.get("metrics", "acc")
    return normalize_config(raw)


def post_process(task, output_path, init_entries):
    root = Path(output_path)
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and p.name not in set(init_entries or [])
                        and len(p.name) == 15 and p.name[:8].isdigit() and p.name[8] == "_" and p.name[9:].isdigit())
    if not candidates:
        raise FileNotFoundError("Text-video runner did not create a date archive")
    directory = candidates[-1]
    result_path = directory / "text_video_result.json"
    if not result_path.is_file():
        raise FileNotFoundError("Missing text-video result in current date archive")
    try:
        payload = json.loads(result_path.read_text())
    except (ValueError, UnicodeDecodeError):
        # Keep corrupt bytes; never replace them with a successful empty run.
        shutil.copyfile(result_path, directory / "text_video_result.corrupt.json")
        raise
    if not isinstance(payload, dict) or payload.get("adapter") != "text_video":
        raise ValueError("Not a text-video result")
    config = task_config(task)
    expected_count = None
    manifest = root / "text_video_inputs/manifest.json"
    if config["mode"] != "evaluate" and manifest.is_file():
        from .run import read_manifest

        expected_count = len(read_manifest(manifest, config["generation"]["limit"])) * config["generation"]["repeats"]
        expected_count *= 2 if config["scenario"] in {"cfg_pair", "embedding_pair"} else 1
    if payload.get("state") == "running":
        payload["state"] = "interrupted"
        payload["complete"] = False
        payload["execution_error"] = "Runner did not finalize; partial generation must not pass"
    if expected_count is not None and payload.get("expected_requests") != expected_count:
        payload["processing_error"] = "Result expected request count differs from staged manifest/configuration"
    records_file = directory / "requests.json"
    if payload.get("complete") is True:
        try:
            from .artifacts import sha256

            rows = json.loads(records_file.read_text())
            good_ids = [(row["sample_id"], row["repeat"], row["arm"]) for row in rows]
            if len(rows) != payload.get("expected_requests") or len(set(good_ids)) != len(rows):
                raise ValueError("Final request evidence has missing or duplicated entries")
            for row in rows:
                if not row.get("success") or row.get("media", {}).get("fully_decoded") is not True:
                    raise ValueError("Final request media validation missing/failed")
                video = (directory / row["video_path"]).resolve()
                allowed_root = Path(config["source_run"]).resolve() if config["mode"] == "evaluate" else directory.resolve()
                if allowed_root not in video.parents:
                    raise ValueError("Video escaped declared generation archive")
                if sha256(video) != row["video_sha256"]:
                    raise ValueError("Archived video bytes changed")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            payload["processing_error"] = str(exc)
    payload["issues"] = result_issues(payload, config)
    payload["result_file"] = "text_video_result.json"
    write_json(result_path, payload)
    return payload, str(directory)


def verify(result, task):
    try:
        issues = result_issues(result.get("benchmark_vbench_result"), task_config(task))
    except (TypeError, ValueError) as exc:
        issues = [{"category": "processing", "reason": f"Invalid text-video configuration: {exc}"}]
    if result.get("benchmark_success") is False and not issues:
        issues.insert(0, {"category": "execution", "reason": "Text-video command failed"})
    if result.get("archive_error"):
        issues.insert(0, {"category": "processing", "reason": str(result["archive_error"])})
    result["video_gate_issues"] = issues
    return not issues


def failure(result, task):
    issues = result.get("video_gate_issues")
    if not issues:
        try:
            issues = result_issues(result.get("benchmark_vbench_result"), task_config(task))
        except (TypeError, ValueError) as exc:
            issues = [{"category": "processing", "reason": f"Invalid text-video configuration: {exc}"}]
    if result.get("archive_error"):
        issues = [{"category": "processing", "reason": str(result["archive_error"])}] + issues
    category = next((kind for kind in ("execution", "processing", "baseline") if any(item["category"] == kind for item in issues)), "processing")
    labels = {"execution": "benchmark执行异常", "processing": "benchmark结果处理异常", "baseline": "benchmark结果与基线比对失败"}
    return labels[category], "; ".join(item["reason"] for item in issues)


def workbook_payload(result):
    payload = result.get("benchmark_vbench_result") or {}
    fields = ("task", "scenario", "complete", "expected_requests", "completed_requests", "failed",
              "performance", "performance_by_arm", "video_quality", "checks_status", "negative_control",
              "embedding_parity", "result_file", "source_run", "offline_evaluation")
    return {"video": {key: payload[key] for key in fields if key in payload}}


def archive_execution(context):
    """Keep command/state evidence local; service logs use existing scheduler archive."""
    root = Path(context.output_file_path)
    path = context.result.get("benchmark_output_path")
    if path:
        directory = Path(path)
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir() and p.name not in set(context.state.get("init_entries") or [])
                            and len(p.name) == 15 and p.name[:8].isdigit() and p.name[8] == "_" and p.name[9:].isdigit())
        directory = candidates[-1] if candidates else new_date_dir(root)
        context.result["benchmark_output_path"] = str(directory)
    if not context.result.get("artifact_host"):
        context.result["artifact_host"] = context.state.get("local_ip") or socket.gethostbyname(socket.gethostname())
    task = context.result.get("task_config") or context.state.get("task_config") or {}
    write_json(directory / "task_config.json", redact(task))
    runtime_config = directory / "config.json"
    if runtime_config.is_file():
        effective = json.loads(runtime_config.read_text())
        effective["task_snapshot"] = redact(task)
        write_json(runtime_config, effective)
        write_json(directory / "runner_config.json", effective)
    for filename in ("benchmark_stdout", "benchmark_stderr"):
        (directory / (filename + ".log")).write_text(context.result.get(filename) or "")
    source = Path(getattr(context, "task_yaml_path", ""))
    if source.is_file():
        import yaml
        snapshot = yaml.safe_load(source.read_text())
        (directory / "source_task.yaml").write_text(yaml.safe_dump(redact(snapshot), allow_unicode=True, sort_keys=False))
    script = root / "run_vbench_test.sh"
    if script.is_file():
        shutil.copyfile(script, directory / script.name)
    service_commands = {}
    for runtime in context.state.get("service_runtimes") or []:
        plan = getattr(runtime, "plan", None)
        if plan is not None:
            name = getattr(plan, "instance_name", "instance")
            service_commands[name] = getattr(plan, "startup_cmd", None)
            # The root-level launch script is overwritten by the next run.
            # Keep the real flags/environment next to this date's command.
            output_dir = getattr(plan, "output_dir", None)
            if output_dir:
                for launch in Path(output_dir).glob("start_xllm_serv*.sh"):
                    target = directory / "service_startup" / name / launch.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(launch, target)
    write_json(directory / "commands.json", {"service_instances": service_commands, "startup": context.result.get("startup_command"),
                "benchmark": context.result.get("benchmark_command") or context.state.get("benchmark_cmd")})
    write_json(directory / "execution_result.json", {
        "success": bool(context.result.get("success")), "message": context.result.get("message"),
        "benchmark_exit_code": context.result.get("benchmark_exit_code"),
        "report_generated": context.result.get("report_generated"),
        "archive_error": context.result.get("archive_error"), "artifact_host": context.result["artifact_host"],
        "artifact_path": str(directory), "execution_time": context.result.get("execution_time"),
    })
