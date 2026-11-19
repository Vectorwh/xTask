# Copyright 2026 The xTask Authors.
"""Generate only explicitly selected T2V/T2AV scripts, never replace Wan I2V."""
from __future__ import annotations

import shlex
from pathlib import Path
from urllib.parse import urlsplit

from xtask.utils.container_manager import path_mapping_in_container
from .artifacts import write_json
from .config import normalize_config
from .run import read_manifest


def endpoint_url(value):
    value = str(value).rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Video endpoint must be an HTTP(S) URL without embedded credentials/query/fragment")
    return value if parsed.path.endswith("/video/generation") else value + ("/video/generation" if value.endswith("/v1") else "/v1/video/generation")


def generate_script(task_config, output_path, shell_file="run_vbench_test.sh", python_exec="python3"):
    raw = dict(task_config["benchmark"]["vbench"])
    raw["metrics"] = task_config.get("metrics", "acc")
    config = normalize_config(raw)
    model = task_config.get("served_model_name") or config.get("model")
    if not model:
        raise ValueError("Text-video served_model_name must be explicit")
    config["model"] = model
    endpoint = raw.get("endpoint")
    if not endpoint:
        endpoint = f"http://{task_config.get('host_addr', '127.0.0.1')}:{task_config.get('port', 14830)}"
    config["endpoint"] = endpoint_url(endpoint)
    if config.get("reference_endpoint"):
        config["reference_endpoint"] = endpoint_url(config["reference_endpoint"])
    from xtask.core.external_service import is_external_service_enabled, resolve_external_location
    if is_external_service_enabled(task_config) and config["mode"] != "evaluate":
        location = resolve_external_location(task_config)
        if not raw.get("endpoint"):
            config["endpoint"] = endpoint_url(location.url)
        config["api_key"] = config.get("api_key") or location.api_key
    config["task_snapshot"] = {key: task_config[key] for key in (
        "case_name", "engine_type", "model_path", "served_model_name", "external_service",
        "deployment", "benchmark", "metrics",
    ) if key in task_config}
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "text_video_inputs"
    inputs.mkdir(mode=0o700, exist_ok=True)
    manifest = Path(config["manifest"]).expanduser().resolve()
    records = read_manifest(manifest)
    semantic = config["checks"].get("negative_control", {})
    if semantic.get("backend") == "benchmark_semantics":
        from .negative import validate_constraints

        if semantic.get("python_exec", "").startswith("/"):
            semantic["python_exec"] = path_mapping_in_container(semantic["python_exec"])

        for record in records:
            validate_constraints(record.get("checks"), semantic)
        for modality in ("vision", "audio"):
            if modality in semantic:
                semantic[modality]["model_path"] = path_mapping_in_container(
                    str(Path(semantic[modality]["model_path"]).expanduser().resolve())
                )
    # Stage immutable input features alongside manifest. Do not assume a remote
    # runner can read caller-local paths or download the model's checkpoint.
    for index, record in enumerate(records):
        for field in ("embedding", "negative_embedding"):
            if field not in record:
                continue
            descriptor = record[field]
            if not isinstance(descriptor, dict) or not descriptor.get("path"):
                raise ValueError(f"Invalid {field} descriptor")
            source = Path(descriptor["path"])
            source = source if source.is_absolute() else manifest.parent / source
            target = inputs / f"{index:05d}_{field}.json"
            target.write_bytes(source.read_bytes())
            record[field] = {**descriptor, "path": target.name}
    write_json(inputs / "manifest.json", records)
    config["manifest"] = path_mapping_in_container(str((inputs / "manifest.json").resolve()))
    config["source_manifest"] = str(manifest)
    if config.get("source_run"):
        config["source_run"] = path_mapping_in_container(str(Path(config["source_run"]).expanduser().resolve()))
    code_root = config["evaluation"].get("code_root")
    if code_root:
        config["evaluation"]["code_root"] = path_mapping_in_container(str(Path(code_root).expanduser().resolve()))
    for specification in config["checks"].values():
        if "command" in specification:
            specification["command"] = [path_mapping_in_container(arg) if arg.startswith("/") else arg for arg in specification["command"]]
    config_file = inputs / "runner_config.json"
    write_json(config_file, config)
    if Path(shell_file).name != shell_file:
        raise ValueError("shell_file must be a basename")
    runner = path_mapping_in_container(str(Path(__file__).with_name("run.py").resolve()))
    cmd = " ".join(shlex.quote(str(value)) for value in (
        python_exec, runner, "--config", path_mapping_in_container(str(config_file.resolve())),
        "--output-dir", path_mapping_in_container(str(output.resolve())),
    ))
    shell = output / shell_file
    shell.write_text("#!/bin/bash\nset -euo pipefail\n" + cmd + "\n", encoding="utf-8")
    shell.chmod(0o700)
    return "bash " + shlex.quote(path_mapping_in_container(str(shell.resolve())))
