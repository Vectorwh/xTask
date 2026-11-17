"""Stage the original archives and generate a reproducible ASR runner command."""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict

from xtask.benchmarks.asr.config import normalize_config
from xtask.utils.container_manager import path_mapping_in_container


def generate_asr_script(
    task_config: Dict[str, Any], output_file_path: str,
    output_shell_file: str = "run_asr_test.sh", python_exec: str = "python3",
) -> str:
    config = normalize_config((task_config.get("benchmark") or {}).get("asr") or {})
    project = Path(__file__).resolve().parents[3]
    output = Path(output_file_path).resolve()
    if Path(output_shell_file).name != output_shell_file:
        raise ValueError("ASR output_shell_file must be a filename")
    inputs = output / "asr_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    # Keep the dataset and client runner-local.  Existing case-output sync
    # transports these archives without shipping them to every service worker.
    for name, default, target in (
        ("infer_archive", project / "asr_doc/infer.zip", "infer.zip"),
        ("dataset_archive", project / "asr_doc/data.zip", "data.zip"),
    ):
        source = Path(config.get(name) or default).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ASR {name} not found: {source}")
        destination = inputs / target
        if source != destination:
            shutil.copy2(source, destination)
        config[name] = path_mapping_in_container(str(destination))
    endpoint = config.get("server_addr")
    api_key = config.get("api_key")
    from xtask.core.external_service import is_external_service_enabled, resolve_external_location

    if is_external_service_enabled(task_config):
        location = resolve_external_location(task_config)
        if not endpoint:
            endpoint = location.origin + location.base_path
        api_key = api_key or location.api_key
    endpoint = str(endpoint or f"http://{task_config.get('host_addr', '127.0.0.1')}:{task_config.get('port', 8018)}/v1")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "http://" + endpoint
    endpoint = endpoint.rstrip("/")
    for suffix in ("/audio/transcriptions", "/chat/completions"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[:-len(suffix)]
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    config["server_addr"] = endpoint
    if api_key:
        config["api_key"] = str(api_key)
    config["model"] = config.get("model") or task_config.get("served_model_name")
    # Service settings and user-provided provenance are archived with each run.
    config["task_snapshot"] = {
        key: task_config[key] for key in (
            "case_name", "metrics", "model_path", "served_model_name", "engine_type",
            "deployment", "serving_extensiable_param", "startup_environment",
        ) if key in task_config
    }
    config_path = inputs / "runner_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path.chmod(0o600)
    runner = path_mapping_in_container(str(Path(__file__).with_name("run_asr.py")))
    command = " ".join(shlex.quote(str(p)) for p in (
        python_exec, runner, "--config", path_mapping_in_container(str(config_path)),
        "--output-dir", path_mapping_in_container(str(output)),
    ))
    shell = output / output_shell_file
    shell.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"cd {shlex.quote(path_mapping_in_container(str(project)))}\n"
        f"{command}\n", encoding="utf-8",
    )
    return "bash " + shlex.quote(path_mapping_in_container(str(shell)))
