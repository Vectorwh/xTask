"""Generate and execute an isolated AISBench SWE-bench runner on an x86 host.

AISBench (mini-swe-agent infer + official harness eval) is not installed into
the xLLM/NPU container. The generated Python config follows the official
``mini_swe_agent_swe_bench_verified_mini.py`` example; execution uses
HostExecutor so a remote x86 runner can be used from an ARM scheduler.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from xtask.benchmarks.aisbench.swebench import (
    DEFAULT_SWEBENCH_BATCH_SIZE,
    DEFAULT_SWEBENCH_SPLIT,
    DEFAULT_SWEBENCH_STEP_LIMIT,
    swebench_agent_container_timeout,
    swebench_config,
    swebench_dataset_name,
    swebench_direct_output_enabled,
    swebench_mode,
    swebench_runner,
    swebench_setup,
    swebench_timeout_s,
    validate_aisbench_swebench_task,
)
from xtask.utils.container_manager import CommandResult
from xtask.utils.host_executor import (
    SshHostExecutor,
    create_host_executor,
    is_local_host,
)


CONFIG_FILENAME = "aisbench_swe_config.py"
SHELL_FILENAME = "run_aisbench_swe.sh"
_SAFE_ARG = re.compile(r"^[A-Za-z0-9_./:=+-]+$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _py_str(value: Any) -> str:
    return repr(str(value))


def _safe_name(value: Any, field: str) -> str:
    text = str(value or "")
    if not text or not _SAFE_NAME.fullmatch(text):
        raise ValueError(
            f"SWE-bench {field} must contain only letters, numbers, '.', '_' or '-': {text!r}"
        )
    return text


def _safe_extra_args(raw: Any) -> List[str]:
    if raw in (None, "", []):
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ValueError("benchmark.swebench.extra_args must be a string or list")
    checked = []
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        if not _SAFE_ARG.fullmatch(token):
            raise ValueError(
                f"benchmark.swebench.extra_args contains an unsafe token: {token!r}"
            )
        checked.append(token)
    return checked


def _docker_copy_fallback_enabled(task_config: Mapping[str, Any]) -> bool:
    raw = swebench_runner(task_config).get("docker_copy_fallback", False)
    if not isinstance(raw, bool):
        raise ValueError("benchmark.swebench.runner.docker_copy_fallback must be a boolean")
    return raw


def _docker_run_retry_enabled(task_config: Mapping[str, Any]) -> bool:
    raw = swebench_runner(task_config).get("docker_run_retry", False)
    if not isinstance(raw, bool):
        raise ValueError("benchmark.swebench.runner.docker_run_retry must be a boolean")
    return raw


def _clean_infer_container_enabled(task_config: Mapping[str, Any]) -> bool:
    raw = swebench_runner(task_config).get("clean_infer_container", False)
    if not isinstance(raw, bool):
        raise ValueError("benchmark.swebench.runner.clean_infer_container must be a boolean")
    return raw


def _github_raw_retry_enabled(task_config: Mapping[str, Any]) -> bool:
    raw = swebench_runner(task_config).get("github_raw_retry", False)
    if not isinstance(raw, bool):
        raise ValueError("benchmark.swebench.runner.github_raw_retry must be a boolean")
    return raw


def _remote_home(executor) -> tuple[Optional[str], Optional[CommandResult]]:
    """Read the login user's HOME on a remote runner."""
    result = executor.run('printf %s "$HOME"', timeout=10)
    if not result.success:
        error = CommandResult(
            success=False,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr or "Failed to read $HOME on the x86 runner",
            execution_time=result.execution_time,
        )
        return None, error

    home = result.stdout.strip().rstrip("/")
    if not home.startswith("/"):
        return None, CommandResult(
            success=False,
            exit_code=-1,
            stdout=result.stdout,
            stderr=f"Invalid $HOME on the x86 runner: {home!r}",
            execution_time=result.execution_time,
        )
    return home, None


def _expand_home(value: Any, home: str) -> Any:
    """Expand HOME placeholders in a user-facing SWE-bench path."""
    if not isinstance(value, str):
        return value
    expanded = value.replace("${HOME}", home).replace("$HOME", home)
    if expanded.startswith("~/"):
        expanded = home + expanded[1:]
    return expanded


def _expand_swebench_home_paths(
    task_config: Mapping[str, Any], home: str
) -> Dict[str, Any]:
    """Return a copy with remote runner paths expanded from ``home``.

    Only SWE-bench runner paths are expanded. Source paths on the control
    node are intentionally untouched because they refer to the local machine.
    """
    import copy

    expanded = copy.deepcopy(dict(task_config))
    benchmark = expanded.setdefault("benchmark", {})
    if not isinstance(benchmark, Mapping):
        return expanded
    swebench = benchmark.setdefault("swebench", {})
    if not isinstance(swebench, Mapping):
        return expanded

    for key in ("dataset_path", "ais_bench_bin"):
        if key in swebench:
            swebench[key] = _expand_home(swebench[key], home)

    runner = swebench.setdefault("runner", {})
    if isinstance(runner, Mapping):
        for key in ("work_dir", "install_dir", "python"):
            if key in runner:
                runner[key] = _expand_home(runner[key], home)

    setup = swebench.get("setup")
    if isinstance(setup, Mapping):
        # Only expand a legacy setup block when the YAML actually provided it.
        for key in ("python", "install_dir"):
            if key in setup:
                setup[key] = _expand_home(setup[key], home)

    return expanded


def _api_url(task_config: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    explicit = str(config.get("url") or config.get("api_url") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    host = str(task_config.get("host_addr") or "127.0.0.1").strip() or "127.0.0.1"
    port = task_config.get("port", 8800)
    return f"http://{host}:{port}/v1"


def _generation_kwargs_repr(config: Mapping[str, Any]) -> str:
    raw = config.get("generation_kwargs") or {}
    if not isinstance(raw, Mapping) or not raw:
        return "dict()"
    if all(str(key).isidentifier() for key in raw):
        return "dict(" + ", ".join(f"{key}={value!r}" for key, value in raw.items()) + ")"
    return repr(dict(raw))


def _render_python_config(task_config: Mapping[str, Any]) -> str:
    config = swebench_config(task_config)
    dataset = swebench_dataset_name(task_config)
    abbr = f"swebench_{dataset}"
    model = str(
        config.get("model") or task_config.get("served_model_name") or ""
    ).strip()
    api_key = str(config.get("api_key") or "EMPTY")
    url = _api_url(task_config, config)
    split = str(config.get("split") or DEFAULT_SWEBENCH_SPLIT)
    try:
        step_limit = int(config.get("step_limit") or DEFAULT_SWEBENCH_STEP_LIMIT)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark.swebench.step_limit must be an integer") from exc
    try:
        batch_size = int(config.get("batch_size") or DEFAULT_SWEBENCH_BATCH_SIZE)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark.swebench.batch_size must be an integer") from exc
    dataset_path = str(config.get("dataset_path") or "")
    filter_spec = str(config.get("filter_spec") or "")
    shuffle = bool(config.get("shuffle", False))
    return f"""# Generated by xTask for AISBench SWE-bench. Do not use for GSM8K perf/acc.
from ais_bench.benchmark.datasets import SWEBenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import SWEBenchInferTask, SWEBenchEvalTask
from ais_bench.benchmark.summarizers import SWEBenchSummarizer

STEP_LIMIT = {step_limit}

models = [
    dict(
        attr="local",
        abbr="swebench",
        type="LiteLLMChat",
        model={_py_str(model)},
        api_key={_py_str(api_key)},
        url={_py_str(url)},
        batch_size={batch_size},
        generation_kwargs={_generation_kwargs_repr(config)},
    )
]

datasets = [
    dict(
        type=SWEBenchDataset,
        abbr={_py_str(abbr)},
        path={_py_str(dataset_path)},
        name={_py_str(dataset)},
        split={_py_str(split)},
        step_limit=STEP_LIMIT,
        filter_spec={_py_str(filter_spec)},
        shuffle={shuffle!r},
    )
]

summarizer = dict(
    attr="accuracy",
    type=SWEBenchSummarizer,
)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchEvalTask),
    ),
)
"""


def _render_shell(task_config: Mapping[str, Any]) -> str:
    config = swebench_config(task_config)
    mode = swebench_mode(task_config)
    reuse = bool(config.get("reuse", True))
    ais_bench_bin = str(config.get("ais_bench_bin") or "ais_bench")
    extra_args = _safe_extra_args(config.get("extra_args"))
    command_parts = [
        '"$AIS_BENCH_BIN"',
        '"$CONFIG_FILE"',
        "-m",
        _quote(mode),
    ]
    if reuse:
        command_parts.append("--reuse")
    if swebench_direct_output_enabled(task_config):
        # AISBench appends its timestamp to --work-dir. Use the cwd after cd
        # so a relative runner.work_dir never creates a second nested folder.
        command_parts.extend(("--work-dir", '"$PWD"'))
    command_parts.extend(_quote(item) for item in extra_args)
    command = " ".join(command_parts)
    shell_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
            f'CONFIG_FILE="$SCRIPT_DIR/{CONFIG_FILENAME}"',
            'WORK_DIR="${SWE_WORK_DIR:-$SCRIPT_DIR}"',
            f'AIS_BENCH_BIN="${{AIS_BENCH_BIN:-{_quote(ais_bench_bin)}}}"',
            'export SWE_WORK_DIR="$WORK_DIR"',
            'mkdir -p "$WORK_DIR"',
            'cd "$WORK_DIR"',
            'if [ ! -f "$CONFIG_FILE" ]; then',
            '    echo "SWE-bench AISBench config is missing: $CONFIG_FILE" >&2',
            "    exit 1",
            "fi",
            f"if ! {command}; then",
            '    echo "AISBench SWE-bench execution failed" >&2',
            "    exit 1",
            "fi",
            'echo "AISBench SWE-bench finished; results under $WORK_DIR"',
            "",
    ]
    copy_fallback = _docker_copy_fallback_enabled(task_config)
    run_retry = _docker_run_retry_enabled(task_config)
    clean_infer = _clean_infer_container_enabled(task_config)
    github_raw_retry = _github_raw_retry_enabled(task_config)
    agent_container_timeout = swebench_agent_container_timeout(task_config)
    if copy_fallback or run_retry or clean_infer or github_raw_retry or agent_container_timeout:
        index = shell_lines.index('export SWE_WORK_DIR="$WORK_DIR"') + 1
        shim_env = [
            'XTASK_SWE_PYTHONPATH="$SCRIPT_DIR/.xtask_swe_copy_fallback"',
            'export XTASK_SWE_PYTHONPATH',
        ]
        if copy_fallback:
            shim_env.append('export XTASK_SWEBENCH_DOCKER_COPY_FALLBACK=1')
        if run_retry:
            shim_env.append('export XTASK_SWEBENCH_DOCKER_RUN_RETRY=1')
        if clean_infer:
            shim_env.append('export XTASK_SWEBENCH_CLEAN_INFER_CONTAINER=1')
        if github_raw_retry:
            shim_env.append('export XTASK_SWEBENCH_GITHUB_RAW_RETRY=1')
        if agent_container_timeout:
            shim_env.append(
                'export XTASK_SWEBENCH_AGENT_CONTAINER_TIMEOUT='
                + _quote(agent_container_timeout)
            )
        shim_env.extend([
            'export PYTHONDONTWRITEBYTECODE=1',
            'export PYTHONPATH="$XTASK_SWE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"',
        ])
        shell_lines[index:index] = shim_env
    return "\n".join(shell_lines)


def generate_aisbench_swe_script(
    task_config: Dict[str, Any],
    output_file_path: str,
    output_shell_file: str = SHELL_FILENAME,
    home: Optional[str] = None,
) -> str:
    """Write AISBench SWE config + host-side runner script. Return bash command."""
    validate_aisbench_swebench_task(task_config)
    if home:
        task_config = _expand_swebench_home_paths(task_config, home)
    _safe_name(output_shell_file, "output_shell_file")
    os.makedirs(output_file_path, exist_ok=True)
    config_path = os.path.join(output_file_path, CONFIG_FILENAME)
    shell_path = os.path.join(output_file_path, output_shell_file)
    with open(config_path, "w", encoding="utf-8") as stream:
        stream.write(_render_python_config(task_config))
    with open(shell_path, "w", encoding="utf-8") as stream:
        stream.write(_render_shell(task_config))
    if (
        _docker_copy_fallback_enabled(task_config)
        or _docker_run_retry_enabled(task_config)
        or _clean_infer_container_enabled(task_config)
        or _github_raw_retry_enabled(task_config)
        or swebench_agent_container_timeout(task_config)
    ):
        shim_dir = os.path.join(output_file_path, ".xtask_swe_copy_fallback")
        os.makedirs(shim_dir, exist_ok=True)
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), "swebench_docker_copy_fallback.py"),
            os.path.join(shim_dir, "sitecustomize.py"),
        )
    os.chmod(shell_path, 0o755)
    logging.info("SWE-bench AISBench script generated: %s", shell_path)
    return f"bash {shlex.quote(shell_path)}"


def _merge_swebench_runner_outputs(downloaded: Path, case_dir: Path,
                                   output_shell_file: str) -> None:
    """Merge AISBench artifacts without replacing the live NPU log directory.

    The x86 runner was seeded from the case directory, so its snapshot may
    contain old xLLM logs and old benchmark trials. Never overwrite a live or
    already archived ``xllm_log`` with those stale copies.
    """
    if not downloaded.is_dir():
        raise RuntimeError(f"SWE-bench runner download is missing: {downloaded}")

    def ignore_runtime_logs(_directory: str, names: list[str]) -> set[str]:
        return {"xllm_log"} & set(names)

    for source in downloaded.iterdir():
        if source.name in {
            "xllm_log", ".xtask_swe_copy_fallback", "diagnostics", "tmp",
            CONFIG_FILENAME, output_shell_file,
        }:
            continue
        destination = case_dir / source.name
        if source.is_symlink() or destination.is_symlink():
            raise RuntimeError(f"Unsafe SWE-bench runner artifact: {source.name}")
        if source.is_dir():
            # New AISBench trials can be renamed directly; historical trials
            # and the legacy outputs/default tree are merged without deleting
            # any live log files or old trial-specific xllm_log directories.
            if destination.exists():
                shutil.copytree(source, destination, dirs_exist_ok=True,
                                ignore=ignore_runtime_logs)
            else:
                shutil.move(str(source), str(destination))
        elif source.is_file() and not destination.exists():
            shutil.copy2(source, destination)


def _download_swebench_runner_outputs(executor, work_dir: str,
                                      output_file_path: str,
                                      output_shell_file: str) -> None:
    """Stage the remote tree outside the case root before merging results."""
    case_dir = Path(output_file_path)
    with tempfile.TemporaryDirectory(
        prefix=".xtask_swe_download_", dir=str(case_dir.parent)
    ) as temporary_dir:
        staged_dir = Path(temporary_dir) / "downloaded"
        if not executor.download(work_dir, str(staged_dir)):
            raise RuntimeError(f"Failed to download SWE-bench results from {work_dir}")
        _merge_swebench_runner_outputs(staged_dir, case_dir, output_shell_file)


def execute_aisbench_swe_benchmark(
    task_config: Mapping[str, Any],
    output_file_path: str,
    timeout: Optional[int] = None,
    output_shell_file: str = SHELL_FILENAME,
) -> CommandResult:
    """Run the generated SWE script on the configured x86 host.

    Syncs AISBench source from the control node and pip-installs AISBench,
    mini-swe-agent and the SWE-bench harness on the runner first. Docker must
    already be available on that host.

    Local runner: execute in the case output directory.
    Remote runner: upload the case directory, run, then download results.
    Never execs into the xLLM/NPU container.
    """
    validate_aisbench_swebench_task(task_config)
    runner = swebench_runner(task_config)
    host = str(runner.get("host") or "").strip()
    user = runner.get("user")
    try:
        port = int(runner.get("port") or 22)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark.swebench.runner.port must be an integer") from exc
    timeout_s = timeout if timeout is not None else swebench_timeout_s(task_config)
    executor = create_host_executor(host, user=user, port=port)
    from xtask.benchmarks.aisbench.swebench_x86_setup import setup_aisbench_swe_runner

    if is_local_host(host):
        remote_home = os.path.expanduser("~")
    else:
        remote_home, home_error = _remote_home(executor)
        if home_error is not None:
            return home_error

    task_config = _expand_swebench_home_paths(task_config, remote_home)
    runner = swebench_runner(task_config)
    config = swebench_config(task_config)
    setup_config = swebench_setup(task_config)
    explicit_work_dir = str(runner.get("work_dir") or "").strip()
    explicit_install_dir = str(setup_config.get("install_dir") or "").strip()

    if explicit_work_dir:
        work_dir = explicit_work_dir
    elif remote_home:
        case_name = _safe_name(
            task_config.get("case_name") or "swebench", "case_name"
        )
        # Container-generated files may be owned by root. A per-run directory
        # avoids upload collisions with stale artifacts from previous runs.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = f"{remote_home}/xtask_swe/{case_name}_{run_id}"
    else:
        work_dir = output_file_path

    install_dir = explicit_install_dir or f"{remote_home}/xtask_swe/runtime"

    from xtask.benchmarks.aisbench.swebench_x86_container import (
        SweRuntimeContainerState,
        cleanup_swebench_runtime_container,
        ensure_swebench_runtime_container,
    )

    lifecycle = SweRuntimeContainerState()
    try:
        try:
            wrappers = ensure_swebench_runtime_container(
                task_config, executor, remote_home, work_dir, install_dir,
                state=lifecycle,
            )
        except (RuntimeError, ValueError) as exc:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Failed to prepare SWE-bench x86 runtime container: {exc}",
                execution_time=0.0,
            )
        if wrappers is not None:
            python_wrapper, bench_wrapper = wrappers
            swe_config = task_config["benchmark"]["swebench"]
            if "runtime_container" in runner:
                swe_config["runner"]["python"] = python_wrapper
            else:
                swe_config.setdefault("setup", {})["python"] = python_wrapper
            swe_config["ais_bench_bin"] = bench_wrapper

        if not str(config.get("dataset_path") or "").strip():
            from xtask.benchmarks.aisbench.swebench_x86_setup import (
                prepare_default_swebench_dataset,
            )

            try:
                default_dataset_path = prepare_default_swebench_dataset(
                    task_config, executor, host, install_dir
                )
            except (OSError, RuntimeError, ValueError) as exc:
                return CommandResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Failed to prepare the default SWE-bench dataset: {exc}",
                    execution_time=0.0,
                )
            if default_dataset_path:
                task_config["benchmark"]["swebench"]["dataset_path"] = default_dataset_path

        local_script = os.path.join(output_file_path, output_shell_file)
        generate_aisbench_swe_script(
            dict(task_config),
            output_file_path,
            output_shell_file=output_shell_file,
            home=remote_home,
        )

        setup = setup_aisbench_swe_runner(
            task_config,
            executor,
            host,
            work_dir,
            install_dir=install_dir,
        )
        if not setup.command.success:
            return setup.command

        ais_bench_prefix = ""
        if setup.ais_bench_bin:
            ais_bench_prefix = f"AIS_BENCH_BIN={shlex.quote(setup.ais_bench_bin)} "
        if is_local_host(host):
            logging.info("SWE-bench: running AISBench on local x86 host %s", host)
            return executor.run(
                f"{ais_bench_prefix}bash {shlex.quote(local_script)}",
                timeout=timeout_s,
            )

        logging.info("SWE-bench: running AISBench on remote x86 host %s", host)
        if isinstance(executor, SshHostExecutor):
            uploaded = executor.upload_dir_to_exact_path(output_file_path, work_dir)
        else:
            uploaded = executor.upload(output_file_path, work_dir)
        if not uploaded:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Failed to upload SWE-bench work dir to {host}:{work_dir}",
                execution_time=0.0,
            )
        remote_script = os.path.join(work_dir, output_shell_file)
        command = (
            f"{ais_bench_prefix}SWE_WORK_DIR={shlex.quote(work_dir)} "
            f"bash {shlex.quote(remote_script)}"
        )
        result = executor.run(command, timeout=timeout_s)
        try:
            _download_swebench_runner_outputs(
                executor, work_dir, output_file_path, output_shell_file
            )
        except (OSError, RuntimeError) as exc:
            logging.error("SWE-bench: result download/merge failed on %s: %s", host, exc)
            if result.success:
                return CommandResult(
                    success=False,
                    exit_code=-1,
                    stdout=result.stdout,
                    stderr=(result.stderr or "")
                    + f"\nFailed to merge SWE-bench results from {host}:{work_dir}: {exc}",
                    execution_time=result.execution_time,
                )
        return result
    finally:
        if not cleanup_swebench_runtime_container(executor, lifecycle):
            raise RuntimeError(
                f"SWE-bench x86 driver cleanup failed: {lifecycle.name} "
                f"({lifecycle.container_id})"
            )
