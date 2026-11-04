"""Safe archive extraction, legacy scoring and result-file discovery."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
import tempfile
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Tuple

DATE_DIR = re.compile(r"^\d{8}_\d{6}$")
SCORE_PATTERN = re.compile(
    r"Overall WER ->\s*([0-9.]+)\s*%\s+N=(\d+)\s+C=(\d+)\s+S=(\d+)\s+D=(\d+)\s+I=(\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_archive(archive: Path, destination: Path) -> None:
    """Reject traversal, symlinks, duplicates and oversized archive contents."""
    with zipfile.ZipFile(archive) as source:
        seen = set()
        total = 0
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Unsafe ASR archive member: {info.filename}")
            if info.filename in seen:
                raise ValueError(f"Duplicate ASR archive member: {info.filename}")
            seen.add(info.filename)
            total += info.file_size
            if total > 2 * 1024 ** 3:
                raise ValueError("ASR archive exceeds 2 GiB extracted size")
        destination.mkdir(parents=True, exist_ok=True)
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or "__MACOSX" in path.parts or path.name == ".DS_Store":
                continue
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as stream, target.open("wb") as output:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)


def keyed_text(path: Path) -> Dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if fields[0] in result:
            raise ValueError(f"Duplicate sample ID {fields[0]} in {path}")
        result[fields[0]] = fields[1] if len(fields) == 2 else ""
    return result


def score_legacy(script: Path, reference: Path, hypothesis: Path, output: Path) -> Dict[str, Any]:
    refs, hyps = keyed_text(reference), keyed_text(hypothesis)
    if not refs or refs.keys() != hyps.keys():
        raise ValueError("Reference/prediction IDs do not match; legacy scorer must not silently drop samples")
    command = [sys.executable, str(script), "--char=1", "--v=1", str(reference), str(hypothesis)]
    with output.open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.PIPE, text=True, check=True, timeout=300)
    match = SCORE_PATTERN.search(output.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("Legacy ASR score output is missing Overall WER")
    _, n, correct, substitutions, deletions, insertions = match.groups()
    n, correct, substitutions, deletions, insertions = map(int, (n, correct, substitutions, deletions, insertions))
    if n <= 0 or correct + substitutions + deletions != n:
        raise ValueError("Invalid legacy score counts")
    errors = substitutions + deletions + insertions
    return {
        "profile": "joyai_legacy", "unit": "mixed_zh_character_latin_segment",
        "N": n, "C": correct, "S": substitutions, "D": deletions, "I": insertions,
        "error_count": errors, "error_rate_pct": errors / n * 100,
        "command": command, "scorer_sha256": sha256(script),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a summary so interruption cannot leave torn JSON."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def portable_result(value: Any, directory: Path) -> Any:
    """Make artifact references relocatable; preserve commands as executed."""
    if isinstance(value, list):
        return [portable_result(item, directory) for item in value]
    if not isinstance(value, dict):
        return value

    def relative(raw: Any) -> Any:
        if not isinstance(raw, str):
            return raw
        path = Path(raw)
        if path.is_absolute():
            try:
                return path.relative_to(directory).as_posix()
            except ValueError:
                return raw
        return raw

    output = {}
    for key, item in value.items():
        if key == "files" and isinstance(item, dict):
            output[key] = {name: relative(path) for name, path in item.items()}
        elif key in {"config_file", "result_file"}:
            output[key] = relative(item)
        else:
            output[key] = portable_result(item, directory)
    return output


def redact(value: Any) -> Any:
    """Do not archive API credentials from task/client configuration."""
    if isinstance(value, dict):
        return {
            key: "***" if any(token in str(key).lower() for token in (
                "api_key", "password", "secret", "authorization"
            )) else redact(item)
            for key, item in value.items() if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def archive_execution(
    directory: Path, case_root: Path, task_config: Dict[str, Any],
    result: Dict[str, Any], source_yaml: str = "",
) -> None:
    """Archive this invocation, not a later run's case-root files."""
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "task.yaml").exists():
        (directory / "task.yaml").write_text(
            yaml.safe_dump(redact(task_config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    if source_yaml and Path(source_yaml).is_file() and not (directory / "source_task.yaml").exists():
        source = yaml.safe_load(Path(source_yaml).read_text(encoding="utf-8"))
        (directory / "source_task.yaml").write_text(
            yaml.safe_dump(redact(source), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    script = case_root / "run_asr_test.sh"
    if script.is_file() and not (directory / script.name).exists():
        (directory / script.name).write_bytes(script.read_bytes())
    config = case_root / "asr_inputs" / "runner_config.json"
    if config.is_file() and not (directory / "runner_config.json").exists():
        write_json(directory / "runner_config.json", redact(json.loads(config.read_text(encoding="utf-8"))))
    if not (directory / "commands.json").exists():
        write_json(directory / "commands.json", {
            "benchmark": result.get("benchmark_command", ""),
            "startup": result.get("startup_command", ""),
        })
    for field, filename in (
        ("benchmark_stdout", "benchmark_stdout.log"),
        ("benchmark_stderr", "benchmark_stderr.log"),
    ):
        # Keep an already captured stream when a later finalize call lacks it.
        if field in result or not (directory / filename).exists():
            (directory / filename).write_text(str(result.get(field) or ""), encoding="utf-8")
    fields = (
        "success", "message", "scheduler_error", "benchmark_success", "benchmark_exit_code",
        "execution_time", "benchmark_execution_time", "container_id", "external_service",
        "artifact_host", "artifact_path", "service_endpoint", "asr_gate_issues", "archive_error", "report_generated",
    )
    summary = {key: result[key] for key in fields if key in result}
    summary["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    summary.update(schema_version=1, run_id=directory.name)
    write_json(directory / "execution_result.json", summary)
    payload = result.get("benchmark_asr_result")
    if isinstance(payload, dict):
        write_run_reports(directory, payload)


def _resolve_result_paths(value: Any, directory: Path) -> Any:
    """Expose host paths in memory without writing them into the archive."""
    if isinstance(value, list):
        return [_resolve_result_paths(item, directory) for item in value]
    if not isinstance(value, dict):
        return value

    def resolve(raw: Any) -> Any:
        if not isinstance(raw, str):
            return raw
        path = Path(raw)
        if not path.is_absolute():
            if ".." in path.parts:
                raise ValueError(f"Unsafe ASR artifact reference: {raw}")
            return str(directory / path)
        # Older runs recorded /workspace paths. Rebase only their dated root.
        if directory.name in path.parts:
            index = path.parts.index(directory.name)
            return str(directory.joinpath(*path.parts[index + 1:]))
        return raw

    output = {}
    for key, item in value.items():
        if key == "files" and isinstance(item, dict):
            output[key] = {name: resolve(path) for name, path in item.items()}
        elif key in {"config_file", "result_file"}:
            output[key] = resolve(item)
        else:
            output[key] = _resolve_result_paths(item, directory)
    return output


def asr_result_post_process(
    task_config: Dict[str, Any], output_path: str, init_entries: Iterable[str],
) -> Tuple[Dict[str, Any], str]:
    root = Path(output_path)
    previous = set(init_entries or [])
    directories = sorted(p for p in root.iterdir() if p.is_dir() and p.name not in previous and DATE_DIR.fullmatch(p.name))
    if len(directories) != 1:
        raise FileNotFoundError(f"Expected one new ASR date directory, found {len(directories)}")
    directory = directories[0]
    path = directory / "asr_result.json"
    damaged = None
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            damaged = f"Invalid ASR result JSON: {exc}"
        else:
            if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                    or not isinstance(payload.get("runs", []), list)):
                damaged = "Invalid ASR result schema"
            elif any(not isinstance(run, dict)
                     or not isinstance(run.get("metrics", {}), dict)
                     or (run.get("score") is not None and not isinstance(run["score"], dict))
                     for run in payload.get("runs", [])):
                damaged = "Invalid ASR per-run result schema"
        if damaged:
            # Preserve original bytes for diagnostics, do not overwrite evidence.
            backup = directory / "asr_result.corrupt.json"
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
    if not path.is_file() or damaged:
        payload = {
            "schema_version": 1, "backend": "legacy_infer",
            "scoring_profile": "joyai_legacy", "success": False, "runs": [],
            "execution_error": damaged or "ASR execution ended before its result summary was written",
            "config_file": "config.json",
        }
    if damaged or payload.get("state") == "running" or not path.is_file():
        payload["success"] = False
        payload["state"] = "interrupted"
        payload["execution_error"] = damaged or "ASR execution interrupted before final result/scoring"
        runs = payload.get("runs")
        runs = runs if isinstance(runs, list) else []
        by_id = {run["run_id"]: run for run in runs if isinstance(run, dict) and isinstance(run.get("run_id"), str)}
        for run_dir in sorted(directory.iterdir()):
            match = re.fullmatch(r"c(\d+)_r(\d+)", run_dir.name)
            if not run_dir.is_dir() or not match:
                continue
            run = by_id.setdefault(run_dir.name, {
                "run_id": run_dir.name, "concurrency": int(match[1]), "repeat": int(match[2]),
                "complete": False, "score": None,
            })
            if run.get("complete") is True:
                continue
            # A completed repeat can survive a torn aggregate write. Keep its
            # own recorded result, but the interrupted campaign remains failed.
            summary_path = run_dir / "summary.json"
            if summary_path.is_file():
                try:
                    completed = json.loads(summary_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    completed = None
                if isinstance(completed, dict) and completed.get("run_id") == run_dir.name and completed.get("complete") is True:
                    # Per-run file references are relative to the per-run root.
                    completed = _resolve_result_paths(completed, run_dir)
                    by_id[run_dir.name] = portable_result(completed, directory)
                    continue
            records = []
            requests = run_dir / "requests.jsonl"
            if requests.is_file():
                with requests.open(encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # A last partial write is not completed.
                        if isinstance(record, dict):
                            records.append(record)
            run.update(
                completed_requests=len(records), complete=False,
                failed_requests=sum(record.get("status") != "ok" for record in records),
                execution_error="Run interrupted before completion and final scoring",
                files={"requests.jsonl": str(requests.relative_to(directory))} if requests.is_file() else {},
            )
        payload["runs"] = list(by_id.values())
        write_json(path, portable_result(payload, directory))
    from xtask.benchmarks.asr.reporting import summarize_runs
    payload["summary_by_concurrency"] = summarize_runs(payload.get("runs"))
    payload["result_file"] = "asr_result.json"
    return _resolve_result_paths(payload, directory), str(directory)


def write_run_reports(directory: Path, payload: Dict[str, Any]) -> None:
    """Small tabular per-run reports and valid-repeat statistics for each load."""
    from xtask.benchmarks.asr.reporting import finite_metrics, summarize_runs
    rows = []
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        row = {name: run.get(name) for name in (
            "run_id", "concurrency", "repeat", "complete", "expected_requests",
            "completed_requests", "failed_requests", "execution_error",
        )}
        score = run.get("score") or {}
        if isinstance(score, dict):
            row.update({name: score.get(name) for name in ("N", "C", "S", "D", "I", "error_count", "error_rate_pct")})
        row.update(finite_metrics(run.get("metrics")))
        rows.append(row)
    headers = list(dict.fromkeys(key for row in rows for key in row)) or ["run_id", "complete"]
    with (directory / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "summary_by_concurrency.json", summarize_runs(payload.get("runs")))
