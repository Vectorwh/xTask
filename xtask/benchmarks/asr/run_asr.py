"""Run the unmodified legacy client with explicit configuration and audit hooks.

The original infer.sh is not executed: it installs dependencies at runtime,
uses fixed paths and overwrites results.  Its Python request implementation,
context processing and compute-wer.py are retained unchanged from infer.zip.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

# Support both package import and direct invocation from a generated script.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from xtask.benchmarks.asr.artifacts import extract_archive, portable_result, score_legacy, sha256, write_json, write_run_reports
from xtask.benchmarks.asr.config import normalize_config, result_issues


def percentile(values: List[float], fraction: float) -> Optional[float]:
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def summarize(records: List[Dict[str, Any]], seconds: float, exclude_first: int) -> Dict[str, Any]:
    measured = [r for r in records if r["idx"] >= exclude_first]
    latency = [r["elapsed_ms"] for r in measured if r["elapsed_ms"] is not None]
    successful = [r for r in measured if r["status"] == "ok"]
    first_text = [r["first_text_ms"] for r in successful if r["first_text_ms"] is not None]
    legacy_ttft = [r["legacy_ttft_ms"] for r in measured if r["legacy_ttft_ms"] is not None]
    durations = sum(r["audio_duration_s"] for r in measured if r["audio_duration_s"] > 0)
    successful_seconds = sum(r["audio_duration_s"] for r in successful if r["audio_duration_s"] > 0)
    # The wall-clock window spans the first measured request to the last
    # measured completion; dropped warmup rows are not left in its denominator.
    window = max((r["finished_at_s"] for r in measured), default=0) - min(
        (r["started_at_s"] for r in measured), default=0
    )
    failures = sum(r["status"] != "ok" for r in records)
    return {
        "e2el_avg_ms": sum(latency) / len(latency) if latency else None,
        "e2el_p90_ms": percentile(latency, 0.9), "e2el_p99_ms": percentile(latency, 0.99),
        "first_text_avg_ms": sum(first_text) / len(first_text) if first_text else None,
        "first_text_p99_ms": percentile(first_text, 0.99),
        "legacy_ttft_avg_ms": sum(legacy_ttft) / len(legacy_ttft) if legacy_ttft else None,
        "legacy_ttft_p99_ms": percentile(legacy_ttft, 0.99),
        "legacy_rtf": sum(latency) / 1000 / durations if latency and durations > 0 else None,
        "request_throughput": len(successful) / window if window > 0 else None,
        "audio_throughput": successful_seconds / window if window > 0 else None,
        "failed_requests": failures, "failure_rate": failures / len(records) if records else 1.0,
        "measured_requests": len(measured), "measured_successful_requests": len(successful),
        "measurement_wall_seconds": max(0.0, window), "batch_wall_seconds": seconds,
        "audio_duration_seconds": durations, "successful_audio_duration_seconds": successful_seconds,
    }


def load_client(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("xtask_asr_legacy_client", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original ASR client: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "***" if any(word in k.lower() for word in ("api_key", "password", "secret", "authorization")) else _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class ObservedResponse:
    """Keep original SSE parsing, while detecting incomplete/error streams."""

    def __init__(self, response: Any, local: threading.local) -> None:
        self.response, self.local = response, local

    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)

    def iter_lines(self) -> Iterator[str]:
        for line in self.response.iter_lines():
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    self.local.finished = True
                else:
                    try:
                        value = json.loads(payload)
                    except json.JSONDecodeError:
                        self.local.stream_error = "Invalid SSE JSON"
                    else:
                        if not isinstance(value, dict):
                            self.local.stream_error = "SSE event must be an object"
                        elif value.get("error"):
                            self.local.stream_error = str(value["error"])
                        else:
                            choices = value.get("choices") or []
                            if choices:
                                choice = choices[0]
                                if choice.get("finish_reason") is not None:
                                    self.local.finished = True
                                    if choice["finish_reason"] not in {"stop", "eos"}:
                                        self.local.stream_error = f"Abnormal SSE finish_reason: {choice['finish_reason']}"
                                content = (choice.get("delta") or {}).get("content")
                                if content and self.local.first_text_ms is None:
                                    self.local.first_text_ms = (time.monotonic() - self.local.started) * 1000
            yield line


class ObservedStream:
    def __init__(self, stream: Any, local: threading.local) -> None:
        self.stream, self.local = stream, local

    def __enter__(self) -> ObservedResponse:
        return ObservedResponse(self.stream.__enter__(), self.local)

    def __exit__(self, *args: Any) -> Any:
        return self.stream.__exit__(*args)


class ObservedClient:
    def __init__(self, client: Any, local: threading.local) -> None:
        self.client, self.local = client, local

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def stream(self, *args: Any, **kwargs: Any) -> ObservedStream:
        self.local.started = time.monotonic()
        return ObservedStream(self.client.stream(*args, **kwargs), self.local)


def _args(config: Dict[str, Any], dataset: Path, output: Path, concurrency: int) -> argparse.Namespace:
    generation = config["generation"]
    return argparse.Namespace(
        api=config["api"], server_addr=config["server_addr"], model=config.get("model"),
        data_dir=str(dataset), data_root="", output_dir=str(output), num_tasks=concurrency,
        repetition_penalty=generation["repetition_penalty"], hotwords=generation.get("hotwords"),
        use_context=config["use_context"], warmup=config["warmup"]["exclude_first"],
        stream=config["stream"], simulate_realtime=config["simulate_realtime"], timeout=config["timeout"],
        language=generation["language"], temperature=generation["temperature"],
        top_p=generation.get("top_p"), top_k=generation.get("top_k"),
        max_completion_tokens=generation.get("max_completion_tokens"), stream_include_usage=False,
        chat_prompt=generation.get("chat_prompt"), max_context_chars=generation["max_context_chars"],
        max_tokens=generation["max_tokens"], verbose=False,
    )


def prepare_items(client: Any, dataset: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = client.load_dataset(dataset)
    if len(items) != config["expected_samples"]:
        raise ValueError(f"Expected {config['expected_samples']} dataset samples, found {len(items)}")
    if len({item["key"] for item in items}) != len(items):
        raise ValueError("Duplicate ASR dataset sample IDs")
    # References belong exclusively to the scorer, never to request payloads.
    for item in items:
        if not isinstance(item["key"], str) or not item["key"] or any(c.isspace() for c in item["key"]):
            raise ValueError("ASR sample IDs must be nonempty strings without whitespace")
        if not isinstance(item["target"], str) or not item["target"].strip():
            raise ValueError(f"Missing reference for sample {item['key']}")
        if any(c in item["target"] for c in ("\n", "\r", "\t")):
            raise ValueError(f"Reference must occupy one line: {item['key']}")
        audio = Path(item["source"]).resolve()
        try:
            audio.relative_to(dataset.resolve())
        except ValueError:
            raise ValueError(f"Audio path is outside extracted dataset: {audio}") from None
        if not audio.is_file():
            raise ValueError(f"Missing audio for sample {item['key']}")
        if not isinstance(item.get("context", ""), str) or not isinstance(item.get("hotwords", ""), str):
            raise ValueError(f"Context/hotwords must be strings: {item['key']}")
        duration = client.get_audio_duration(str(audio))
        if duration <= 0 or duration > config["max_audio_seconds"]:
            raise ValueError(f"Invalid/out-of-range audio duration {duration}: {item['key']}")
        item["audio_duration_s"] = duration
        item["audio_sha256"] = sha256(audio)
    if config["limit"]:
        items = items[:config["limit"]]
    if config["warmup"]["exclude_first"] >= len(items):
        raise ValueError("warmup.exclude_first leaves no measured samples")
    return items


def perform_warmup(client: Any, dataset: Path, client_dir: Path, config: Dict[str, Any]) -> None:
    count = config["warmup"]["synthetic_requests"]
    if not count:
        return
    import httpx
    args = _args(config, dataset, client_dir, 1)
    args.api, args.stream, args.use_context = config["warmup"]["api"], False, True
    args.hotwords = None
    args.timeout = min(config["timeout"], 120)
    with httpx.Client(trust_env=False, headers={"Authorization": f"Bearer {config.get('api_key') or 'EMPTY'}"}) as transport:
        response = transport.get(config["server_addr"] + "/models", timeout=10)
        response.raise_for_status()
        model = config.get("model") or response.json()["data"][0]["id"]
        for name in ("w2", "w8", "w20")[:count]:
            item = {"key": "warmup_" + name, "source": str(client_dir / f"warmup_{name}.wav"),
                    "target": "", "context": "十六今年十六岁，身高只有158cm，是你从小一起长大的妹妹。", "hotwords": ""}
            result = client._dispatch_transcribe(transport, config["server_addr"], model, item, args, item["key"])
            if result[2]:
                raise RuntimeError(f"ASR warmup failed: {result[2]}")


def run_one(client_dir: Path, dataset: Path, output: Path, config: Dict[str, Any], concurrency: int, repeat: int) -> Dict[str, Any]:
    client = load_client(client_dir / "infer_dataset.py")
    items = prepare_items(client, dataset, config)
    write_json(output / "samples.json", [{k: item[k] for k in ("key", "source", "audio_duration_s", "audio_sha256")} for item in items])
    local, lock = threading.local(), threading.Lock()
    records = []
    transports = []
    original_dispatch, original_form, original_chat = client._dispatch_transcribe, client._build_form, client._chat_body

    def build_form(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        form = original_form(*args, **kwargs)
        local.request = {"endpoint": "audio/transcriptions", "form": form}
        return form

    def build_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        body = original_chat(*args, **kwargs)
        # Omit base64 waveforms while retaining exact model prompts/parameters.
        audit = {k: v for k, v in body.items() if k != "messages"}
        audit["messages"] = [{"role": message["role"], "content": [
            {"type": "audio_url", "audio_url": {"url": "<audio: see sample SHA256>"}} if part["type"] == "audio_url" else part
            for part in message["content"]
        ]} for message in body["messages"]]
        local.request = {"endpoint": "chat/completions", "body": audit}
        return body

    def dispatch(transport: Any, base_url: str, model: str, item: Dict[str, Any], args: Any, request_id: str):
        local.started = time.monotonic()
        started_at = local.started
        local.first_text_ms, local.finished, local.stream_error, local.request = None, False, None, None
        result = original_dispatch(transport, base_url, model, item, args, request_id)
        finished_at = time.monotonic()
        key, text, error, stats, duration = result
        if args.stream and not error:
            error = local.stream_error or (None if local.finished else "SSE stream ended without completion marker")
        if not error and not text.strip():
            error = "Empty transcript for a nonempty-reference sample"
        stats = stats or client._empty_stats((finished_at - started_at) * 1000, request_id)
        record = {
            "idx": int(request_id.rsplit("_", 1)[1]), "sample_id": key, "request_id": request_id,
            "reference": item["target"], "prediction_raw": text, "status": "failed" if error else "ok",
            "error": error, "audio_sha256": item["audio_sha256"], "audio_duration_s": duration,
            "elapsed_ms": stats["elapsed_ms"], "legacy_ttft_ms": stats["ttft_ms"] if stats.get("ttft_ok") else None,
            "first_text_ms": local.first_text_ms if args.stream else None,
            "n_chunks": stats["n_chunks"], "chunk_times_ms": stats["chunk_times_ms"],
            "started_at_s": started_at, "finished_at_s": finished_at,
            "request": _redact(local.request), "attempts": 1,
        }
        with lock:
            records.append(record)
            with (output / "requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return key, text, error, stats, duration

    import httpx
    import urllib.request
    import urllib.error

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def urlopen(request: Any, **kwargs: Any) -> Any:
        request.add_header("Authorization", f"Bearer {config.get('api_key') or 'EMPTY'}")
        return opener.open(request, **kwargs)

    # Override only this imported client's urllib surface, not process globals.
    client.urllib = SimpleNamespace(
        request=SimpleNamespace(Request=urllib.request.Request, urlopen=urlopen),
        error=urllib.error,
    )
    client.load_dataset = lambda *args, **kwargs: items
    client._build_form, client._chat_body, client._dispatch_transcribe = build_form, build_chat, dispatch
    def transport(*args: Any, **kwargs: Any) -> ObservedClient:
        value = ObservedClient(
            httpx.Client(trust_env=False, timeout=None, headers={"Authorization": f"Bearer {config.get('api_key') or 'EMPTY'}"}), local
        )
        transports.append(value)
        return value

    client._no_proxy_transport = transport
    t0 = time.monotonic()
    execution_error = None
    try:
        client.main(_args(config, dataset, output, concurrency))
    except (Exception, SystemExit) as exc:
        # Required subprocess/client boundary: keep partial evidence and fail.
        execution_error = f"Legacy ASR client failed: {exc}"
    finally:
        for transport in transports:
            transport.close()
    elapsed = time.monotonic() - t0
    records.sort(key=lambda record: record["idx"])
    expected = {item["key"] for item in items}
    complete = len(records) == len(items) and {r["sample_id"] for r in records} == expected
    score = None
    if complete and not execution_error:
        try:
            score = score_legacy(client_dir / "wer_tools/compute-wer.py", output / "text.ref", output / "text.hyp", output / "wer")
        except Exception as exc:
            execution_error = f"Legacy ASR scoring failed: {exc}"
    metrics = summarize(records, elapsed, config["warmup"]["exclude_first"])
    if score:
        metrics.update(error_count=score["error_count"], error_rate_pct=score["error_rate_pct"])
    result = {
        "run_id": output.name, "concurrency": concurrency, "repeat": repeat,
        "expected_requests": len(items), "completed_requests": len(records), "complete": complete,
        "failed_requests": metrics["failed_requests"], "execution_error": execution_error,
        "score": score, "metrics": metrics,
        "files": {name: str(output / name) for name in ("requests.jsonl", "samples.json", "text.hyp", "text.ref", "err.log", "wer", "latency.tsv", "latency_summary.txt")},
    }
    write_json(output / "summary.json", portable_result(result, output))
    return result


def run(config: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    config = normalize_config(config)
    if not config.get("server_addr"):
        raise ValueError("ASR server_addr is required")
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now()
    while True:
        directory = output_root / stamp.strftime("%Y%m%d_%H%M%S")
        try:
            directory.mkdir()
            break
        except FileExistsError:
            stamp += timedelta(seconds=1)
    snapshot = _redact(config)
    snapshot["metric_contract"] = {
        "percentile": "linear interpolation", "legacy_rtf": "sum request latency / sum audio duration",
        "throughput": "successful measured requests/audio seconds divided by measured wall-clock window",
        "first_text": "first nonempty SSE content; unavailable for synchronous responses",
        "legacy_ttft": "original first choices-bearing chunk; not necessarily first token",
        "failure_policy": "no retries; failed predictions are empty in legacy score; any failure fails completeness gate",
        "cache_condition": "one service lifetime; runner never restarts the server between concurrency/repeat runs",
        "request_latency": "original client's boundary (chat excludes audio/base64 preparation; multipart includes upload)",
    }
    snapshot["runtime"] = {"python": platform.python_version(), "platform": platform.platform(), "argv": list(sys.argv)}
    try:
        import httpx
        snapshot["runtime"]["httpx"] = httpx.__version__
    except ImportError:
        # Dependency validation is reported in asr_result.json below.
        pass
    write_json(directory / "config.json", snapshot)
    write_json(directory / "runner_config.json", _redact(config))
    script = output_root / "run_asr_test.sh"
    if script.is_file():
        (directory / script.name).write_bytes(script.read_bytes())
    payload = {"schema_version": 1, "backend": "legacy_infer", "scoring_profile": "joyai_legacy", "runs": [], "execution_error": None, "state": "running"}
    payload["config_file"] = str(directory / "config.json")
    write_json(directory / "asr_result.json", portable_result(payload, directory))
    try:
        assets = directory / "assets"
        infer_archive, data_archive = Path(config["infer_archive"]), Path(config["dataset_archive"])
        snapshot["archive_sha256"] = {"infer": sha256(infer_archive), "dataset": sha256(data_archive)}
        write_json(directory / "config.json", snapshot)
        extract_archive(infer_archive, assets)
        extract_archive(data_archive, assets)
        client_dir = assets / "infer"
        dataset = assets / "data" / config["dataset_name"]
        client = load_client(client_dir / "infer_dataset.py")
        prepare_items(client, dataset, config)
        t0 = time.monotonic()
        perform_warmup(client, dataset, client_dir, config)
        snapshot["warmup_wall_seconds"] = time.monotonic() - t0
        write_json(directory / "config.json", snapshot)
        for concurrency in config["concurrency"]:
            for repeat in range(1, config["repeats"] + 1):
                run_dir = directory / f"c{concurrency}_r{repeat}"
                run_dir.mkdir()
                expected = min(config["expected_samples"], config["limit"]) if config["limit"] else config["expected_samples"]
                payload["runs"].append({
                    "run_id": run_dir.name, "concurrency": concurrency, "repeat": repeat,
                    "expected_requests": expected, "completed_requests": 0,
                    "complete": False, "failed_requests": 0, "score": None,
                })
                write_json(directory / "asr_result.json", portable_result(payload, directory))
                payload["runs"][-1] = run_one(client_dir, dataset, run_dir, config, concurrency, repeat)
                write_json(directory / "asr_result.json", portable_result(payload, directory))
    except (Exception, SystemExit) as exc:
        payload["execution_error"] = str(exc)
    from xtask.benchmarks.asr.reporting import summarize_runs
    payload["state"] = "completed"
    payload["summary_by_concurrency"] = summarize_runs(payload["runs"])
    payload["gate_issues"] = result_issues(payload, {"benchmark": {"asr": config}})
    payload["success"] = not payload["gate_issues"]
    payload["config_file"] = str(directory / "config.json")
    write_json(directory / "asr_result.json", portable_result(payload, directory))
    write_run_reports(directory, payload)
    print(f"ASR result: {directory / 'asr_result.json'}; success={payload['success']}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(json.loads(args.config.read_text(encoding="utf-8")), args.output_dir)["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
