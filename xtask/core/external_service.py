"""External already-running service mode for xTask.

Default is off. When enabled, xTask does not start xLLM / xllm_service / etcd
and only prepares datasets, a client (or VBench) container, benchmark, and
results against a caller-provided IP:port or API URL.

This module must not import the scheduler. Callers gate every behavior on
:func:`is_external_service_enabled` so existing E2E YAML is unchanged.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse


_FORBIDDEN_ASSESS_TOKENS = (
    "serving_extensiable_param",
    "serving_extensible_param",
    "device_ids",
    "pd_instances",
    "service_instances",
    "xllm_service",
    "pd_control",
    "service_control",
    "assess_topologies",
    "deployment_topology",
    "local_port",
)

_TOPOLOGY_DEPLOYMENT_KEYS = (
    "instances",
    "pd_instances",
    "service_instances",
    "xllm_service",
    "pd_control",
    "service_control",
)


@dataclass(frozen=True)
class ExternalServiceLocation:
    """Parsed live-service address used by health probes and bench scripts."""

    host: str
    port: int
    scheme: str
    path: str
    origin: str
    url: str
    health_path: Optional[str]
    api_key: Optional[str] = None
    api_path: str = "/v1"

    @property
    def base_path(self) -> str:
        """URL path without a trailing slash; empty when the endpoint is origin-only."""
        return self.path.rstrip("/") if self.path else ""


def is_external_service_enabled(task_config: Optional[Dict[str, Any]]) -> bool:
    """Return True only when the YAML switch is explicitly on.

    Missing / false / unexpected types are off, so historical cases stay E2E.
    """
    if not isinstance(task_config, dict):
        return False
    value = task_config.get("external_service")
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, dict):
        if "enabled" in value:
            return bool(value["enabled"])
        return bool(
            value.get("endpoint")
            or (value.get("host") and value.get("port") is not None)
        )
    return False


def is_vbench_evaluate_only(task_config: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(task_config, dict):
        return False
    benchmark = task_config.get("benchmark") or {}
    if str(benchmark.get("benchmark_type", "")).lower() != "vbench":
        return False
    vbench = benchmark.get("vbench") or {}
    if not isinstance(vbench, dict):
        return False
    return str(vbench.get("mode", "generate_and_evaluate")).lower() == "evaluate"


def requires_external_endpoint(task_config: Dict[str, Any]) -> bool:
    if not is_external_service_enabled(task_config):
        return False
    if is_vbench_evaluate_only(task_config):
        return False
    from xtask.benchmarks.aisbench.swebench import is_swebench_eval_only

    return not is_swebench_eval_only(task_config)


def _external_block(task_config: Dict[str, Any]) -> Dict[str, Any]:
    value = task_config.get("external_service")
    return value if isinstance(value, dict) else {}


def _raw_endpoint_spec(task_config: Dict[str, Any]) -> str:
    block = _external_block(task_config)
    endpoint = block.get("endpoint")
    if endpoint:
        return str(endpoint).strip()
    if block.get("host") and block.get("port") is not None:
        return f"http://{block['host']}:{block['port']}"
    deployment = task_config.get("deployment") or {}
    bench = deployment.get("benchmark") if isinstance(deployment, dict) else None
    if isinstance(bench, dict) and bench.get("endpoint"):
        return str(bench["endpoint"]).strip()
    host = task_config.get("host_addr")
    port = task_config.get("port")
    if host and port is not None:
        return f"http://{host}:{port}"
    return ""


def _normalize_health_path(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    path = str(value)
    return path if path.startswith("/") else f"/{path}"


def resolve_health_path(task_config: Dict[str, Any]) -> Optional[str]:
    """Health URL suffix. None means skip the live probe."""
    block = _external_block(task_config)
    if "health_path" in block:
        return _normalize_health_path(block.get("health_path"))
    deployment = task_config.get("deployment") or {}
    bench = deployment.get("benchmark") if isinstance(deployment, dict) else None
    if isinstance(bench, dict) and "health_path" in bench:
        return _normalize_health_path(bench.get("health_path"))
    return "/health"


def _parse_endpoint(raw: str) -> ExternalServiceLocation:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("external_service endpoint is empty")
    if "://" not in text:
        text = f"http://{text}"
    parsed = urlparse(text)
    host = parsed.hostname
    if not host:
        raise ValueError(f"external_service endpoint has no host: {raw!r}")
    scheme = parsed.scheme or "http"
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    path = parsed.path or ""
    if path == "/":
        path = ""
    origin = f"{scheme}://{host}:{int(port)}"
    return ExternalServiceLocation(
        host=str(host),
        port=int(port),
        scheme=scheme,
        path=path,
        origin=origin,
        url=origin + path,
        health_path="/health",
    )


def resolve_external_location(task_config: Dict[str, Any]) -> ExternalServiceLocation:
    raw = _raw_endpoint_spec(task_config)
    if not raw:
        raise ValueError(
            "external_service is enabled but no endpoint was provided; "
            "set external_service.endpoint, deployment.benchmark.endpoint, "
            "or host_addr+port"
        )
    parsed = _parse_endpoint(raw)
    block = _external_block(task_config)
    api_key = block.get("api_key")
    api_path = block.get("api_path") or "/v1"
    if isinstance(api_path, str) and api_path and not api_path.startswith("/"):
        api_path = f"/{api_path}"
    return ExternalServiceLocation(
        host=parsed.host,
        port=parsed.port,
        scheme=parsed.scheme,
        path=parsed.path,
        origin=parsed.origin,
        url=parsed.url,
        health_path=resolve_health_path(task_config),
        api_key=str(api_key) if api_key not in (None, "") else None,
        api_path=str(api_path),
    )


def resolve_external_origin(task_config: Dict[str, Any]) -> str:
    """Scheme+host+port used for health checks (path is not included)."""
    return resolve_external_location(task_config).origin


def resolve_openai_api_base(task_config: Dict[str, Any]) -> str:
    location = resolve_external_location(task_config)
    return location.origin + location.base_path


def _aisbench_url_required(location: ExternalServiceLocation) -> bool:
    """Whether AISBench's ``host_ip`` field can express this endpoint.

    AISBench validates ``host_ip`` with ``ipaddress.ip_address()`` and only
    whitelists ``localhost``, so a hostname / https / sub-path endpoint cannot
    be expressed through ``host_ip``. Those must go through AISBench's ``url``
    field instead (which makes AISBench ignore host_ip/host_port).
    """
    if location.base_path or location.scheme != "http":
        return True
    try:
        ipaddress.ip_address(location.host)
    except ValueError:
        return True
    return False


def resolve_aisbench_url_override(task_config: Dict[str, Any]) -> str:
    """Return the AISBench ``url`` value, or "" to keep using ``host_ip``.

    Returns "" (historical behaviour, byte-identical scripts) whenever the
    endpoint is expressible as ``host_ip`` + ``host_port``: plain IPv4/IPv6
    over http with no path. Empty means "keep writing host_ip".

    Non-empty values are always forced to end with "/": AISBench joins the
    endpoint with ``urllib.parse.urljoin(base_url, "v1/chat/completions")``,
    which *replaces* the last path segment when the base has no trailing
    slash (``https://gw/a/b`` -> ``https://gw/a/v1/...``), silently dropping
    a gateway sub-path.
    """
    if not is_external_service_enabled(task_config):
        return ""
    location = resolve_external_location(task_config)
    if not _aisbench_url_required(location):
        return ""
    return location.origin + location.base_path + "/"


def topology_fields_ignored(task_config: Dict[str, Any]) -> List[str]:
    deployment = task_config.get("deployment") or {}
    if not isinstance(deployment, dict):
        return []
    ignored = []
    for key in _TOPOLOGY_DEPLOYMENT_KEYS:
        if deployment.get(key):
            ignored.append(f"deployment.{key}")
    return ignored


def _iter_assess_paths(mapping: Any, prefix: str = "") -> Iterable[str]:
    if not isinstance(mapping, dict):
        return
    for raw_key, raw_value in mapping.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(raw_value, dict):
            yield from _iter_assess_paths(raw_value, path)
        else:
            yield path


def validate_external_service_assess_params(task_config: Dict[str, Any]) -> None:
    if not is_external_service_enabled(task_config):
        return
    if task_config.get("assess_topologies"):
        raise ValueError(
            "external_service cannot search PD topologies "
            "(assess_topologies); the live service is not started by xTask"
        )
    assess = task_config.get("assess_test_param") or {}
    for path in _iter_assess_paths(assess):
        lowered = path.lower()
        for token in _FORBIDDEN_ASSESS_TOKENS:
            if token in lowered:
                raise ValueError(
                    "external_service does not support searching serving or "
                    f"deployment parameter {path!r}; only benchmark request "
                    "parameters (for example batch_size / num_prompts) can be searched"
                )


def validate_external_service_config(task_config: Dict[str, Any]) -> None:
    """Fail closed when the switch is on but the live target cannot be used."""
    if not is_external_service_enabled(task_config):
        return
    validate_external_service_assess_params(task_config)
    if not requires_external_endpoint(task_config):
        return
    resolve_external_location(task_config)


def apply_external_service_to_task_config(task_config: Dict[str, Any]) -> None:
    """Fill host_addr/port (and VBench endpoint) from the live target.

    No-op when the switch is off, so E2E generators keep reading topology.
    """
    if not is_external_service_enabled(task_config):
        return
    if is_vbench_evaluate_only(task_config) and not _raw_endpoint_spec(task_config):
        return
    from xtask.benchmarks.aisbench.swebench import is_swebench_eval_only

    if is_swebench_eval_only(task_config) and not _raw_endpoint_spec(task_config):
        return
    location = resolve_external_location(task_config)
    task_config["host_addr"] = location.host
    task_config["port"] = location.port
    benchmark = task_config.get("benchmark")
    if not isinstance(benchmark, dict):
        return
    if str(benchmark.get("benchmark_type", "")).lower() != "vbench":
        return
    vbench = benchmark.get("vbench")
    if not isinstance(vbench, dict):
        vbench = {}
        benchmark["vbench"] = vbench
    if vbench.get("endpoint"):
        return
    if location.path:
        vbench["endpoint"] = location.url
