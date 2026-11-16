"""Canonical xLLM-Service configuration, independent of Engine topology.

Legacy keys are normalized without changing Engine topology. Split Service
plans bind Coordinator/Router processes to existing Engine containers and keep
registration addresses separate from the benchmark's Router endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional, Sequence


XLLM_SERVICE_CONFIG_KEY = "xllm_service"
LEGACY_XLLM_SERVICE_CONFIG_KEYS = ("pd_control", "service_control")


def normalize_xllm_service_config(task_config: Dict[str, Any]) -> Dict[str, Any]:
    """Move one legacy control block to ``deployment.xllm_service`` in place.

    Multiple spellings are rejected even when their values match: there must be
    one source of truth. Keeping the mapping itself preserves YAML aliases and
    caller updates, and removing the old key makes normalization idempotent.
    This helper intentionally does not expand Engine instance lists.
    """
    if not isinstance(task_config, dict):
        raise TypeError("task_config must be a mapping")
    deployment = task_config.get("deployment")
    if not isinstance(deployment, dict):
        return task_config

    keys = [
        key
        for key in (XLLM_SERVICE_CONFIG_KEY,) + LEGACY_XLLM_SERVICE_CONFIG_KEYS
        if key in deployment
    ]
    if len(keys) > 1:
        raise ValueError(
            "Conflicting xLLM-Service configuration blocks: "
            + ", ".join(f"deployment.{key}" for key in keys)
            + "; use only deployment.xllm_service"
        )
    if not keys:
        return task_config

    key = keys[0]
    control = deployment[key]
    if control is not None and not isinstance(control, dict):
        raise TypeError(f"deployment.{key} must be a mapping")
    if key != XLLM_SERVICE_CONFIG_KEY:
        deployment[XLLM_SERVICE_CONFIG_KEY] = deployment.pop(key)
    return task_config


def get_xllm_service_config(task_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical control mapping, accepting legacy configurations."""
    normalize_xllm_service_config(task_config)
    deployment = task_config.get("deployment") or {}
    if not isinstance(deployment, dict):
        return {}
    return deployment.get(XLLM_SERVICE_CONFIG_KEY) or {}


def canonical_xllm_service_parameter_path(path: str) -> str:
    """Accept legacy dotted assess paths without retaining duplicate blocks."""
    for key in LEGACY_XLLM_SERVICE_CONFIG_KEYS:
        prefix = f"deployment.{key}"
        if path == prefix or path.startswith(prefix + "."):
            return "deployment.xllm_service" + path[len(prefix) :]
    return path


# The first implementation co-locates Service processes with existing Engine
# containers. These plans never enter ranktable/device accounting.


@dataclass(frozen=True)
class XllmServiceInstance:
    name: str
    role: str
    host: str
    http_server_port: int
    rpc_server_port: int
    container_instance: Optional[str] = None


@dataclass(frozen=True)
class XllmServiceTopology:
    etcd_host: str
    etcd_client_port: int
    etcd_peer_port: int
    etcd_container_instance: Optional[str]
    instances: tuple[XllmServiceInstance, ...]
    router_name: str
    explicit: bool

    @property
    def router(self) -> XllmServiceInstance:
        return next(item for item in self.instances if item.name == self.router_name)


def _service_profile(control):
    profile = str(control.get("service_binary_profile", "auto")).strip().lower()
    if profile not in {"auto", "legacy", "refactor"}:
        raise ValueError("service_binary_profile must be auto, legacy or refactor")
    return profile


def has_service_instance_lists(task_config: Dict[str, Any]) -> bool:
    control = get_xllm_service_config(task_config)
    return "coordinators" in control or "routers" in control


def _service_port(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a port between 1 and 65535")
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a port between 1 and 65535") from None
    if str(value).strip() != str(port) or not 1 <= port <= 65535:
        raise ValueError(f"{field} must be a port between 1 and 65535")
    return port


def _service_host(host, field):
    # Hosts appear in curl URLs and etcd advertised endpoints. Phase 1 uses
    # IPv4/DNS host syntax; shell metacharacters and wildcard binds are not
    # valid advertised addresses.
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError(f"{field} must identify an existing Engine node by IPv4/DNS host")
    if host == "0.0.0.0":
        raise ValueError(f"{field} cannot be a wildcard listen address")
    return host


def _instance_list(control, key, role, default_host):
    raw = control.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"xllm_service.{key} must be a non-empty list")
    result = []
    for index, item in enumerate(raw):
        prefix = f"xllm_service.{key}[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{prefix} must be a mapping")
        unknown = set(item) - {
            "name",
            "host",
            "http_server_port",
            "rpc_server_port",
            "container_instance",
        }
        if unknown:
            raise ValueError(f"Unsupported {prefix} fields: {sorted(unknown)}")
        name = item.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"{prefix}.name must be a safe, non-empty instance name")
        host = _service_host(item.get("host") or default_host, prefix + ".host")
        container = item.get("container_instance")
        if container is not None and (not isinstance(container, str) or not container):
            raise ValueError(f"{prefix}.container_instance must be a node instance name")
        result.append(
            XllmServiceInstance(
                name,
                role,
                host,
                _service_port(item.get("http_server_port"), prefix + ".http_server_port"),
                _service_port(item.get("rpc_server_port"), prefix + ".rpc_server_port"),
                container,
            )
        )
    return result


def build_xllm_service_topology(task_config: Dict[str, Any]) -> XllmServiceTopology:
    """Resolve proposed split topology before launching any process.

    Flat configurations reserve deterministic additional ports for a possible
    new binary. Actual legacy execution still uses its original single process.
    Defaults are checked, never silently shifted around an occupied port.
    """
    control = get_xllm_service_config(task_config)
    deployment = task_config.get("deployment") or {}
    default_host = control.get("host") or deployment.get("master_host")
    etcd = control.get("etcd") or {}
    if not isinstance(etcd, dict):
        raise TypeError("xllm_service.etcd must be a mapping")
    unknown = set(etcd) - {"host", "client_port", "peer_port", "container_instance"}
    if unknown:
        raise ValueError(f"Unsupported xllm_service.etcd fields: {sorted(unknown)}")
    for nested, flat in (("client_port", "etcd_client_port"), ("peer_port", "etcd_peer_port")):
        if nested in etcd and flat in control and etcd[nested] != control[flat]:
            raise ValueError(f"Conflicting xllm_service.etcd.{nested} and {flat}")
    host = _service_host(etcd.get("host") or default_host, "xllm_service.etcd.host")
    client = _service_port(
        etcd.get("client_port", control.get("etcd_client_port", 3389)),
        "xllm_service.etcd.client_port",
    )
    peer = _service_port(
        etcd.get("peer_port", control.get("etcd_peer_port", 3390)), "xllm_service.etcd.peer_port"
    )
    container = etcd.get("container_instance", control.get("container_instance"))
    if container is not None and (not isinstance(container, str) or not container):
        raise ValueError("etcd.container_instance must be an Engine node instance name")
    for field, default in (
        ("startup_attempts", 2),
        ("startup_wait_seconds", 3),
        ("master_key_wait_seconds", 30),
    ):
        raw = control.get(field, default)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1 or raw > 300:
            if field == "startup_wait_seconds" and raw == 0 and not isinstance(raw, bool):
                continue
            raise ValueError(f"xllm_service.{field} must be a bounded positive integer")
    namespace = control.get("etcd_namespace", "")
    if not isinstance(namespace, str) or any(ord(char) < 32 for char in namespace):
        raise ValueError("xllm_service.etcd_namespace must be a printable string")
    explicit = has_service_instance_lists(task_config)
    if explicit:
        engine_roles = {
            str(item.get("role", "")).lower() for item in deployment.get("instances") or []
        }
        if deployment.get("mode") not in {"pd_disaggregated", "service_routed"} and not (
            deployment.get("pd_instances") or engine_roles.intersection({"prefill", "decode"})
        ):
            raise ValueError("Coordinator/Router lists require PD or service_routed deployment")
        if _service_profile(control) == "legacy":
            raise ValueError("Coordinator/Router lists require a new Service binary")
        ignored = {
            "service_deployment_role",
            "http_server_port",
            "rpc_server_port",
            "coordinator_http_server_port",
            "router_rpc_server_port",
        }.intersection(control)
        if ignored:
            raise ValueError(
                f"Do not combine instance lists with flat Service fields: {sorted(ignored)}"
            )
        instances = _instance_list(control, "coordinators", "COORDINATOR", default_host)
        instances += _instance_list(control, "routers", "ROUTER", default_host)
    else:
        if default_host and host != default_host:
            raise ValueError(
                "Separate etcd and Service hosts require explicit Coordinator/Router lists"
            )
        http = _service_port(control.get("http_server_port", 38888), "http_server_port")
        rpc = _service_port(control.get("rpc_server_port", 38889), "rpc_server_port")
        coordinator_http = _service_port(
            control.get("coordinator_http_server_port", http + 10000),
            "coordinator_http_server_port",
        )
        router_rpc = _service_port(
            control.get("router_rpc_server_port", rpc + 10000), "router_rpc_server_port"
        )
        container = control.get("container_instance")
        instances = [
            XllmServiceInstance(
                "coordinator-0", "COORDINATOR", host, coordinator_http, rpc, container
            ),
            XllmServiceInstance("router-0", "ROUTER", host, http, router_rpc, container),
        ]
    reserved_names = {"etcd", "remote", "archive"}
    if any(item.name in reserved_names for item in instances):
        raise ValueError("Service instance names cannot use etcd/remote/archive")
    names = [item.name for item in instances]
    if len(names) != len(set(names)):
        raise ValueError("xllm_service instance names must be globally unique")
    routers = [item for item in instances if item.role == "ROUTER"]
    selected = (deployment.get("benchmark") or {}).get("router_instance")
    if not selected:
        if len(routers) != 1:
            raise ValueError("Multiple Routers require deployment.benchmark.router_instance")
        selected = routers[0].name
    if selected not in [item.name for item in routers]:
        raise ValueError(f"Unknown benchmark Router: {selected!r}")
    # Named-shard model subscriptions are a separate deployment feature. Fail
    # closed rather than silently configuring only one side of that contract.
    if control.get("control_plane_shard_id", "default") != "default":
        raise ValueError("Phase 1 supports control_plane_shard_id=default only")
    if control.get("master_service_key", "XLLM:SERVICE:MASTER") != "XLLM:SERVICE:MASTER":
        raise ValueError(
            "Phase 1 derives the default-shard master key; custom master_service_key is unsupported"
        )
    return XllmServiceTopology(
        host,
        client,
        peer,
        etcd.get("container_instance", control.get("container_instance")),
        tuple(instances),
        selected,
        explicit,
    )


def select_service_container(
    candidates: Sequence, host: str, container_instance=None, benchmark_runner=None
):
    """Choose a node plan/runtime without relying on list ordering."""
    matches = [item for item in candidates if str(getattr(item, "plan", item).host) == host]
    if container_instance:
        matches = [
            item
            for item in matches
            if getattr(item, "plan", item).instance_name == container_instance
        ]
    elif len(matches) > 1 and benchmark_runner:
        preferred = [
            item
            for item in matches
            if getattr(item, "plan", item).instance_name == benchmark_runner
        ]
        if preferred:
            matches = preferred
    elif len(matches) > 1 and benchmark_runner is False:
        preferred = [
            item
            for item in matches
            if getattr(getattr(item, "plan", item), "is_benchmark_runner", False)
        ]
        if len(preferred) == 1:
            matches = preferred
    if len(matches) != 1:
        raise ValueError(
            f"Service host {host!r} must match a service node host with one existing Engine container; "
            "set container_instance when the host has multiple Engine nodes"
        )
    return matches[0]


def service_container_nodes(task_config: Dict[str, Any], candidates: Sequence):
    """All containers that need Service artifacts, including the etcd owner."""
    control = get_xllm_service_config(task_config)
    runner = (task_config.get("deployment") or {}).get("benchmark", {}).get("runner_instance")
    if _service_profile(control) == "legacy" and not has_service_instance_lists(task_config):
        if control.get("etcd") or control.get("etcd_namespace"):
            raise ValueError(
                "Legacy Service flow requires flat etcd configuration without namespace"
            )
        host = control.get("host") or (task_config.get("deployment") or {}).get("master_host")
        return [
            select_service_container(
                candidates, host, control.get("container_instance"), runner or False
            )
        ]
    topology = build_xllm_service_topology(task_config)
    result = [
        select_service_container(
            candidates,
            topology.etcd_host,
            topology.etcd_container_instance,
            None if topology.explicit else runner or False,
        )
    ]
    for item in topology.instances:
        chosen = select_service_container(
            candidates,
            item.host,
            item.container_instance,
            None if topology.explicit else runner or False,
        )
        if all(chosen is not existing for existing in result):
            result.append(chosen)
    return result


def service_port_claims(task_config: Dict[str, Any]):
    """Yield host-specific listeners, with legacy flat configuration preserved."""
    control = get_xllm_service_config(task_config)
    if _service_profile(control) == "legacy" and not has_service_instance_lists(task_config):
        if control.get("etcd") or control.get("etcd_namespace"):
            raise ValueError(
                "Legacy Service flow requires flat etcd configuration without namespace"
            )
        host = control.get("host") or (task_config.get("deployment") or {}).get("master_host")
        return [
            (host, _service_port(control.get(key, default), key), f"xllm_service.{key}")
            for key, default in (
                ("etcd_client_port", 3389),
                ("etcd_peer_port", 3390),
                ("http_server_port", 38888),
                ("rpc_server_port", 38889),
            )
        ]
    topology = build_xllm_service_topology(task_config)
    claims = [
        (topology.etcd_host, topology.etcd_client_port, "xllm_service.etcd.client_port"),
        (topology.etcd_host, topology.etcd_peer_port, "xllm_service.etcd.peer_port"),
    ]
    for item in topology.instances:
        claims.extend(
            [
                (item.host, item.http_server_port, f"{item.name}.http_server_port"),
                (item.host, item.rpc_server_port, f"{item.name}.rpc_server_port"),
            ]
        )
    return claims


def service_etcd_location(task_config: Dict[str, Any]):
    """Common registration address; no role counts enter Engine topology."""
    control = get_xllm_service_config(task_config)
    deployment = task_config.get("deployment") or {}
    etcd = control.get("etcd") or {}
    if not isinstance(etcd, dict):
        raise TypeError("xllm_service.etcd must be a mapping")
    host = etcd.get("host") or control.get("host") or deployment.get("master_host")
    port = int(etcd.get("client_port", control.get("etcd_client_port", 3389)))
    return host, port
