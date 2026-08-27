"""Multi-host service group planning for one inference service."""

from __future__ import annotations

import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xtask.utils.container_manager import ContainerConfig, ContainerManager
from xtask.utils.host_executor import create_host_executor, is_local_host


def validate_pd_deployment_authoring(task_config: Dict[str, Any]) -> None:
    """Require hierarchical authoring for every PD deployment.

    ``deployment.instances`` remains the scheduler's canonical runtime shape,
    including the result produced by :func:`normalize_pd_deployment`.  It is
    no longer accepted as the YAML authoring shape for PD, however.  Keeping
    this validation at configuration boundaries lets non-PD service groups
    continue to use flat instances without creating a second PD runtime path.
    """
    if not isinstance(task_config, dict):
        raise TypeError("task_config must be a mapping")

    deployment = task_config.get("deployment")
    if not isinstance(deployment, dict):
        return

    service_instances = deployment.get("service_instances")
    service_mode = deployment.get("mode") == "service_routed"
    if service_mode or service_instances is not None:
        if not service_mode:
            raise ValueError(
                "deployment.service_instances requires "
                "deployment.mode: service_routed"
            )
        existing_instances = deployment.get("instances") or []
        if service_instances is None and existing_instances and all(
            isinstance(item, dict) and item.get("service_group")
            for item in existing_instances
        ):
            return
        if deployment.get("instances") not in (None, []):
            raise ValueError(
                "deployment.service_instances cannot be combined with "
                "deployment.instances"
            )
        if not isinstance(service_instances, list) or not service_instances:
            raise ValueError(
                "deployment.service_instances must be a non-empty list"
            )
        service_control = deployment.get("service_control")
        if not isinstance(service_control, dict):
            raise ValueError(
                "service_routed deployment requires deployment.service_control"
            )
        if not (service_control.get("host") or deployment.get("master_host")):
            raise ValueError(
                "service_routed deployment requires service_control.host or "
                "deployment.master_host"
            )
        return

    # The hierarchical form is validated and expanded by the normalizer.
    if deployment.get("pd_instances") is not None:
        return

    instances = deployment.get("instances") or []
    roles = {
        str(instance.get("role", "")).strip().lower()
        for instance in instances
        if isinstance(instance, dict)
    }
    is_pd = (
        deployment.get("mode") == "pd_disaggregated"
        or bool(roles & {"prefill", "decode"})
    )
    if is_pd:
        raise ValueError(
            "PD deployment YAML must use deployment.pd_instances; flat "
            "deployment.instances with prefill/decode roles is no longer supported"
        )


def normalize_service_routed_deployment(task_config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand independent ordinary xLLM service instances into node plans.

    ``service_instances`` is deliberately separate from ``instances``.  The
    latter keeps its historical meaning (nodes of one ordinary distributed
    runtime), while each item here is one independently routable DEFAULT
    runtime and may contain multiple nodes.
    """
    if not isinstance(task_config, dict):
        raise TypeError("task_config must be a mapping")

    deployment = task_config.get("deployment")
    if not isinstance(deployment, dict):
        return task_config
    service_instances = deployment.get("service_instances")
    if service_instances is None:
        return task_config
    if deployment.get("mode") != "service_routed":
        raise ValueError(
            "deployment.service_instances requires "
            "deployment.mode: service_routed"
        )
    if deployment.get("instances") not in (None, []):
        raise ValueError(
            "deployment.service_instances cannot be combined with "
            "deployment.instances"
        )
    if not isinstance(service_instances, list) or not service_instances:
        raise ValueError("deployment.service_instances must be a non-empty list")

    flattened: List[Dict[str, Any]] = []
    seen_groups = set()
    seen_nodes = set()
    for group_index, group in enumerate(service_instances):
        if not isinstance(group, dict):
            raise TypeError(
                f"deployment.service_instances[{group_index}] must be a mapping"
            )
        group_name = str(group.get("name") or "").strip()
        if not group_name:
            raise ValueError(
                f"deployment.service_instances[{group_index}].name is required"
            )
        if group_name in seen_groups:
            raise ValueError(f"Duplicate service logical instance name: {group_name}")
        seen_groups.add(group_name)

        role = str(group.get("role", "worker")).strip().lower()
        if role in {"prefill", "decode"}:
            raise ValueError(
                "service_routed instances are ordinary DEFAULT runtimes; "
                "use deployment.pd_instances for prefill/decode roles"
            )
        nodes = group.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(
                f"deployment.service_instances[{group_index}].nodes must be a "
                "non-empty list"
            )

        group_defaults = {
            key: copy.deepcopy(value)
            for key, value in group.items()
            if key not in {"name", "role", "nodes", "service_group", "group"}
        }
        group_extension = group_defaults.pop("serving_extensiable_param", None)
        if group_extension is None:
            group_extension = group_defaults.pop("serving_extensible_param", None)
        if group_extension is not None and not isinstance(group_extension, dict):
            raise TypeError(
                f"deployment.service_instances[{group_index}]."
                "serving_extensiable_param must be a mapping"
            )
        group_extension = copy.deepcopy(group_extension or {})

        for node_index, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise TypeError(
                    f"deployment.service_instances[{group_index}].nodes"
                    f"[{node_index}] must be a mapping"
                )
            node_name = str(
                node.get("name") or f"{group_name}-node-{node_index}"
            ).strip()
            if not node_name:
                raise ValueError(f"Empty node name in service instance {group_name}")
            if node_name in seen_nodes:
                raise ValueError(f"Duplicate service node name: {node_name}")
            seen_nodes.add(node_name)
            host = node.get("host", group_defaults.get("host"))
            if not host:
                raise ValueError(f"Service node {node_name} requires a host")

            for internal_field in ("service_group", "group"):
                if internal_field in node:
                    raise ValueError(
                        f"deployment.service_instances[{group_index}].nodes"
                        f"[{node_index}].{internal_field} is an internal field; "
                        "it is derived from the logical instance name"
                    )
            node_role = str(node.get("role", role)).strip().lower()
            if node_role in {"prefill", "decode"}:
                raise ValueError(
                    "service_routed instances are ordinary DEFAULT runtimes; "
                    "use deployment.pd_instances for prefill/decode roles"
                )
            flattened_node = copy.deepcopy(group_defaults)
            flattened_node.update(copy.deepcopy(node))
            flattened_node["name"] = node_name
            flattened_node["role"] = node_role or "worker"
            flattened_node["service_group"] = group_name

            node_extension = node.get("serving_extensiable_param")
            if node_extension is None:
                node_extension = node.get("serving_extensible_param")
            if node_extension is not None and not isinstance(node_extension, dict):
                raise TypeError(
                    f"Service node {node_name}.serving_extensiable_param must be a mapping"
                )
            merged_extension = copy.deepcopy(group_extension)
            merged_extension.update(copy.deepcopy(node_extension or {}))
            if merged_extension:
                flattened_node["serving_extensiable_param"] = merged_extension
            flattened_node.pop("serving_extensible_param", None)
            flattened.append(flattened_node)

    deployment["instances"] = flattened
    deployment.pop("service_instances", None)
    return task_config


def is_service_routed_deployment(task_config: Dict[str, Any]) -> bool:
    normalize_service_routed_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    return deployment.get("mode") == "service_routed"


def normalize_pd_deployment(task_config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand the optional hierarchical PD topology into node instances.

    The scheduler historically consumes ``deployment.instances`` where one
    item represents one host/node.  ``deployment.pd_instances`` adds an
    opt-in, hierarchical spelling for one xLLM endpoint containing multiple
    logical Prefill/Decode instances, each of which owns one or more nodes::

        deployment:
          pd_instances:
            - name: prefill-0
              role: prefill
              nodes:
                - host: 10.0.0.1
                  device_ids: [0, 1]

    Each logical instance ``name`` becomes its internal ``pd_group``.  The
    hierarchical authoring format deliberately does not expose ``pd_group``;
    this avoids two independently configurable names for the same logical
    instance.  The generated group keeps ranktable, rank-offset, and
    world-size calculations isolated while the existing scheduler still
    launches one runtime per node.  Flat ``deployment.instances`` remains the
    canonical runtime representation and is also still the authoring format
    for non-PD single-host/multi-host service groups.
    """
    if not isinstance(task_config, dict):
        raise TypeError("task_config must be a mapping")

    normalize_service_routed_deployment(task_config)

    deployment = task_config.get("deployment")
    if not isinstance(deployment, dict):
        return task_config

    pd_instances = deployment.get("pd_instances")
    if pd_instances is None:
        return task_config

    existing_instances = deployment.get("instances")
    if existing_instances not in (None, []):
        raise ValueError(
            "deployment.pd_instances cannot be combined with "
            "deployment.instances"
        )
    if not isinstance(pd_instances, list) or not pd_instances:
        raise ValueError("deployment.pd_instances must be a non-empty list")

    flattened = []
    seen_group_names = set()
    seen_node_names = set()

    for group_index, group in enumerate(pd_instances):
        if not isinstance(group, dict):
            raise TypeError(
                f"deployment.pd_instances[{group_index}] must be a mapping"
            )

        for internal_field in ("pd_group", "group"):
            if internal_field in group:
                raise ValueError(
                    f"deployment.pd_instances[{group_index}].{internal_field} "
                    "is an internal field; use name to identify the logical "
                    "PD instance"
                )

        logical_name = str(group.get("name") or "").strip()
        if not logical_name:
            raise ValueError(
                f"deployment.pd_instances[{group_index}].name is required"
            )

        role = str(group.get("role") or "").strip().lower()
        if role not in {"prefill", "decode"}:
            raise ValueError(
                f"deployment.pd_instances[{group_index}].role must be "
                "'prefill' or 'decode'"
            )

        pd_group = logical_name
        if pd_group in seen_group_names:
            raise ValueError(f"Duplicate PD logical instance name: {pd_group}")
        seen_group_names.add(pd_group)

        nodes = group.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError(
                f"deployment.pd_instances[{group_index}].nodes must be a "
                "non-empty list"
            )

        group_defaults = {
            key: copy.deepcopy(value)
            for key, value in group.items()
            if key not in {"name", "role", "group", "pd_group", "nodes"}
        }
        group_extension = group_defaults.pop("serving_extensiable_param", None)
        if group_extension is None:
            group_extension = group_defaults.pop("serving_extensible_param", None)
        if group_extension is not None and not isinstance(group_extension, dict):
            raise TypeError(
                f"deployment.pd_instances[{group_index}].serving_extensiable_param "
                "must be a mapping"
            )
        group_extension = copy.deepcopy(group_extension or {})

        for node_index, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise TypeError(
                    f"deployment.pd_instances[{group_index}].nodes[{node_index}] "
                    "must be a mapping"
                )

            for internal_field in ("pd_group", "group"):
                if internal_field in node:
                    raise ValueError(
                        f"deployment.pd_instances[{group_index}].nodes"
                        f"[{node_index}].{internal_field} is an internal field; "
                        "it is derived from the logical instance name"
                    )

            node_name = str(
                node.get("name") or f"{logical_name}-node-{node_index}"
            ).strip()
            if not node_name:
                raise ValueError(
                    f"Empty node name in PD logical instance {logical_name}"
                )
            if node_name in seen_node_names:
                raise ValueError(f"Duplicate PD node name: {node_name}")
            seen_node_names.add(node_name)

            host = node.get("host", group_defaults.get("host"))
            if not host:
                raise ValueError(
                    f"PD node {node_name} requires a host"
                )
            host = str(host)

            flattened_node = copy.deepcopy(group_defaults)
            flattened_node.update(copy.deepcopy(node))
            flattened_node["name"] = node_name
            node_role = str(node.get("role") or role).lower()
            if node_role != role:
                raise ValueError(
                    f"PD node {node_name} role {node_role!r} does not match "
                    f"logical instance role {role!r}"
                )
            flattened_node["role"] = node_role
            if node_role not in {"prefill", "decode"}:
                raise ValueError(
                    f"PD node {node_name} role must be 'prefill' or 'decode'"
                )

            flattened_node["pd_group"] = pd_group

            node_extension = node.get("serving_extensiable_param")
            if node_extension is None:
                node_extension = node.get("serving_extensible_param")
            if node_extension is not None and not isinstance(node_extension, dict):
                raise TypeError(
                    f"PD node {node_name}.serving_extensiable_param must be a mapping"
                )
            merged_extension = copy.deepcopy(group_extension)
            merged_extension.update(copy.deepcopy(node_extension or {}))
            if merged_extension:
                flattened_node["serving_extensiable_param"] = merged_extension
            flattened_node.pop("serving_extensible_param", None)
            flattened.append(flattened_node)

    deployment["instances"] = flattened
    # The normalized list is the runtime source of truth.  Removing the
    # authoring-only hierarchy makes repeated normalization idempotent and
    # prevents downstream code from accidentally processing it twice.
    deployment.pop("pd_instances", None)
    return task_config


def is_multi_host_deployment(task_config: Dict[str, Any]) -> bool:
    """True when task uses deployment.instances (including single-node).

    Single-host cases are modeled as a one-instance deployment; the scheduler
    service-group path handles both single- and multi-node layouts.
    """
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    instances = deployment.get("instances") or []
    return bool(instances)


def resolve_instance_local_port(
    instance: Dict[str, Any],
    task_config: Optional[Dict[str, Any]] = None,
    default: int = 8800,
) -> int:
    """Resolve per-instance HTTP listen port (local_port; legacy service.port)."""
    if instance.get("local_port") is not None:
        return int(instance["local_port"])
    service = instance.get("service") or {}
    if service.get("port") is not None:
        return int(service["port"])
    if task_config and task_config.get("port") is not None:
        return int(task_config["port"])
    return int(default)


def apply_host_addr_override(task_config: Dict[str, Any], host_addr: str) -> None:
    """Apply CLI/host override to top-level and single-instance deployment."""
    if not host_addr:
        return
    normalize_pd_deployment(task_config)
    task_config["host_addr"] = host_addr
    deployment = task_config.setdefault("deployment", {})
    instances = deployment.get("instances") or []
    if len(instances) != 1:
        return
    instances[0]["host"] = host_addr
    deployment["master_host"] = host_addr


def resolve_master_host(task_config: Dict[str, Any]) -> str:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    if deployment.get("mode") == "service_routed":
        service_control = deployment.get("service_control") or {}
        if service_control.get("host"):
            return str(service_control["host"])
    if deployment.get("master_host"):
        return deployment["master_host"]
    for instance in deployment.get("instances") or []:
        role = str(instance.get("role", "")).lower()
        if role in {"master", "prefill", "standalone"}:
            return instance["host"]
    instances = deployment.get("instances") or []
    if instances:
        return instances[0]["host"]
    return task_config.get("host_addr", "127.0.0.1")


def resolve_global_nnodes(task_config: Dict[str, Any]) -> int:
    """World size for multi-host: prefer sum of instance devices over top-level world_size."""
    normalize_pd_deployment(task_config)
    instances = (task_config.get("deployment") or {}).get("instances") or []
    if instances:
        total = 0
        for instance in instances:
            device_ids = instance.get("device_ids")
            if device_ids:
                total += len(device_ids)
            else:
                total += int(instance.get("num_devices", 0) or 0)
        if total > 0:
            return total
    if task_config.get("world_size"):
        return int(task_config["world_size"])
    return int(task_config.get("num_devices", 1) or 1)


def _device_ids_from_instance(instance: Dict[str, Any], task_config: Dict[str, Any]) -> List[int]:
    if instance.get("device_ids"):
        return list(instance["device_ids"])
    start = int(instance.get("start_device_id", task_config.get("start_device_id", 0)))
    count = int(instance.get("num_devices", task_config.get("num_devices", 1)))
    return list(range(start, start + count))


def merge_instance_task_config(
    task_config: Dict[str, Any],
    instance: Dict[str, Any],
    master_host: str,
    global_nnodes: int,
) -> Dict[str, Any]:
    """Overlay one deployment instance onto the shared task config.

    Per-instance fields (host / device_ids / local_port / communication ports /
    serving_extensiable_param) take precedence. Top-level host_addr / num_devices /
    start_device_id / world_size / communication ports / serving_extensiable_param
    are optional fallbacks for older YAMLs.
    """
    normalize_pd_deployment(task_config)
    cfg = copy.deepcopy(task_config)
    device_ids = _device_ids_from_instance(instance, task_config)

    cfg["host_addr"] = instance["host"]
    cfg["start_device_id"] = min(device_ids) if device_ids else 0
    cfg["num_devices"] = len(device_ids) if device_ids else 1
    cfg["device_ids"] = device_ids
    cfg["port"] = resolve_instance_local_port(instance, task_config)
    cfg["role"] = instance.get("role", "worker")
    cfg["rank_start"] = int(instance.get("rank_start", 0))
    cfg["master_host"] = master_host
    cfg["global_nnodes"] = global_nnodes
    cfg["world_size"] = global_nnodes
    cfg["instance_name"] = instance.get("name", "serving")
    if instance.get("pd_group") is not None:
        cfg["pd_group"] = str(instance["pd_group"])
    if instance.get("service_group") is not None:
        cfg["service_group"] = str(instance["service_group"])

    for field_name in ("hccl_if_port", "progress_conn_port", "disagg_pd_port"):
        if instance.get(field_name) is not None:
            cfg[field_name] = int(instance[field_name])

    for field_name in ("ipc_mode", "shm_size"):
        if instance.get(field_name) is not None:
            cfg[field_name] = str(instance[field_name])

    # Existing-container deployments need these values preserved in the
    # instance-merged config used by both the scheduler and script generator.
    for field_name in (
        "container_name",
        "reuse_existing_container",
        "xllm_binary",
        "triton_binary_path",
        "serving_ready_timeout",
    ):
        if instance.get(field_name) is not None:
            cfg[field_name] = copy.deepcopy(instance[field_name])

    # Instance-level serving_extensiable_param is the primary source; top-level
    # remains a shared default for backward-compatible multihost YAMLs.
    base_ext = copy.deepcopy(cfg.get("serving_extensiable_param") or {})
    base_ext.update(instance.get("serving_extensiable_param") or {})
    deployment = task_config.get("deployment") or {}
    pd_control = deployment.get("pd_control") or {}
    instance_role = str(instance.get("role", cfg.get("role", ""))).lower()
    if deployment.get("mode") == "pd_disaggregated" or instance_role in {"prefill", "decode"}:
        cfg["enable_disagg_pd"] = True
        cfg["pd_role"] = instance_role.upper()
        cfg["pd_master_host"] = pd_control.get("host") or master_host
        cfg["pd_etcd_port"] = int(pd_control.get("etcd_client_port", 3389))
        cfg["pd_service_port"] = int(pd_control.get("http_server_port", 38888))
        cfg["transfer_port_base"] = int(
            instance.get(
                "transfer_port_base",
                pd_control.get("transfer_port_base", 26000),
            )
        )
        instance_hosts = [
            str(item.get("host"))
            for item in deployment.get("instances") or []
            if item.get("host")
        ]
        configured_no_proxy = pd_control.get("no_proxy_hosts") or []
        cfg["pd_no_proxy"] = ",".join(
            dict.fromkeys(
                ["localhost", "127.0.0.1", "0.0.0.0", cfg["pd_master_host"]]
                + instance_hosts
                + [str(item) for item in configured_no_proxy]
            )
        )
    elif deployment.get("mode") == "service_routed":
        service_control = deployment.get("service_control") or {}
        reserved_routing_parameters = {
            "enable_service_routing",
            "etcd_addr",
            "enable_disagg_pd",
            "instance_role",
        }
        conflicting_parameters = sorted(
            reserved_routing_parameters.intersection(base_ext)
        )
        if conflicting_parameters:
            raise ValueError(
                "service_routed control parameters are managed by xTask and "
                "must not appear in serving_extensiable_param: "
                f"{conflicting_parameters}"
            )
        control_host = service_control.get("host") or resolve_master_host(task_config)
        cfg["enable_service_routing"] = True
        cfg["service_control_host"] = str(control_host)
        cfg["service_etcd_port"] = int(
            service_control.get("etcd_client_port", 3389)
        )
        instance_hosts = [
            str(item.get("host"))
            for item in deployment.get("instances") or []
            if item.get("host")
        ]
        configured_no_proxy = service_control.get("no_proxy_hosts") or []
        cfg["service_no_proxy"] = ",".join(
            dict.fromkeys(
                ["localhost", "127.0.0.1", "0.0.0.0", str(control_host)]
                + instance_hosts
                + [str(item) for item in configured_no_proxy]
            )
        )
    cfg["serving_extensiable_param"] = base_ext

    if instance.get("readiness"):
        cfg["readiness"] = instance["readiness"]
    return cfg


@dataclass
class ServiceLaunchPlan:
    instance_name: str
    role: str
    host: str
    is_local: bool
    rank_start: int
    device_ids: List[int]
    service_port: int
    master_host: str
    global_nnodes: int
    instance_config: Dict[str, Any]
    output_dir: str
    startup_cmd: str = ""
    readiness: Dict[str, Any] = field(default_factory=dict)
    is_benchmark_runner: bool = False
    # True when multiple runtime containers share one physical host.
    isolate_ipc: bool = False
    # True when a logical group needs shared memory across colocated containers.
    requires_shared_ipc: bool = False


@dataclass
class ServiceRuntime:
    plan: ServiceLaunchPlan
    container_config: ContainerConfig
    container_manager: ContainerManager
    container_id: Optional[str] = None


def _validate_port_number(port: int, description: str) -> int:
    port = int(port)
    if port < 1 or port > 65535:
        raise ValueError(f"{description} must be between 1 and 65535, got {port}")
    return port


def _config_bool(value: Any) -> bool:
    """Interpret YAML/CLI-style boolean values without treating "false" as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _plan_uses_shm(plan: ServiceLaunchPlan) -> bool:
    extension = plan.instance_config.get("serving_extensiable_param") or {}
    return _config_bool(extension.get("enable_shm", False))


def _validate_service_plans(
    task_config: Dict[str, Any],
    plans: List[ServiceLaunchPlan],
    pd_mode: bool,
) -> None:
    """Reject same-host resource and port collisions before containers start."""
    groups: Dict[str, List[ServiceLaunchPlan]] = {}
    for plan in plans:
        if pd_mode:
            group_name = str(plan.instance_config.get("pd_group"))
        elif is_service_routed_deployment(task_config):
            group_name = str(plan.instance_config.get("service_group"))
        else:
            group_name = "default"
        groups.setdefault(group_name, []).append(plan)

    device_owners: Dict[tuple[str, int], str] = {}
    for plan in plans:
        if not plan.device_ids:
            raise ValueError(f"Instance {plan.instance_name} requires device_ids")
        if len(set(plan.device_ids)) != len(plan.device_ids):
            raise ValueError(
                f"Instance {plan.instance_name} contains duplicate device_ids: "
                f"{plan.device_ids}"
            )
        for device_id in plan.device_ids:
            device_id = int(device_id)
            if device_id < 0:
                raise ValueError(
                    f"Instance {plan.instance_name} has negative device id {device_id}"
                )
            key = (str(plan.host), device_id)
            previous = device_owners.get(key)
            if previous is not None:
                raise ValueError(
                    f"Device {device_id} on host {plan.host} is assigned to both "
                    f"{previous} and {plan.instance_name}"
                )
            device_owners[key] = plan.instance_name

    # progress_conn_port identifies the coordinator of one logical group.
    # PD-only transfer/disaggregation ports have the same group-level scope.
    group_level_fields = ["progress_conn_port"]
    if pd_mode:
        group_level_fields.extend(("transfer_port_base", "disagg_pd_port"))

    group_level_defaults = {
        "progress_conn_port": 9590,
        "transfer_port_base": 26000,
        "disagg_pd_port": 7777,
    }

    # Validate values that every runtime in a logical group must share.
    for group_name, members in groups.items():
        for field_name in group_level_fields:
            values = {
                int(
                    member.instance_config.get(
                        field_name, group_level_defaults[field_name]
                    )
                )
                for member in members
            }
            if len(values) != 1:
                raise ValueError(
                    f"Logical group {group_name} must use one shared {field_name}; "
                    f"got {sorted(values)}"
                )

        group_masters = {
            str(member.instance_config.get("master_host")) for member in members
        }
        if len(group_masters) != 1:
            raise ValueError(
                f"Logical group {group_name} has inconsistent master hosts: "
                f"{sorted(group_masters)}"
            )
        if is_service_routed_deployment(task_config):
            group_master = next(iter(group_masters))
            group_hosts = {str(member.host) for member in members}
            if group_master not in group_hosts:
                raise ValueError(
                    f"Logical group {group_name} master host {group_master!r} "
                    "must match one of its node hosts"
                )
        ranks = []
        for member in members:
            ranks.extend(
                range(member.rank_start, member.rank_start + len(member.device_ids))
            )
        expected_ranks = list(range(sum(len(member.device_ids) for member in members)))
        if sorted(ranks) != expected_ranks:
            raise ValueError(
                f"Logical group {group_name} rank ranges must cover "
                f"0..{len(expected_ranks) - 1} exactly; got {sorted(ranks)}"
            )

        runtime_counts_by_host: Dict[str, int] = {}
        for member in members:
            host = str(member.host)
            runtime_counts_by_host[host] = runtime_counts_by_host.get(host, 0) + 1
        if any(_plan_uses_shm(member) for member in members):
            for member in members:
                ipc_mode = str(member.instance_config.get("ipc_mode") or "").lower()
                if (
                    runtime_counts_by_host[str(member.host)] > 1
                    and ipc_mode == "private"
                ):
                    raise ValueError(
                        f"Logical group {group_name} has multiple runtimes on host "
                        f"{member.host} with enable_shm=true; instance "
                        f"{member.instance_name} cannot use ipc_mode=private"
                    )

    port_owners: Dict[tuple[str, int], str] = {}

    def claim_port(host: str, port: int, owner: str) -> None:
        port = _validate_port_number(port, owner)
        key = (str(host), port)
        previous = port_owners.get(key)
        if previous is not None:
            raise ValueError(
                f"Port {port} on host {host} is used by both {previous} and {owner}"
            )
        port_owners[key] = owner

    for plan in plans:
        for offset in range(len(plan.device_ids)):
            claim_port(
                plan.host,
                plan.service_port + offset,
                f"instance {plan.instance_name} local_port range",
            )
            if pd_mode:
                transfer_base = int(plan.instance_config["transfer_port_base"])
                claim_port(
                    plan.host,
                    transfer_base + plan.rank_start + offset,
                    f"logical group {plan.instance_config['pd_group']} transfer ports",
                )

    # HCCL is runtime/node scoped: colocated runtimes need different ports,
    # while runtimes on different hosts may reuse the same port.
    for plan in plans:
        hccl_if_port = int(plan.instance_config.get("hccl_if_port", 43439))
        claim_port(
            plan.host,
            hccl_if_port,
            f"instance {plan.instance_name} hccl_if_port",
        )

    # Only one coordinator listens for each logical group, so claim its
    # progress port once on the group's master host.
    for group_name, members in groups.items():
        first = members[0]
        claim_port(
            first.instance_config["master_host"],
            int(first.instance_config.get("progress_conn_port", 9590)),
            f"logical group {group_name} progress_conn_port",
        )

    # xLLM shared-memory object names include progress_conn_port and rank, but
    # not the logical-group name. Groups that overlap on a physical host must
    # therefore use distinct progress ports when shared memory is enabled.
    shm_progress_owners: Dict[tuple[str, int], str] = {}
    for group_name, members in groups.items():
        if not any(_plan_uses_shm(member) for member in members):
            continue
        progress_port = _validate_port_number(
            int(members[0].instance_config.get("progress_conn_port", 9590)),
            f"logical group {group_name} shared-memory progress_conn_port",
        )
        for host in {str(member.host) for member in members}:
            key = (host, progress_port)
            previous = shm_progress_owners.get(key)
            if previous is not None:
                raise ValueError(
                    f"Shared-memory progress_conn_port {progress_port} on host "
                    f"{host} is reused by logical groups {previous} and "
                    f"{group_name}; assign distinct progress_conn_port values"
                )
            shm_progress_owners[key] = group_name

    # Validate disagg_pd_port for each logical group in PD mode
    if pd_mode:
        for group_name, members in groups.items():
            first = members[0]
            claim_port(
                first.instance_config["master_host"],
                int(first.instance_config.get("disagg_pd_port", 7777)),
                f"logical group {group_name} disagg_pd_port",
            )

    if pd_mode:
        deployment = task_config.get("deployment") or {}
        pd_control = deployment.get("pd_control") or {}
        control_host = str(pd_control.get("host") or resolve_master_host(task_config))
        for field_name, default in (
            ("etcd_client_port", 3389),
            ("etcd_peer_port", 3390),
            ("http_server_port", 38888),
            ("rpc_server_port", 38889),
        ):
            claim_port(
                control_host,
                int(pd_control.get(field_name, default)),
                f"pd_control.{field_name}",
            )
    elif is_service_routed_deployment(task_config):
        deployment = task_config.get("deployment") or {}
        service_control = deployment.get("service_control") or {}
        control_host = str(service_control.get("host") or resolve_master_host(task_config))
        runtime_hosts = {str(plan.host) for plan in plans}
        if control_host not in runtime_hosts:
            raise ValueError(
                f"service_control.host {control_host!r} must match a service node host"
            )
        for field_name, default in (
            ("etcd_client_port", 3389),
            ("etcd_peer_port", 3390),
            ("http_server_port", 38888),
            ("rpc_server_port", 38889),
        ):
            claim_port(
                control_host,
                int(service_control.get(field_name, default)),
                f"service_control.{field_name}",
            )


def build_service_plans(
    task_config: Dict[str, Any],
    output_file_path: str,
) -> List[ServiceLaunchPlan]:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    instances = deployment.get("instances") or []
    if not instances:
        raise ValueError("build_service_plans requires deployment.instances")

    master_host = resolve_master_host(task_config)
    global_nnodes = resolve_global_nnodes(task_config)
    benchmark_cfg = deployment.get("benchmark") or {}
    runner_instance = benchmark_cfg.get("runner_instance")
    if not runner_instance:
        for instance in instances:
            if str(instance.get("role", "")).lower() in {"master", "standalone", "prefill"}:
                runner_instance = instance.get("name")
                break
        if not runner_instance:
            runner_instance = instances[0].get("name")

    plans: List[ServiceLaunchPlan] = []
    pd_mode = (
        deployment.get("mode") == "pd_disaggregated"
        or any(
            str(instance.get("role", "")).lower() in {"prefill", "decode"}
            for instance in instances
        )
    )
    service_routed = is_service_routed_deployment(task_config)
    if pd_mode and service_routed:
        raise ValueError("PD and service_routed deployment modes are mutually exclusive")
    group_devices: Dict[str, int] = {}
    group_instances: Dict[str, List[Dict[str, Any]]] = {}
    for instance in instances:
        if pd_mode:
            pd_group = instance.get("pd_group")
            if not pd_group:
                raise ValueError(
                    f"PD runtime instance {instance.get('name')!r} is missing "
                    "pd_group; author PD YAML with deployment.pd_instances"
                )
            group_name = str(pd_group)
        elif service_routed:
            group_name = instance.get("service_group")
            if not group_name:
                raise ValueError(
                    f"Service runtime instance {instance.get('name')!r} is missing "
                    "service_group; author YAML with deployment.service_instances"
                )
            group_name = str(group_name)
        else:
            group_name = "default"
        device_count = len(_device_ids_from_instance(instance, task_config))
        group_devices[group_name] = group_devices.get(group_name, 0) + device_count
        group_instances.setdefault(group_name, []).append(instance)

    group_masters: Dict[str, str] = {}
    for group_name, group_members in group_instances.items():
        explicit_master = next(
            (member.get("group_master_host") for member in group_members if member.get("group_master_host")),
            None,
        )
        group_masters[group_name] = str(
            explicit_master or group_members[0]["host"]
        )

    # A normal deployment is one logical group. Service-routed deployments
    # resolve one coordinator port per independent logical instance.
    master_progress_port = task_config.get("progress_conn_port")
    if master_progress_port is None:
        master_progress_port = 9590
    if not pd_mode and not service_routed:
        for instance in instances:
            role = str(instance.get("role", "worker")).lower()
            if role in {"master", "standalone"}:
                configured_port = instance.get("progress_conn_port")
                if configured_port is not None:
                    master_progress_port = configured_port
                    break

    group_progress_ports: Dict[str, int] = {}
    if service_routed:
        for group_name, group_members in group_instances.items():
            configured = [
                member.get("progress_conn_port")
                for member in group_members
                if member.get("progress_conn_port") is not None
            ]
            configured_values = {int(value) for value in configured}
            if len(configured_values) > 1:
                raise ValueError(
                    f"Logical group {group_name} must use one shared "
                    f"progress_conn_port; got {sorted(configured_values)}"
                )
            group_progress_ports[group_name] = int(
                configured[0]
                if configured
                else task_config.get("progress_conn_port", 9590)
            )

    group_rank_offsets: Dict[str, int] = {}
    for instance in instances:
        name = instance.get("name")
        if not name:
            raise ValueError("deployment.instances[].name is required")
        host = instance["host"]
        device_ids = _device_ids_from_instance(instance, task_config)
        if pd_mode:
            group_name = str(instance["pd_group"])
        elif service_routed:
            group_name = str(instance["service_group"])
        else:
            group_name = "default"
        instance_for_merge = copy.deepcopy(instance)
        if "rank_start" not in instance_for_merge:
            instance_for_merge["rank_start"] = group_rank_offsets.get(group_name, 0)
        group_rank_offsets[group_name] = int(instance_for_merge.get("rank_start", 0)) + len(device_ids)

        if not pd_mode and not service_routed:
            instance_for_merge["progress_conn_port"] = master_progress_port
        elif service_routed:
            instance_for_merge["progress_conn_port"] = group_progress_ports[group_name]

        effective_master = group_masters[group_name] if (pd_mode or service_routed) else master_host
        effective_world = group_devices[group_name] if (pd_mode or service_routed) else global_nnodes
        instance_config = merge_instance_task_config(
            task_config, instance_for_merge, effective_master, effective_world
        )
        output_dir = os.path.join(output_file_path, "xllm_log", name)
        os.makedirs(output_dir, exist_ok=True)
        plans.append(
            ServiceLaunchPlan(
                instance_name=name,
                role=str(instance.get("role", "worker")),
                host=host,
                is_local=is_local_host(host),
                rank_start=int(instance_config.get("rank_start", 0)),
                device_ids=device_ids,
                service_port=resolve_instance_local_port(instance, task_config),
                master_host=effective_master,
                global_nnodes=effective_world,
                instance_config=instance_config,
                output_dir=output_dir,
                readiness=instance.get("readiness") or {},
                is_benchmark_runner=(name == runner_instance),
            )
        )
    _validate_service_plans(task_config, plans, pd_mode)
    runtime_counts_by_host: Dict[str, int] = {}
    runtime_counts_by_group_host: Dict[tuple[str, str], int] = {}
    group_uses_shm: Dict[str, bool] = {}
    for plan in plans:
        host_key = str(plan.host)
        if pd_mode:
            group_key = str(plan.instance_config.get("pd_group"))
        elif service_routed:
            group_key = str(plan.instance_config.get("service_group"))
        else:
            group_key = "default"
        runtime_counts_by_host[host_key] = runtime_counts_by_host.get(host_key, 0) + 1
        group_host_key = (group_key, host_key)
        runtime_counts_by_group_host[group_host_key] = (
            runtime_counts_by_group_host.get(group_host_key, 0) + 1
        )
        group_uses_shm[group_key] = group_uses_shm.get(group_key, False) or (
            _plan_uses_shm(plan)
        )
    for plan in plans:
        host_key = str(plan.host)
        if pd_mode:
            group_key = str(plan.instance_config.get("pd_group"))
        elif service_routed:
            group_key = str(plan.instance_config.get("service_group"))
        else:
            group_key = "default"
        plan.isolate_ipc = runtime_counts_by_host[host_key] > 1
        plan.requires_shared_ipc = (
            runtime_counts_by_group_host[(group_key, host_key)] > 1
            and group_uses_shm[group_key]
        )
    return plans


def create_runtime_for_plan(
    plan: ServiceLaunchPlan,
    engine_config: Dict[str, Any],
    task_yaml_path: str,
    container_config_builder,
) -> ServiceRuntime:
    executor = create_host_executor(plan.host)
    # Reuse an explicitly named container when requested; otherwise retain the
    # historical unique-name behavior for xTask-created containers.
    case_name = plan.instance_config.get("case_name", "xllm")
    import time

    container_config = container_config_builder.create_container_config(
        engine_config, plan.instance_config, task_yaml_path
    )
    if plan.isolate_ipc:
        if container_config.ipc_mode is None:
            container_config.ipc_mode = "host"
        elif plan.requires_shared_ipc and (
            str(container_config.ipc_mode).lower() == "private"
        ):
            raise ValueError(
                f"Instance {plan.instance_name} shares host {plan.host} with "
                "another runtime and uses enable_shm=true; "
                "ipc_mode=private would isolate xLLM shared memory"
            )
    requested_name = plan.instance_config.get("container_name")
    if requested_name:
        container_config.container_name = str(requested_name)
    else:
        container_config.container_name = (
            f"{case_name}-{plan.instance_name}-{int(time.time())}"
        )
    manager = ContainerManager(container_config, logging.getLogger(), executor=executor)
    return ServiceRuntime(
        plan=plan,
        container_config=container_config,
        container_manager=manager,
    )


def get_benchmark_endpoint(task_config: Dict[str, Any], plans: List[ServiceLaunchPlan]) -> str:
    """Derive the Benchmark endpoint from deployment topology.

    PD deployments use ``pd_control.host:http_server_port`` and ordinary
    service-routed deployments use ``service_control.host:http_server_port``.
    Other deployments use ``master_host`` and the matching instance's
    ``local_port``. An explicit ``deployment.benchmark.endpoint`` is
    intentionally ignored so topology has one authoritative source for
    service addressing.
    """
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    pd_control = deployment.get("pd_control")
    if pd_control is not None:
        control_host = pd_control.get("host")
        control_port = pd_control.get("http_server_port")
        if not control_host or control_port is None:
            raise ValueError(
                "PD deployment requires pd_control.host and "
                "pd_control.http_server_port to derive Benchmark endpoint"
            )
        return f"http://{control_host}:{int(control_port)}"

    if deployment.get("mode") == "service_routed":
        service_control = deployment.get("service_control") or {}
        control_host = service_control.get("host") or resolve_master_host(task_config)
        control_port = service_control.get("http_server_port", 38888)
        return f"http://{control_host}:{int(control_port)}"

    master_host = deployment.get("master_host")
    if not master_host:
        raise ValueError(
            "Non-PD deployment requires deployment.master_host to derive "
            "Benchmark endpoint"
        )

    instances = deployment.get("instances") or []
    master_instances = [
        instance
        for instance in instances
        if str(instance.get("host")) == str(master_host)
    ]
    if not master_instances:
        raise ValueError(
            f"No deployment instance matches master_host {master_host!r}"
        )

    preferred_master = next(
        (
            instance
            for instance in master_instances
            if str(instance.get("role", "")).lower()
            in {"master", "standalone"}
        ),
        master_instances[0],
    )
    if (
        preferred_master.get("local_port") is None
        and (preferred_master.get("service") or {}).get("port") is None
    ):
        raise ValueError(
            f"Master instance {preferred_master.get('name')!r} requires "
            "local_port to derive Benchmark endpoint"
        )
    port = resolve_instance_local_port(preferred_master)
    return f"http://{master_host}:{port}"


def sync_top_level_from_plans(
    task_config: Dict[str, Any],
    plans: List[ServiceLaunchPlan],
) -> None:
    """Fill legacy top-level host/port/device fields from instance plans."""
    if not plans:
        return
    endpoint = get_benchmark_endpoint(task_config, plans)
    host = resolve_master_host(task_config)
    port = int(task_config.get("port", 8800) or 8800)
    if "://" in endpoint:
        host_port = endpoint.split("://", 1)[1]
        if ":" in host_port:
            host, port_s = host_port.split(":", 1)
            port = int(port_s.split("/")[0])
    task_config["host_addr"] = host
    task_config["port"] = port
    runner = next((p for p in plans if p.is_benchmark_runner), plans[0])
    task_config["world_size"] = (
        runner.global_nnodes
        if is_service_routed_deployment(task_config)
        else resolve_global_nnodes(task_config)
    )
    task_config["num_devices"] = len(runner.device_ids) or int(
        task_config.get("num_devices", 1) or 1
    )
    task_config["start_device_id"] = min(runner.device_ids) if runner.device_ids else 0
    if runner.instance_config.get("serving_extensiable_param") is not None:
        task_config["serving_extensiable_param"] = copy.deepcopy(
            runner.instance_config.get("serving_extensiable_param")
        )


def get_benchmark_runtime(runtimes: List[ServiceRuntime]) -> Optional[ServiceRuntime]:
    for runtime in runtimes:
        if runtime.plan.is_benchmark_runner:
            return runtime
    return runtimes[0] if runtimes else None


def run_on_runtimes_parallel(runtimes: List[ServiceRuntime], fn, max_workers: int = 8):
    """Run fn(runtime) for each runtime; collect (runtime, result_or_exc)."""
    results = []
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(runtimes)))) as pool:
        futures = {pool.submit(fn, runtime): runtime for runtime in runtimes}
        for future in as_completed(futures):
            runtime = futures[future]
            try:
                results.append((runtime, future.result(), None))
            except Exception as exc:
                results.append((runtime, None, exc))
    return results


def cleanup_service_group(
    runtimes: List[ServiceRuntime],
    should_clean: bool,
    stop_timeout: int = 30,
) -> None:
    if not should_clean:
        return

    def _cleanup_one(runtime):
        if runtime.container_manager is not None:
            runtime.container_manager.cleanup(stop_timeout=stop_timeout)

    # Container stop can spend its whole timeout waiting for a process that is
    # already gone.  Cleanup is independent per host, so do it concurrently.
    outcomes = run_on_runtimes_parallel(runtimes, _cleanup_one)
    for runtime, _result, exc in outcomes:
        if exc is not None:
            logging.error(
                "Failed to cleanup instance %s on %s: %s",
                runtime.plan.instance_name,
                runtime.plan.host,
                exc,
            )
