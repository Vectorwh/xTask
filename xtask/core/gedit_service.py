"""Self-service mode for gedit (Qwen-Image-Edit) decoupled deployments.

The gedit two-stage protocol (Text Encoder + DiT, optionally an LLM judge)
needs *multiple* independently configured xLLM services for one case.  The
generic scheduler only manages one distributed runtime per case, so this
module adds a minimal multi-service lifecycle that is deliberately gated on
``deployment.encoder`` / ``deployment.dit`` / ``deployment.judge`` appearing
in the task YAML (sibling keys of ``deployment.benchmark`` / ``pd_control``
/ ``service_control``):

    deployment:
      encoder:
        device_ids: [7]
        port: 18007
        health_path: /health
        startup_environment: [...]
        serving_extensiable_param: {...}
      dit:
        device_ids: [0, 1, 2, 3, 4, 5]
        port: 19000          # start port; consecutive ports per rank
        master_node_addr: 127.0.0.1:19757
        health_path: /health
        serving_extensiable_param: {...}
      judge:                                                # optional
        ...

Activation rules (zero impact on other cases):
  * ``benchmark_type == "image_edit"`` (case 36) or
    ``benchmark_type == "aisbench"`` with an ``api_type`` starting with
    ``gedit`` (case 37), *and*
  * ``deployment`` carries at least one of encoder/dit/judge, *and*
  * ``external_service`` is off (the external mode keeps its legacy path).

The services are launched from the same ``pkg/xllm`` workspace closure the
scheduler's ``download_xllm.sh`` installs, reusing the standard engine image
and the host mounts already configured in ``config/xTask_config.json``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Default host-side directory of the xllm binary used by gedit services.
# The YAML may override per-service with ``xllm_binary``.  Empty default
# means "resolve at container runtime from the pip-installed xllm wheel"
# (site-packages/xllm/xllm), which is the same artifact every normal case
# installs via tools/download_xllm.sh; the container prep sequence runs
# that script before the gedit startup scripts.
def _default_xllm_binary() -> str:
    return os.path.join(os.getcwd(), "tmp", "gedit_svc", "xllm")

# Shell snippet resolving the wheel-installed binary at container runtime.
# Falls back to the legacy workspace copy when the wheel is not installed
# (environments that intentionally run gedit without download_xllm.sh).
_WHEEL_BINARY_RESOLVER = (
    'XLLM_BIN="$(python3 -c \'import xllm, pathlib; '
    'print(pathlib.Path(xllm.__file__).parent / "xllm")\' 2>/dev/null || true)"\n'
    'if [ -z "$XLLM_BIN" ] || [ ! -x "$XLLM_BIN" ]; then\n'
    '  XLLM_BIN="{legacy}"\n'
    'fi\n'
    'if [ -z "$XLLM_BIN" ] || [ ! -x "$XLLM_BIN" ]; then\n'
    '  echo "gedit {name}: xllm binary not found (wheel not installed and '
    '{legacy} missing); run tools/download_xllm.sh first" >&2\n'
    '  exit 1\n'
    'fi'
)

# Extra bind mounts the gedit containers need.  Without these the NPU
# runtime fails aclInit with error 507008 / core dump (see case notes).
# ``volumes`` from the YAML is merged on top of these defaults.
DEFAULT_SERVICE_VOLUMES = {
    "/etc/hccn.conf": "/etc/hccn.conf",
    "/usr/local/Ascend/driver": "/usr/local/Ascend/driver",
    "/usr/local/Ascend/add-ons/": "/usr/local/Ascend/add-ons/",
    "/usr/local/sbin/npu-smi": "/usr/local/sbin/npu-smi",
    "/var/log/npu/slog": "/var/log/npu/slog",
}

# vendors custom_xllm_math is required by the xllm binary; the host path is
# resolved relative to the xTask workspace (pkg/xllm/vendors/custom_xllm_math).
VENDORS_MOUNT_TARGET = (
    "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_xllm_math"
)

# Ascend env sourced inside the startup script (mirrors the templates).
ASCEND_ENV_SOURCES = (
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
    "source /usr/local/Ascend/nnal/atb/set_env.sh",
)

# The vendors custom_xllm_math bundle exports ASCEND_CUSTOM_OPP_PATH and
# appends its op_api/lib (libcust_opapi.so) to LD_LIBRARY_PATH; the engine
# image ships neither, so both xllm linking and custom-op loading depend
# on sourcing this after the base Ascend env.
VENDORS_ENV_SOURCE = (
    f"source {VENDORS_MOUNT_TARGET}/bin/set_env.bash"
)

# Runtime shared libraries the xllm binary links against but which the
# engine image does not ship (libetcd_wrapper.so, libasio.so,
# liburing.so.2, ...).  They live in the xLLM build tree inside the
# workspace; without them on LD_LIBRARY_PATH the binary dies instantly
# with "error while loading shared libraries: libetcd_wrapper.so" (seen
# on the first self-service run of case 36).
def _default_runtime_lib_dirs() -> List[str]:
    return [
        os.path.join(
            os.getcwd(),
            "xllm",
            "build",
            "lib.linux-aarch64-cpython-312",
            "xllm",
        )
    ]

# Default environment shared by gedit xLLM services (DiT/encoder/judge).
DEFAULT_STARTUP_ENVIRONMENT = [
    "export ASDOPS_LOG_TO_STDOUT=1",
    "export ASDOPS_LOG_LEVEL=ERROR",
    "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
    "export NPU_MEMORY_FRACTION=0.98",
    "export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3",
    "export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1",
    "export OMP_NUM_THREADS=12",
    "export HCCL_CONNECT_TIMEOUT=7200",
    "export INF_NAN_MODE_ENABLE=0",
    "export INF_NAN_MODE_FORCE_DISABLE=1",
]


def _is_gedit_benchmark(task_config: Optional[Dict[str, Any]]) -> bool:
    """True only for the gedit benchmark types this module serves."""
    if not isinstance(task_config, dict):
        return False
    benchmark = task_config.get("benchmark") or {}
    if not isinstance(benchmark, dict):
        return False
    benchmark_type = str(benchmark.get("benchmark_type", "")).lower()
    if benchmark_type == "image_edit":
        return True
    if benchmark_type == "aisbench":
        # gedit accuracy uses api_type gedit_image_edit; other aisbench
        # cases (sharegpt, swebench...) never carry gedit services.
        acc_cfg = benchmark.get("acc") or {}
        api_type = str((acc_cfg or {}).get("api_type", "")).lower()
        if api_type.startswith("gedit"):
            return True
    return False


# deployment sibling keys this module consumes (encoder/dit/judge).  They sit
# next to deployment.benchmark / pd_control / service_control and stay fully
# ignored by the multi-host service-group path, which only looks at
# instances/pd_instances/service_instances/mode.
GEDIT_SERVICE_KEYS = ("encoder", "dit", "judge")


def _services_block(task_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    deployment = (task_config or {}).get("deployment") or {}
    if not isinstance(deployment, dict):
        return {}
    services: Dict[str, Any] = {}
    for key in GEDIT_SERVICE_KEYS:
        value = deployment.get(key)
        if isinstance(value, dict):
            services[key] = value
    return services


def is_gedit_services_enabled(task_config: Optional[Dict[str, Any]]) -> bool:
    """Return True when this case asks xTask to self-start gedit services.

    Strictly limited to gedit benchmark shapes so every other
    ``benchmark_type`` is a no-op even if a stray key appears.
    """
    if not _is_gedit_benchmark(task_config):
        return False
    if _services_block(task_config) == {}:
        return False
    # external_service keeps its own legacy path (probe-only).
    from xtask.core.external_service import is_external_service_enabled

    return not is_external_service_enabled(task_config)


def _as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


@dataclass
class GeditServiceSpec:
    """One gedit xLLM service (encoder / dit / judge)."""

    name: str
    role: str  # "encoder" | "dit" | "judge" | custom
    model_path: str
    device_ids: List[int]
    ports: List[int]  # per-rank HTTP ports (len == len(device_ids))
    master_node_addr: str
    health_path: str
    serving_extensiable_param: Dict[str, Any]
    startup_environment: List[str]
    xllm_binary: str
    extra_volumes: Dict[str, str]
    readiness_timeout: int = 900
    hccl_if_port: Optional[int] = None
    benchmark_runner: bool = False  # True for the service owning host_addr/port
    runtime_lib_dirs: List[str] = field(default_factory=list)

    @property
    def container_name(self) -> str:
        return f"gedit-{self.name}"

    def start_port(self) -> int:
        return self.ports[0]


def _parse_master_node_addr(raw: Any, default_port: int) -> str:
    text = str(raw or "").strip()
    if not text:
        return f"127.0.0.1:{default_port}"
    if "://" in text:
        text = text.split("://", 1)[1]
    if ":" not in text:
        text = f"{text}:{default_port}"
    return text


def build_gedit_service_specs(
    task_config: Dict[str, Any],
) -> List[GeditServiceSpec]:
    """Build launch specs from ``deployment.encoder/dit/judge``.

    Field resolution per service (defaults keep the verified manual setup):
      * model_path: service value, else
        ``<task model_path>/text_encoder`` for encoder,
        ``<task model_path>`` for dit/judge with their own
        ``model_path`` override (judge uses an absolute path).
      * device_ids: required per service (list of ints).
      * port: single int; one xLLM process per device with consecutive
        ports by rank (DiT 19000 -> 19000..19005).
      * master_node_addr: per-service (DiT cluster rendezvous differs from
        the encoder's).
      * serving_extensiable_param: per-service dict passed as xLLM flags.
      * startup_environment: per-service shell export lines.
      * health_path: default ``/health``.
      * xllm_binary: unset by default; the startup script resolves the
        pip-installed wheel binary (site-packages/xllm/xllm) with the
        legacy ``tmp/gedit_svc/xllm`` workspace copy as fallback.  An
        explicit value pins a fixed binary path.
    """
    services = _services_block(task_config)
    if not services:
        raise ValueError("deployment.encoder/dit/judge is empty")

    task_model_path = str(task_config.get("model_path") or "")

    # Validate no unknown service roles sneak in (encoder/dit/judge or custom
    # names are allowed; only structural checks below matter).
    specs: List[GeditServiceSpec] = []
    seen_ports: Dict[int, str] = {}
    seen_devices: Dict[int, str] = {}

    # The benchmark runner endpoint host/port is taken from the ``dit``
    # service (or the service marked ``benchmark_runner: true``).
    runner_name = None

    for name, raw_cfg in services.items():
        if not isinstance(raw_cfg, dict):
            raise ValueError(
                f"deployment.{name} must be a mapping"
            )
        role = str(raw_cfg.get("role") or name).lower()

        # ---- model path ----
        model_path = str(raw_cfg.get("model_path") or "").strip()
        if not model_path:
            if role == "encoder":
                if not task_model_path:
                    raise ValueError(
                        f"gedit service {name!r} requires model_path "
                        "(encoder uses <model_path>/text_encoder)"
                    )
                model_path = os.path.join(task_model_path, "text_encoder")
            elif role in ("dit",):
                model_path = task_model_path
            else:
                raise ValueError(
                    f"gedit service {name!r} (role {role!r}) requires "
                    "model_path"
                )

        # ---- devices ----
        device_ids = _as_int_list(raw_cfg.get("device_ids"))
        if not device_ids:
            raise ValueError(
                f"gedit service {name!r} requires device_ids (list of NPU ids)"
            )
        for device_id in device_ids:
            if device_id < 0:
                raise ValueError(
                    f"gedit service {name!r} has negative device id {device_id}"
                )
            owner = seen_devices.get(device_id)
            if owner is not None:
                raise ValueError(
                    f"Device {device_id} is assigned to both {owner} and "
                    f"{name} in deployment services"
                )
            seen_devices[device_id] = name

        # ---- ports ----
        if raw_cfg.get("ports") is not None:
            raise ValueError(
                f"deployment.{name}: the per-rank ``ports`` list is no "
                "longer supported; use the single ``port`` (start port, "
                "consecutive ports are assigned per rank)"
            )
        if raw_cfg.get("port") is None:
            raise ValueError(
                f"deployment.{name} requires port (start port; one port "
                "per rank is assigned consecutively)"
            )
        base = int(raw_cfg["port"])
        # One xLLM process per device: consecutive ports by default
        # (encoder single card -> [port]; DiT 6 cards -> 19000..19005).
        ports = [base + i for i in range(len(device_ids))]
        for port in ports:
            if not 1 <= port <= 65535:
                raise ValueError(
                    f"deployment.{name} has invalid port {port}"
                )
            owner = seen_ports.get(port)
            if owner is not None:
                raise ValueError(
                    f"Port {port} is used by both {owner} and {name} in "
                    "deployment services"
                )
            seen_ports[port] = name

        # ---- master_node_addr ----
        master_node_addr = _parse_master_node_addr(
            raw_cfg.get("master_node_addr"), ports[0]
        )

        # ---- extensiable params ----
        ext = raw_cfg.get("serving_extensiable_param")
        if ext is not None and not isinstance(ext, dict):
            raise ValueError(
                f"deployment.{name} serving_extensiable_param must be "
                "a mapping"
            )
        ext = dict(ext or {})

        # ---- startup environment ----
        env_cfg = raw_cfg.get("startup_environment")
        if env_cfg is None:
            env_lines = list(DEFAULT_STARTUP_ENVIRONMENT)
        else:
            env_lines = _as_str_list(env_cfg)

        # ---- xllm binary ----
        # Unset means "resolve from the pip-installed wheel at container
        # runtime"; an explicit value pins a fixed binary path (legacy envs
        # that still rely on tmp/gedit_svc/xllm can set it there).
        xllm_binary = str(raw_cfg.get("xllm_binary") or "").strip()

        # ---- extra volumes ----
        extra_volumes_cfg = raw_cfg.get("volumes")
        if extra_volumes_cfg is not None and not isinstance(extra_volumes_cfg, dict):
            raise ValueError(
                f"deployment.{name} volumes must be a mapping "
                "(host_path: container_path)"
            )
        extra_volumes = dict(extra_volumes_cfg or {})

        hccl_if_port = raw_cfg.get("hccl_if_port")
        if hccl_if_port is not None:
            hccl_if_port = int(hccl_if_port)

        readiness_timeout = int(raw_cfg.get("readiness_timeout") or 900)

        # Runtime library directories (libetcd_wrapper.so & friends).
        # YAML list wins; default points at the workspace xllm build tree.
        lib_dirs_cfg = raw_cfg.get("runtime_lib_dirs")
        if lib_dirs_cfg is not None:
            runtime_lib_dirs = _as_str_list(lib_dirs_cfg)
        else:
            runtime_lib_dirs = _default_runtime_lib_dirs()

        if raw_cfg.get("benchmark_runner"):
            runner_name = name

        specs.append(
            GeditServiceSpec(
                name=name,
                role=role,
                model_path=model_path,
                device_ids=device_ids,
                ports=ports,
                master_node_addr=master_node_addr,
                health_path=str(raw_cfg.get("health_path") or "/health"),
                serving_extensiable_param=ext,
                startup_environment=env_lines,
                xllm_binary=xllm_binary,
                extra_volumes=extra_volumes,
                readiness_timeout=readiness_timeout,
                hccl_if_port=hccl_if_port,
                runtime_lib_dirs=runtime_lib_dirs,
            )
        )

    if not specs:
        raise ValueError("deployment.encoder/dit/judge produced no services")

    # NOTE on HCCL_IF_BASE_PORT spacing (learned the hard way on case 37):
    # bases must be spaced so no service's base lands inside another
    # service's NIC-channel window.  judge base 43442 vs DiT base 43432
    # corrupted DiT's first collective with aicpu ERR02005 / acl 507018;
    # the verified spacing is DiT 43432 / encoder 43438 / judge 43500.
    # The exact window width is xllm-version dependent, so no hard
    # validation is enforced here — keep bases >= 60 ports apart in YAML.

    # Default benchmark endpoint owner: the ``dit`` service when present,
    # else the first service (encoder-only debugging).
    if runner_name is None:
        runner_name = next(
            (spec.name for spec in specs if spec.role == "dit"),
            specs[0].name,
        )
    for spec in specs:
        spec.benchmark_runner = spec.name == runner_name

    return specs


def render_gedit_startup_script(
    spec: GeditServiceSpec,
    output_dir: str,
) -> str:
    """Render one bash startup script for a gedit service.

    The script starts one xLLM process per device (per-rank port / rank /
    device), redirecting logs into ``<output_dir>/log/node_<i>.log``.  It
    mirrors the verified tmp/gedit_svc/start_*.sh scripts.  Model and binary
    paths are mapped into the container path space via the configured
    DOCKER_CONFIG volumes.
    """
    from xtask.utils.container_manager import path_mapping_in_container

    os.makedirs(os.path.join(output_dir, "log"), exist_ok=True)

    model_path = path_mapping_in_container(str(spec.model_path))
    ext_flags = []
    for key, value in spec.serving_extensiable_param.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        ext_flags.append(f"--{key}={value}")
    ext_flags_str = " \\\n  ".join(ext_flags) if ext_flags else ""

    lines: List[str] = []
    lines.append("#!/bin/bash")
    lines.append(f"# gedit service: {spec.name} (role={spec.role})")
    for source_line in ASCEND_ENV_SOURCES:
        lines.append(source_line)
    lines.append(VENDORS_ENV_SOURCE)
    # Engine image lacks libetcd_wrapper.so/libasio.so/liburing.so.2; they
    # ship with the xllm build tree and must precede the process launch.
    for lib_dir in spec.runtime_lib_dirs:
        mapped_dir = path_mapping_in_container(str(lib_dir))
        lines.append(f'export LD_LIBRARY_PATH="{mapped_dir}:$LD_LIBRARY_PATH"')
    for env_line in spec.startup_environment:
        lines.append(env_line)
    if spec.hccl_if_port is not None:
        lines.append(f"export HCCL_IF_BASE_PORT={spec.hccl_if_port}")

    lines.append(f'MODEL_PATH="{model_path}"')
    lines.append(f'NNODES={len(spec.device_ids)}')
    lines.append(f'MASTER_NODE_ADDR="{spec.master_node_addr}"')
    if spec.xllm_binary:
        # Explicit ``xllm_binary`` in the case yaml: fixed path, fail fast
        # if it is missing.
        binary = path_mapping_in_container(str(spec.xllm_binary))
        lines.append(f'XLLM_BIN="{binary}"')
        lines.append(
            'if [ ! -x "$XLLM_BIN" ]; then\n'
            f'  echo "gedit {spec.name}: xllm binary not found: $XLLM_BIN" >&2\n'
            "  exit 1\n"
            "fi"
        )
    else:
        # Default: resolve from the pip-installed wheel first (same artifact
        # every normal case installs via download_xllm.sh), fall back to the
        # legacy workspace copy.
        legacy = path_mapping_in_container(_default_xllm_binary())
        lines.append(
            _WHEEL_BINARY_RESOLVER.format(legacy=legacy, name=spec.name)
        )
    lines.append('LOG_DIR="log"')
    lines.append("mkdir -p $LOG_DIR")

    for rank, (device_id, port) in enumerate(zip(spec.device_ids, spec.ports)):
        # ASCEND_RT_VISIBLE_DEVICES selects the NPU; xllm itself has no
        # --devices flag (verified in tmp/gedit_svc/run_dit.sh).
        command = (
            f"ASCEND_RT_VISIBLE_DEVICES={device_id} "
            '"$XLLM_BIN" --model "$MODEL_PATH" '
            f"--port {port} "
            '--master_node_addr="$MASTER_NODE_ADDR" '
            f"--nnodes={len(spec.device_ids)} "
            f"--node_rank={rank} "
            '--communication_backend=hccl'
        )
        if ext_flags_str:
            command += f" \\\n  {ext_flags_str}"
        command += f" > $LOG_DIR/node_{rank}.log 2>&1 &"
        lines.append(command)

    lines.append(
        f'echo "gedit service {spec.name}: {len(spec.device_ids)} node(s) '
        'launched."'
    )

    script_path = os.path.join(output_dir, f"start_gedit_{spec.name}.sh")
    with open(script_path, "w", encoding="utf-8") as script_file:
        script_file.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def gedit_health_url(spec: GeditServiceSpec) -> str:
    return f"http://127.0.0.1:{spec.ports[0]}{spec.health_path}"


def gedit_runner_origin(specs: List[GeditServiceSpec]) -> str:
    """HTTP origin of the service that owns the benchmark endpoint."""
    for spec in specs:
        if spec.benchmark_runner:
            return f"http://127.0.0.1:{spec.ports[0]}"
    return f"http://127.0.0.1:{specs[0].ports[0]}"


def gedit_encoder_url(specs: List[GeditServiceSpec]) -> Optional[str]:
    for spec in specs:
        if spec.role == "encoder":
            return f"http://127.0.0.1:{spec.ports[0]}"
    return None


def gedit_judge_endpoint(specs: List[GeditServiceSpec]) -> Optional[str]:
    for spec in specs:
        if spec.role == "judge":
            return f"127.0.0.1:{spec.ports[0]}"
    return None


def apply_gedit_services_to_task_config(
    task_config: Dict[str, Any],
    specs: List[GeditServiceSpec],
) -> None:
    """Point benchmark addressing fields at the self-started services.

    Only the fields the gedit generators already consume are touched:
      * host_addr/port -> benchmark runner (DiT) origin;
      * benchmark.perf.encoder_url (case 36) -> encoder origin;
      * benchmark.acc.judge_endpoint (case 37) -> judge endpoint;
      * benchmark.acc generation_kwargs.encoder_url -> encoder origin.

    ``deployment.xllm_service.route_gedit_traffic: true`` reroutes the gedit
    benchmark endpoint (and the stage-1 encoder URL) to the xLLM-Service
    router instead of the engine origins, so service-side gaps (missing
    engine registration, unsupported /v1/image/generation route) fail the
    case loudly. The switch is YAML-only: absent or false keeps the direct
    engine path, so xTask behavior never depends on code changes.
    """
    runner = gedit_runner_origin(specs)
    host_port = runner.split("://", 1)[-1]
    host, _, port_str = host_port.rpartition(":")
    task_config["host_addr"] = host
    task_config["port"] = int(port_str)

    deployment = task_config.get("deployment") or {}
    control = deployment.get("xllm_service") or {}
    if control.get("route_gedit_traffic") is True:
        router_host = control.get("host") or deployment.get("master_host") or host
        router_port = int(control.get("http_server_port", 38888) or 38888)
        runner = f"http://{router_host}:{router_port}"
        task_config["host_addr"] = router_host
        task_config["port"] = router_port

    benchmark = task_config.get("benchmark") or {}
    if not isinstance(benchmark, dict):
        return

    encoder_url = gedit_encoder_url(specs)
    if control.get("route_gedit_traffic") is True and encoder_url:
        # Stage-1 embeddings go through the router too: /v1/embeddings is a
        # service-native route, so this isolates the missing engine
        # registration instead of only failing on the generation route.
        encoder_url = runner
    if encoder_url:
        perf_cfg = benchmark.get("perf")
        if isinstance(perf_cfg, dict) and not perf_cfg.get("encoder_url"):
            perf_cfg["encoder_url"] = encoder_url

    judge_endpoint = gedit_judge_endpoint(specs)
    if judge_endpoint:
        acc_cfg = benchmark.get("acc")
        if isinstance(acc_cfg, dict):
            if not acc_cfg.get("judge_endpoint"):
                acc_cfg["judge_endpoint"] = judge_endpoint

    # Keep model_config generation_kwargs encoder_url in sync when the YAML
    # didn't pin it explicitly (gedit_image_edit.py default matches anyway).
    if encoder_url:
        acc_cfg = benchmark.get("acc")
        if isinstance(acc_cfg, dict):
            generation_kwargs = acc_cfg.get("generation_kwargs")
            if isinstance(generation_kwargs, dict):
                if not generation_kwargs.get("encoder_url"):
                    generation_kwargs["encoder_url"] = encoder_url
