"""Lifecycle management for the xLLM etcd/xllm-service control plane."""
from __future__ import annotations

import logging
import os
import secrets
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Dict

from xtask.core.service_group import ServiceRuntime, normalize_pd_deployment
from xtask.core.xllm_service_config import get_xllm_service_config, service_etcd_location

_RUN_TOKENS = {}
DEFAULT_MASTER_SERVICE_KEY = "XLLM:SERVICE:MASTER"
DEFAULT_SERVICE_DEPLOYMENT_ROLE = "COORDINATOR"
DEFAULT_MASTER_KEY_WAIT_SECONDS = 30
DEFAULT_SERVICE_BINARY_PROFILE = "auto"
SERVICE_BINARY_PROFILE_LEGACY = "legacy"
SERVICE_BINARY_PROFILE_REFACTOR = "refactor"
SERVICE_BINARY_PROFILES = frozenset(
    {DEFAULT_SERVICE_BINARY_PROFILE, SERVICE_BINARY_PROFILE_LEGACY, SERVICE_BINARY_PROFILE_REFACTOR}
)


@dataclass(frozen=True)
class XllmServiceBinaryCapabilities:
    """Runtime capabilities of the installed xllm_master_serving binary."""

    supports_deployment_role: bool = False
    supports_readyz: bool = False
    # Whether the control-plane anchor (XLLM:SERVICE:MASTER / /readyz) must be
    # present before xTask declares the control plane healthy. The refactor
    # binary defaults to ROUTER and may never elect a master, so a hard anchor
    # gate is required to catch that regression. The legacy binary self-elects
    # and writes MASTER with a short (3s) lease on startup; gating hard on that
    # lease is fragile, so the anchor check is advisory for legacy binaries.
    requires_control_plane_anchor: bool = False


def resolve_internal_api_token(control: Dict[str, Any]) -> str:
    """Resolve once per run; generated credentials stay in private memory only."""
    from xtask.utils.secrets import register_secret

    token = str(control.get("internal_api_token") or "").strip()
    if not token:
        token = str(os.environ.get("XLLM_INTERNAL_TOKEN", "")).strip()
    if not token:
        cached = _RUN_TOKENS.get(id(control))
        token = cached[1] if cached and cached[0] is control else None
    if not token:
        token = secrets.token_urlsafe(32)
        _RUN_TOKENS[id(control)] = (control, token)
    if not token.isascii() or any(ord(char) < 32 or ord(char) > 126 for char in token):
        raise ValueError("internal_api_token must be non-empty printable ASCII")
    register_secret(token)
    return token


def begin_service_run(task_config):
    """Rotate only generated credentials; explicit YAML/env overrides win."""
    control = get_xllm_service_config(task_config)
    _RUN_TOKENS.pop(id(control), None)


@dataclass(frozen=True)
class XllmControlConfig:
    host: str
    etcd_client_port: int = 3389
    etcd_peer_port: int = 3390
    http_server_port: int = 38888
    rpc_server_port: int = 38889
    startup_attempts: int = 2
    startup_wait_seconds: int = 3
    no_proxy_hosts: tuple[str, ...] = ()
    output_subdir: str = "pd_control"
    internal_api_token: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    service_deployment_role: str = DEFAULT_SERVICE_DEPLOYMENT_ROLE
    master_service_key: str = DEFAULT_MASTER_SERVICE_KEY
    master_key_wait_seconds: int = DEFAULT_MASTER_KEY_WAIT_SECONDS
    service_binary_profile: str = DEFAULT_SERVICE_BINARY_PROFILE


def _resolve_control_plane_options(control: Dict[str, Any]) -> Dict[str, Any]:
    role = str(control.get("service_deployment_role", DEFAULT_SERVICE_DEPLOYMENT_ROLE)).strip()
    master_key = str(control.get("master_service_key", DEFAULT_MASTER_SERVICE_KEY)).strip()
    profile = str(
        control.get("service_binary_profile", DEFAULT_SERVICE_BINARY_PROFILE)
    ).strip().lower()
    if profile not in SERVICE_BINARY_PROFILES:
        raise ValueError(
            "service_binary_profile must be one of: auto, legacy, refactor; "
            f"got '{profile}'"
        )
    return {
        "service_deployment_role": role or DEFAULT_SERVICE_DEPLOYMENT_ROLE,
        "master_service_key": master_key or DEFAULT_MASTER_SERVICE_KEY,
        "master_key_wait_seconds": int(
            control.get("master_key_wait_seconds", DEFAULT_MASTER_KEY_WAIT_SECONDS)
        ),
        "service_binary_profile": profile,
    }


def is_pd_deployment(task_config: Dict[str, Any]) -> bool:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    if deployment.get("mode") == "pd_disaggregated":
        return True
    roles = {str(item.get("role", "")).lower() for item in deployment.get("instances") or []}
    return bool(roles & {"prefill", "decode"})


def is_service_routed_deployment(task_config: Dict[str, Any]) -> bool:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    return deployment.get("mode") == "service_routed"


def _build_control_config(
    task_config: Dict[str, Any], output_subdir: str
) -> XllmControlConfig:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    control = get_xllm_service_config(task_config)
    if not control:
        control = deployment.setdefault("xllm_service", {})
        if control is None:
            control = deployment["xllm_service"] = {}
    host, etcd_port = service_etcd_location(task_config)
    if not host:
        raise ValueError(
            "xLLM-Service requires deployment.xllm_service.host or "
            "deployment.master_host"
        )
    host = str(host)
    instance_hosts = [
        str(instance.get("host"))
        for instance in deployment.get("instances") or []
        if instance.get("host")
    ]
    no_proxy_hosts = tuple(
        dict.fromkeys(
            ["localhost", "127.0.0.1", "0.0.0.0", host]
            + instance_hosts
            + [str(item) for item in control.get("no_proxy_hosts") or []]
        )
    )
    return XllmControlConfig(
        host=host,
        etcd_client_port=etcd_port,
        etcd_peer_port=int(
            (control.get("etcd") or {}).get("peer_port", control.get("etcd_peer_port", 3390))
        ),
        http_server_port=int(control.get("http_server_port", 38888)),
        rpc_server_port=int(control.get("rpc_server_port", 38889)),
        startup_attempts=int(control.get("startup_attempts", 2)),
        startup_wait_seconds=int(control.get("startup_wait_seconds", 3)),
        no_proxy_hosts=no_proxy_hosts,
        # Retain historical log paths and the PD-specific response environment.
        output_subdir=output_subdir,
        internal_api_token=resolve_internal_api_token(control),
        **_resolve_control_plane_options(control),
    )


def build_xllm_service_config(task_config: Dict[str, Any]) -> XllmControlConfig:
    """Parse the shared Service block without changing deployment behavior."""
    output_subdir = "pd_control" if is_pd_deployment(task_config) else "service_control"
    return _build_control_config(task_config, output_subdir)


def build_pd_control_config(task_config: Dict[str, Any]) -> XllmControlConfig:
    """Compatibility entry point; configuration lives in xllm_service."""
    return _build_control_config(task_config, "pd_control")


def build_service_control_config(task_config: Dict[str, Any]) -> XllmControlConfig:
    """Compatibility entry point; configuration lives in xllm_service."""
    return _build_control_config(task_config, "service_control")


class XllmControlService:
    """Start and validate etcd plus xllm_master_serving in one runtime."""

    def __init__(self, runtime: ServiceRuntime, config: XllmControlConfig, model_path: str, output_dir: str) -> None:
        self.runtime = runtime
        self.config = config
        from xtask.utils.secrets import register_secret
        register_secret(config.internal_api_token)
        self.model_path = model_path
        from xtask.utils.container_manager import path_mapping_in_container

        self.output_dir = os.path.join(
            path_mapping_in_container(output_dir),
            config.output_subdir,
        )
        self._binary_capabilities: XllmServiceBinaryCapabilities | None = None
        self._managed_etcd = None
        self._managed_service = None

    def start_with_retry(self) -> None:
        last_error = "unknown error"
        for attempt in range(1, self.config.startup_attempts + 1):
            self.stop()
            try:
                self._start_owned_processes()
                time.sleep(self.config.startup_wait_seconds)
                healthy, detail = self._check_health()
                if healthy:
                    logging.info("xLLM control services are healthy on %s", self.config.host)
                    return
                last_error = detail
            except RuntimeError as exc:
                last_error = str(exc)
            logging.warning(
                "xLLM control startup attempt %d/%d failed: %s",
                attempt,
                self.config.startup_attempts,
                last_error,
            )
        self.stop()
        raise RuntimeError(f"xLLM control startup failed: {last_error}")

    def _start_owned_processes(self):
        """Legacy binary behavior, with fenced per-run process ownership."""
        from xtask.core.xllm_service_cluster import ManagedServiceProcess
        if self._managed_etcd is None:
            directory = self.output_dir + "/owned-" + secrets.token_hex(12)
            self._managed_etcd = ManagedServiceProcess(self.runtime, directory + "/etcd", "etcd.log")
            self._managed_service = ManagedServiceProcess(self.runtime, directory + "/service", "xservice.log")
        cfg = self.config
        environment = {"no_proxy": ",".join(cfg.no_proxy_hosts), "NO_PROXY": ",".join(cfg.no_proxy_hosts)}
        if cfg.output_subdir == "pd_control":
            environment["ENABLE_DECODE_RESPONSE_TO_SERVICE"] = "true"
        check = self._managed_etcd.execute(self._managed_etcd._helper("check"))
        if not check.success:
            raise RuntimeError("Safe process ownership helper/pidfd unavailable")
        self._managed_etcd.start([
            "etcd", "--data-dir", self._managed_etcd.directory + "/data",
            "--listen-peer-urls", "http://0.0.0.0:%s" % cfg.etcd_peer_port,
            "--listen-client-urls", "http://0.0.0.0:%s" % cfg.etcd_client_port,
            "--advertise-client-urls", "http://%s:%s" % (cfg.host, cfg.etcd_client_port),
            "--initial-advertise-peer-urls", "http://%s:%s" % (cfg.host, cfg.etcd_peer_port),
            "--initial-cluster", "xtask-etcd=http://%s:%s" % (cfg.host, cfg.etcd_peer_port),
            "--name", "xtask-etcd",
        ], environment)
        time.sleep(1)
        argv = ["xllm_master_serving", "--etcd_addr=%s:%s" % (cfg.host, cfg.etcd_client_port),
                "--http_server_port=%s" % cfg.http_server_port, "--rpc_server_port=%s" % cfg.rpc_server_port,
                "--internal_api_token=" + cfg.internal_api_token, "--tokenizer_path=" + self.model_path]
        if self._resolve_binary_capabilities().supports_deployment_role:
            argv.append("--service_deployment_role=" + cfg.service_deployment_role)
        self._managed_service.start(argv, environment)
        # Preserve historical top-level log names without erasing etcd data.
        links = self._managed_service.execute(
            "mkdir -p " + shlex.quote(self.output_dir) + "; ln -sfn "
            + shlex.quote(self._managed_etcd.log_file) + " " + shlex.quote(self.output_dir + "/etcd.log")
            + "; ln -sfn " + shlex.quote(self._managed_service.log_file) + " "
            + shlex.quote(self.output_dir + "/xservice.log"))
        if not links.success:
            raise RuntimeError("Unable to preserve owned Service log links")

    def stop(self) -> None:
        errors = []
        for process in (self._managed_service, self._managed_etcd):
            if process is not None:
                try:
                    process.stop()
                except RuntimeError as exc:
                    errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    def _resolve_binary_capabilities(self) -> XllmServiceBinaryCapabilities:
        if self._binary_capabilities is not None:
            return self._binary_capabilities

        profile = self.config.service_binary_profile
        if profile == SERVICE_BINARY_PROFILE_LEGACY:
            capabilities = XllmServiceBinaryCapabilities(
                supports_deployment_role=False,
                supports_readyz=False,
                requires_control_plane_anchor=False,
            )
        elif profile == SERVICE_BINARY_PROFILE_REFACTOR:
            capabilities = XllmServiceBinaryCapabilities(
                supports_deployment_role=True,
                supports_readyz=True,
                requires_control_plane_anchor=True,
            )
        else:
            # Probe each capability independently. The binary is gflags-based and
            # prints single-dash flags ("-service_deployment_role"), while the
            # older probe only accepted double-dash, so a capable binary could be
            # classified as legacy and then start as the default ROUTER without
            # ever electing a master. Matching on the bare flag name accepts both
            # dash styles.
            deployment_role_result = self._exec(
                "xllm_master_serving --help 2>&1 | grep -q 'service_deployment_role'",
                timeout=15,
            )
            # /readyz is an HTTP route compiled into the binary, not a gflags
            # option, so --help never mentions it. Probe the embedded route
            # string instead. The route answers {"status":"ready"} on
            # COORDINATOR and ROUTER alike, so its presence is a property of the
            # build rather than of the role.
            readyz_result = self._exec(
                "grep -aq 'readyz' \"$(command -v xllm_master_serving)\"",
                timeout=15,
            )
            capabilities = XllmServiceBinaryCapabilities(
                supports_deployment_role=deployment_role_result.success,
                supports_readyz=readyz_result.success,
                # Without the role flag xTask cannot force COORDINATOR, so the
                # binary may run as ROUTER and never publish the master key; only
                # then must a missing anchor block startup.
                requires_control_plane_anchor=deployment_role_result.success,
            )

        self._binary_capabilities = capabilities
        logging.info(
            "xLLM service binary profile=%s capabilities: deployment_role=%s "
            "readyz=%s anchor=%s",
            profile,
            capabilities.supports_deployment_role,
            capabilities.supports_readyz,
            capabilities.requires_control_plane_anchor,
        )
        return capabilities

    def _build_start_command(self) -> str:
        cfg = self.config
        control_dir = shlex.quote(self.output_dir)
        tokenizer_path = shlex.quote(self.model_path)
        from xtask.utils.secrets import register_secret
        register_secret(cfg.internal_api_token)
        internal_api_token = shlex.quote(cfg.internal_api_token)
        no_proxy = ",".join(cfg.no_proxy_hosts)
        decode_response_env = ""
        if self.config.output_subdir == "pd_control":
            decode_response_env = "ENABLE_DECODE_RESPONSE_TO_SERVICE=true "
        capabilities = self._resolve_binary_capabilities()
        deployment_role_flag = ""
        if capabilities.supports_deployment_role:
            deployment_role_flag = (
                f"--service_deployment_role={shlex.quote(cfg.service_deployment_role)} "
            )
        return (
            f"export no_proxy='{no_proxy}' NO_PROXY='{no_proxy}'; "
            f"mkdir -p {control_dir}; "
            f"rm -rf {control_dir}/etcd-data; "
            f"nohup etcd --data-dir {control_dir}/etcd-data "
            f"--listen-peer-urls http://0.0.0.0:{cfg.etcd_peer_port} "
            f"--listen-client-urls http://0.0.0.0:{cfg.etcd_client_port} "
            f"--advertise-client-urls http://{cfg.host}:{cfg.etcd_client_port} "
            f"--name my-etcd-1 > {control_dir}/etcd.log 2>&1 </dev/null & sleep 1; "
            f"{decode_response_env}nohup xllm_master_serving "
            f"--etcd_addr={cfg.host}:{cfg.etcd_client_port} "
            f"--http_server_port {cfg.http_server_port} --rpc_server_port {cfg.rpc_server_port} "
            f"{deployment_role_flag}"
            f"--internal_api_token={internal_api_token} "
            f"--tokenizer_path={tokenizer_path} > {control_dir}/xservice.log 2>&1 </dev/null &"
        )

    def _build_basic_health_command(self) -> str:
        cfg = self.config
        return (
            "pgrep -f '[e]tcd' >/dev/null && pgrep -f '[x]llm_master_serving' >/dev/null && "
            f"curl --noproxy '*' -fsS http://{cfg.host}:{cfg.etcd_client_port}/health >/dev/null && "
            f"! grep -Eiq 'fatal|panic|segmentation fault|check failed|address already in use' "
            f"{shlex.quote(self.output_dir)}/xservice.log"
        )

    def _build_master_key_probe_command(self) -> str:
        cfg = self.config
        master_key = shlex.quote(cfg.master_service_key)
        endpoint = f"http://{cfg.host}:{cfg.etcd_client_port}"
        endpoint_q = shlex.quote(endpoint)
        # Primary: etcdctl over gRPC (same transport xLLM engines use to
        # discover the master), which avoids HTTP-proxy / REST-gateway quirks.
        # Fallback: the etcd v3 REST gateway (/v3/kv/range) via curl, in case
        # etcdctl is not installed alongside etcd.
        return (
            f"ETCDCTL_API=3 etcdctl --endpoints={endpoint_q} "
            f"--dial-timeout=2s --command-timeout=5s get {master_key} "
            f"--print-value-only 2>/dev/null | grep -q . || ("
            f"KEY_B64=$(printf %s {master_key} | base64 | tr -d '\\n'); "
            f"curl --noproxy '*' -fsS {endpoint}/v3/kv/range "
            "-H 'Content-Type: application/json' "
            "-d \"{\\\"key\\\":\\\"${KEY_B64}\\\"}\" | grep -q '\"kvs\"'"
            f")"
        )

    def _build_readyz_probe_command(self) -> str:
        cfg = self.config
        return (
            f"curl --noproxy '*' -fsS http://{cfg.host}:{cfg.http_server_port}/readyz "
            "| grep -q '\"status\":\"ready\"'"
        )

    def _control_plane_anchor_ready(self) -> tuple[bool, str]:
        """Poll the control-plane anchor (MASTER key / /readyz).

        Returns (ready, detail). ``ready`` is True once the anchor is observed.
        ``detail`` carries the last probe status for logging.
        """
        capabilities = self._resolve_binary_capabilities()
        deadline = time.time() + self.config.master_key_wait_seconds
        master_probe = self._build_master_key_probe_command()
        readyz_probe = self._build_readyz_probe_command()
        last_detail = "no probe attempted"
        while time.time() < deadline:
            master_result = self._exec(master_probe, timeout=10)
            if master_result.success:
                return True, "etcd key XLLM:SERVICE:MASTER present"
            last_detail = (
                master_result.stderr or master_result.stdout
                or "etcd key not found"
            )
            if capabilities.supports_readyz:
                readyz_result = self._exec(readyz_probe, timeout=10)
                if readyz_result.success:
                    return True, "HTTP /readyz status=ready"
                last_detail = (
                    readyz_result.stderr or readyz_result.stdout or last_detail
                )
            time.sleep(1)
        return False, last_detail

    def _check_health(self):
        if self._managed_etcd is not None:
            if not self._managed_etcd.healthy() or not self._managed_service.healthy():
                return False, "Owned control-plane process exited or emitted a fatal error"
        result = self._exec(self._build_basic_health_command(), timeout=30)
        if not result.success:
            logs = self._exec(
                f"tail -n 80 {shlex.quote(self.output_dir)}/etcd.log 2>/dev/null; "
                f"tail -n 80 {shlex.quote(self.output_dir)}/xservice.log 2>/dev/null",
                timeout=30,
            )
            return False, logs.stdout or result.stderr or result.stdout
        capabilities = self._resolve_binary_capabilities()
        anchor_ready, anchor_detail = self._control_plane_anchor_ready()
        ready_hint = (
            " or HTTP /readyz status=ready" if capabilities.supports_readyz else ""
        )
        if not anchor_ready:
            message = (
                f"control plane anchor not ready within "
                f"{self.config.master_key_wait_seconds}s: expected etcd key "
                f"{self.config.master_service_key}{ready_hint} ({anchor_detail})"
            )
            # The refactor binary may default to ROUTER and never elect a
            # master, so a missing anchor is a real failure there. The legacy
            # binary self-elects with a short lease; the anchor probe is timing
            # sensitive, so we only warn and rely on the basic health check
            # (process + etcd /health + no fatal log) that historically worked.
            if capabilities.requires_control_plane_anchor:
                return False, message
            logging.warning(
                "xLLM control plane anchor not confirmed (legacy profile, "
                "non-blocking): %s", message
            )
        return True, "ok"

    def wait_for_http_endpoint(self, timeout: int = 90) -> bool:
        """Poll the xllm_master_serving HTTP port until it opens.

        The endpoint may start only after the configured instances have
        registered with etcd, so this is called after instance readiness.
        """
        cfg = self.config
        url = f"http://{cfg.host}:{cfg.http_server_port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self._exec(
                f"curl --noproxy '*' -fsS {url} >/dev/null",
                timeout=10,
            )
            if result.success:
                return True
            time.sleep(1)
        return False

    def _exec(self, command: str, timeout: int):
        return self.runtime.container_manager.exec_in_container(self.runtime.container_config.container_name, command, timeout=timeout)


# Existing PD imports remain valid while the implementation is now generic.
PdControlConfig = XllmControlConfig
PdControlService = XllmControlService
