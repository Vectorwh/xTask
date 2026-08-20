"""Lifecycle management for the xLLM etcd/xllm-service control plane."""
from __future__ import annotations

import logging
import os
import shlex
import time
from dataclasses import dataclass
from typing import Any, Dict

from xtask.core.service_group import ServiceRuntime, normalize_pd_deployment


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


def is_pd_deployment(task_config: Dict[str, Any]) -> bool:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    if deployment.get("mode") == "pd_disaggregated":
        return True
    roles = {str(item.get("role", "")).lower() for item in deployment.get("instances") or []}
    return bool(roles & {"prefill", "decode"})


def build_pd_control_config(task_config: Dict[str, Any]) -> XllmControlConfig:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    control = deployment.get("pd_control") or {}
    host = control.get("host") or deployment.get("master_host")
    if not host:
        raise ValueError("PD deployment requires deployment.pd_control.host or master_host")
    instance_hosts = [
        str(instance.get("host"))
        for instance in deployment.get("instances") or []
        if instance.get("host")
    ]
    configured_no_proxy = control.get("no_proxy_hosts") or []
    no_proxy_hosts = tuple(
        dict.fromkeys(
            ["localhost", "127.0.0.1", "0.0.0.0", host]
            + instance_hosts
            + [str(item) for item in configured_no_proxy]
        )
    )
    return XllmControlConfig(
        host=host,
        etcd_client_port=int(control.get("etcd_client_port", 3389)),
        etcd_peer_port=int(control.get("etcd_peer_port", 3390)),
        http_server_port=int(control.get("http_server_port", 38888)),
        rpc_server_port=int(control.get("rpc_server_port", 38889)),
        startup_attempts=int(control.get("startup_attempts", 2)),
        startup_wait_seconds=int(control.get("startup_wait_seconds", 3)),
        no_proxy_hosts=no_proxy_hosts,
        output_subdir="pd_control",
    )


def is_service_routed_deployment(task_config: Dict[str, Any]) -> bool:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    return deployment.get("mode") == "service_routed"


def build_service_control_config(task_config: Dict[str, Any]) -> XllmControlConfig:
    """Build the shared control-plane config for ordinary routed instances."""
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    control = deployment.get("service_control") or {}
    host = control.get("host") or deployment.get("master_host")
    if not host:
        raise ValueError(
            "service_routed deployment requires deployment.service_control.host "
            "or master_host"
        )
    instance_hosts = [
        str(instance.get("host"))
        for instance in deployment.get("instances") or []
        if instance.get("host")
    ]
    configured_no_proxy = control.get("no_proxy_hosts") or []
    no_proxy_hosts = tuple(
        dict.fromkeys(
            ["localhost", "127.0.0.1", "0.0.0.0", str(host)]
            + instance_hosts
            + [str(item) for item in configured_no_proxy]
        )
    )
    return XllmControlConfig(
        host=str(host),
        etcd_client_port=int(control.get("etcd_client_port", 3389)),
        etcd_peer_port=int(control.get("etcd_peer_port", 3390)),
        http_server_port=int(control.get("http_server_port", 38888)),
        rpc_server_port=int(control.get("rpc_server_port", 38889)),
        startup_attempts=int(control.get("startup_attempts", 2)),
        startup_wait_seconds=int(control.get("startup_wait_seconds", 3)),
        no_proxy_hosts=no_proxy_hosts,
        output_subdir="service_control",
    )


class XllmControlService:
    """Start and validate etcd plus xllm_master_serving in one runtime."""

    def __init__(self, runtime: ServiceRuntime, config: XllmControlConfig, model_path: str, output_dir: str) -> None:
        self.runtime = runtime
        self.config = config
        self.model_path = model_path
        from xtask.utils.container_manager import path_mapping_in_container

        self.output_dir = os.path.join(
            path_mapping_in_container(output_dir),
            config.output_subdir,
        )

    def start_with_retry(self) -> None:
        last_error = "unknown error"
        for attempt in range(1, self.config.startup_attempts + 1):
            self.stop()
            result = self._exec(self._build_start_command(), timeout=60)
            if result.success:
                time.sleep(self.config.startup_wait_seconds)
                healthy, detail = self._check_health()
                if healthy:
                    logging.info(
                        "xLLM control services are healthy on %s", self.config.host
                    )
                    return
                last_error = detail
            else:
                last_error = result.stderr or result.stdout
            logging.warning(
                "xLLM control startup attempt %d/%d failed: %s",
                attempt,
                self.config.startup_attempts,
                last_error,
            )
        self.stop()
        raise RuntimeError(f"xLLM control startup failed: {last_error}")

    def stop(self) -> None:
        self._exec("pkill -f '[x]llm_master_serving' 2>/dev/null || true; pkill -f '[e]tcd' 2>/dev/null || true", timeout=30)

    def _build_start_command(self) -> str:
        cfg = self.config
        control_dir = shlex.quote(self.output_dir)
        tokenizer_path = shlex.quote(self.model_path)
        no_proxy = ",".join(cfg.no_proxy_hosts)
        decode_response_env = ""
        if self.config.output_subdir == "pd_control":
            decode_response_env = "ENABLE_DECODE_RESPONSE_TO_SERVICE=true "
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
            f"--tokenizer_path={tokenizer_path} > {control_dir}/xservice.log 2>&1 </dev/null &"
        )

    def _check_health(self):
        cfg = self.config
        command = (
            "pgrep -f '[e]tcd' >/dev/null && pgrep -f '[x]llm_master_serving' >/dev/null && "
            f"curl --noproxy '*' -fsS http://{cfg.host}:{cfg.etcd_client_port}/health >/dev/null && "
            f"! grep -Eiq 'fatal|panic|segmentation fault|check failed|address already in use' {shlex.quote(self.output_dir)}/xservice.log"
        )
        result = self._exec(command, timeout=30)
        if result.success:
            return True, "ok"
        logs = self._exec(f"tail -n 80 {shlex.quote(self.output_dir)}/etcd.log 2>/dev/null; tail -n 80 {shlex.quote(self.output_dir)}/xservice.log 2>/dev/null", timeout=30)
        return False, logs.stdout or result.stderr or result.stdout

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
