"""Phase-1 Coordinator/Router orchestration in existing Engine containers.

The Engine runtime list and device/rank allocation remain untouched. One owner
starts etcd, then Coordinator members, then Routers; retries affect only the
owned child process. Inference readiness is checked after Engine registration.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import re
import shlex
import time
import uuid

from xtask.core.pd_control_service import XllmControlService, build_xllm_service_config
from xtask.core.xllm_service_config import (
    build_xllm_service_topology,
    get_xllm_service_config,
    select_service_container,
)


class ManagedServiceProcess:
    """One fenced child process; commands are encoded across docker's shell."""

    def __init__(self, runtime, directory, log_name):
        self.runtime = runtime
        self.directory = directory
        self.pid_file = directory + "/process.json"
        self.log_file = directory + "/" + log_name
        self.started = False

    def execute(self, command, timeout=30):
        # ContainerManager wraps commands in another double-quoted shell.
        # A base64 payload prevents host expansion of dollars/quotes/tokens.
        from xtask.utils.secrets import register_encoded_command
        register_encoded_command(command)
        payload = base64.b64encode(command.encode()).decode()
        script = (
            "import base64,subprocess,sys;"
            "sys.exit(subprocess.call(base64.b64decode(bytes("
            + repr(list(payload.encode()))
            + ")).decode(),shell=True))"
        )
        return self.runtime.container_manager.exec_in_container(
            self.runtime.container_config.container_name,
            "python3 -c " + shlex.quote(script),
            timeout=timeout,
        )

    def _helper(self, action, arguments=()):
        return shlex.join(
            [
                "python3",
                "/workspace/tools/manage_xllm_service_process.py",
                action,
                "--pid-file",
                self.pid_file,
                *arguments,
            ]
        )

    def start(self, argv, environment):
        cmd = self._helper("start", ["--log-file", self.log_file, "--", *argv])
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items())
        self.started = True
        result = self.execute(exports + " " + cmd, timeout=30)
        # A failed invocation can still have created the child. Cleanup the
        # ownership record on both success and failure.
        self.started = True
        if not result.success:
            raise RuntimeError(f"Managed process launch failed: {self.log_file}")

    def healthy(self, timeout=10):
        result = self.execute(self._helper("status"), timeout=timeout)
        if not result.success:
            return False
        check = self.execute(
            "! grep -Eiq 'fatal|panic|segmentation fault|check failed|address already in use' "
            + shlex.quote(self.log_file),
            timeout=timeout,
        )
        return check.success

    def stop(self, grace=15):
        if not self.started:
            return
        result = self.execute(self._helper("stop", ["--grace", str(grace)]), timeout=grace + 10)
        if not result.success:
            raise RuntimeError(f"Failed to stop owned process: {self.log_file}")
        self.started = False


class XllmServiceCluster:
    """Select legacy single-process or split-role flow once per case run."""

    def __init__(self, task_config, runtimes, output_dir):
        self.task_config = task_config
        self.runtimes = runtimes
        self.topology = build_xllm_service_topology(task_config)
        self.control = get_xllm_service_config(task_config)
        from xtask.core.pd_control_service import begin_service_run
        begin_service_run(task_config)
        self.config = build_xllm_service_config(task_config)
        self.shutdown_grace = int(self.control.get("shutdown_grace_seconds", 15))
        if not 1 <= self.shutdown_grace <= 300:
            raise ValueError("xllm_service.shutdown_grace_seconds must be between 1 and 300")
        self.shutdown_drain_seconds = int(self.control.get("shutdown_drain_seconds", 0))
        if not 0 <= self.shutdown_drain_seconds <= 300:
            raise ValueError("xllm_service.shutdown_drain_seconds must be between 0 and 300")
        runner = (task_config.get("deployment") or {}).get("benchmark", {}).get("runner_instance")
        self.etcd_runtime = select_service_container(
            runtimes,
            self.topology.etcd_host,
            self.topology.etcd_container_instance,
            None if self.topology.explicit else runner or False,
        )
        self.bindings = [
            (
                item,
                select_service_container(
                    runtimes,
                    item.host,
                    item.container_instance,
                    None if self.topology.explicit else runner or False,
                ),
            )
            for item in self.topology.instances
        ]
        from xtask.utils.container_manager import path_mapping_in_container

        self.root = os.path.join(
            path_mapping_in_container(output_dir), "xllm_service", uuid.uuid4().hex
        )
        self.local_root = os.path.join(output_dir, "xllm_service", self.root.rsplit("/", 1)[-1])
        self.etcd = ManagedServiceProcess(self.etcd_runtime, self.root + "/etcd", "etcd.log")
        self.members = [
            (item, ManagedServiceProcess(runtime, self.root + "/" + item.name, "xservice.log"))
            for item, runtime in self.bindings
        ]
        self.legacy = None
        self.is_split = False
        self.benchmark_endpoint = None
        self.last_error = ""
        self._deadline = None

    def _environment(self):
        hosts = list(
            dict.fromkeys(
                ["localhost", "127.0.0.1", self.topology.etcd_host]
                + [str(runtime.plan.host) for runtime in self.runtimes]
                + list(self.control.get("no_proxy_hosts") or [])
            )
        )
        env = {"no_proxy": ",".join(hosts), "NO_PROXY": ",".join(hosts)}
        if self.config.output_subdir == "pd_control":
            env["ENABLE_DECODE_RESPONSE_TO_SERVICE"] = "true"
        return env

    def _service_argv(self, item):
        argv = [
            "xllm_master_serving",
            f"--etcd_addr={self.topology.etcd_host}:{self.topology.etcd_client_port}",
            f"--http_server_port={item.http_server_port}",
            f"--rpc_server_port={item.rpc_server_port}",
            f"--server_host={item.host}",
            f"--service_advertise_host={item.host}",
            f"--service_deployment_role={item.role}",
            f"--internal_api_token={self.config.internal_api_token}",
            f"--etcd_namespace={self.control.get('etcd_namespace', '')}",
            "--control_plane_shard_id=default",
            f"--tokenizer_path={self.task_config.get('model_path', '')}",
        ]
        # extra_service_args are scoped to this YAML; omitted flags keep binary defaults.
        for name, value in (self.control.get("extra_service_args") or {}).items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"xllm_service.extra_service_args.{name} is not a valid flag name")
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise TypeError(f"xllm_service.extra_service_args.{name} must be a scalar")
            argv.append(f"--{name}={value}")
        return argv

    def _wait(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        self._deadline = deadline
        try:
            while time.monotonic() < deadline:
                if predicate():
                    return time.monotonic() < deadline
                time.sleep(min(1, max(0, deadline - time.monotonic())))
            return False
        finally:
            self._deadline = None

    def _healthy(self, process):
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return False
        timeout = (
            10
            if self._deadline is None
            else min(10, max(0.1, (self._deadline - time.monotonic()) / 2))
        )
        return process.healthy(timeout=timeout)

    def _probe(self, process, command, timeout=10):
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return SimpleNamespace(success=False, stdout="", stderr="deadline expired")
        if self._deadline is not None:
            timeout = min(timeout, max(0.1, self._deadline - time.monotonic()))
        return process.execute(command, timeout=timeout)

    def _start_member(self, item, process):
        for attempt in range(self.config.startup_attempts):
            process.stop()
            try:
                process.start(self._service_argv(item), self._environment())
            except RuntimeError:
                logging.warning("Service %s launch failed, attempt=%s", item.name, attempt + 1)
                continue
            time.sleep(self.config.startup_wait_seconds)
            if process.healthy():
                logging.info(
                    "Started %s %s on %s in %s",
                    item.role,
                    item.name,
                    item.host,
                    process.runtime.plan.instance_name,
                )
                return
            logging.warning("Service %s failed launch check, attempt=%s", item.name, attempt + 1)
        raise RuntimeError(f"Service {item.name} failed startup: {process.log_file}")

    def _anchor_ready(self):
        # One active Coordinator is sufficient; follower processes are checked
        # for liveness, never required to report themselves as leader.
        namespace = str(self.control.get("etcd_namespace", "")).strip("/")
        key = ("/" + namespace + "/" if namespace else "") + "XLLM:SERVICE:MASTER"
        cmd = shlex.join(
            [
                "etcdctl",
                f"--endpoints=http://{self.topology.etcd_host}:{self.topology.etcd_client_port}",
                "--dial-timeout=2s",
                "--command-timeout=5s",
                "get",
                key,
                "--print-value-only",
            ]
        )
        result = self._probe(self.etcd, "ETCDCTL_API=3 " + cmd)
        expected = {
            f"{item.host}:{item.rpc_server_port}"
            for item, _ in self.members
            if item.role == "COORDINATOR"
        }
        return result.success and result.stdout.strip() in expected

    def start_with_retry(self):
        probe = XllmControlService(
            self.etcd_runtime,
            self.config,
            self.task_config.get("model_path", ""),
            os.path.dirname(os.path.dirname(self.local_root)),
        )
        capabilities = probe._resolve_binary_capabilities()
        if not capabilities.supports_deployment_role:
            if self.control.get("etcd") or self.control.get("etcd_namespace"):
                raise RuntimeError(
                    "Legacy Service flow requires flat etcd configuration without namespace"
                )
            if self.topology.explicit:
                raise RuntimeError("Configured Coordinator/Router lists require a new binary")
            self.legacy = probe
            self.benchmark_endpoint = f"http://{self.config.host}:{self.config.http_server_port}"
            probe.start_with_retry()
            return
        self.is_split = True
        if (
            not self.topology.explicit
            and self.control.get("service_deployment_role", "COORDINATOR") != "COORDINATOR"
        ):
            raise ValueError(
                "Flat ROUTER-only configuration cannot elect a Coordinator; use instance lists"
            )
        self._preflight()
        self._start_etcd()
        for item, process in self.members:
            if item.role == "COORDINATOR":
                self._start_member(item, process)
        if not self._wait(self._anchor_ready, self.config.master_key_wait_seconds):
            raise RuntimeError("Coordinator leader is not one of this case's configured members")
        for item, process in self.members:
            if item.role == "ROUTER":
                self._start_member(item, process)
        self._write_metadata()

    def _preflight(self):
        """Verify each hosting container once before starting any process."""
        binary_digest = None
        containers = {id(self.etcd_runtime): self.etcd}
        containers.update({id(p.runtime): p for _, p in self.members})
        for process in containers.values():
            result = process.execute(process._helper("check"))
            if not result.success:
                raise RuntimeError(
                    "Process ownership helper/pidfd unavailable in "
                    + process.runtime.plan.instance_name
                )
        checked = set()
        for _, process in self.members:
            if id(process.runtime) in checked:
                continue
            checked.add(id(process.runtime))
            result = process.execute(
                "xllm_master_serving --help 2>&1 | grep -q service_deployment_role"
            )
            if not result.success:
                raise RuntimeError(
                    "Service binary lacks deployment roles in "
                    + process.runtime.plan.instance_name
                )
            digest = process.execute("sha256sum $(command -v xllm_master_serving)")
            value = digest.stdout.split()[0] if digest.success and digest.stdout.split() else None
            if not value or (binary_digest is not None and value != binary_digest):
                raise RuntimeError(
                    "Coordinator/Router hosting containers use different Service binaries"
                )
            binary_digest = value

    def _start_etcd(self):
        self.etcd.start(
            [
                "etcd",
                "--data-dir",
                self.etcd.directory + "/data",
                "--listen-peer-urls",
                f"http://0.0.0.0:{self.topology.etcd_peer_port}",
                "--listen-client-urls",
                f"http://0.0.0.0:{self.topology.etcd_client_port}",
                "--advertise-client-urls",
                f"http://{self.topology.etcd_host}:{self.topology.etcd_client_port}",
                "--initial-advertise-peer-urls",
                f"http://{self.topology.etcd_host}:{self.topology.etcd_peer_port}",
                "--initial-cluster",
                f"xtask-etcd=http://{self.topology.etcd_host}:{self.topology.etcd_peer_port}",
                "--name",
                "xtask-etcd",
            ],
            self._environment(),
        )
        if not self._wait(
            lambda: self._healthy(self.etcd)
            and self._probe(
                self.etcd,
                f"curl --noproxy '*' -fsS --max-time 5 http://{self.topology.etcd_host}:"
                f"{self.topology.etcd_client_port}/health",
                timeout=10,
            ).success,
            self.config.master_key_wait_seconds,
        ):
            raise RuntimeError("Owned etcd failed readiness")

    def _write_metadata(self):
        router = self.topology.router
        self.benchmark_endpoint = f"http://{router.host}:{router.http_server_port}"
        metadata = {
            "benchmark_endpoint": self.benchmark_endpoint,
            "instances": [
                {
                    "name": item.name,
                    "role": item.role,
                    "host": item.host,
                    "http_server_port": item.http_server_port,
                    "rpc_server_port": item.rpc_server_port,
                    "container_instance": process.runtime.plan.instance_name,
                }
                for item, process in self.members
            ],
        }
        write = "import pathlib;p=pathlib.Path(" + repr(self.root + "/topology.json") + ");"
        write += "p.write_text(" + repr(json.dumps(metadata, indent=2) + "\n") + ")"
        if not self.etcd.execute("python3 -c " + shlex.quote(write)).success:
            raise RuntimeError("Failed to write Service run metadata")

    def wait_for_http_endpoint(self, timeout=90):
        if self.legacy:
            return self.legacy.wait_for_http_endpoint(timeout)
        model = (
            self.task_config.get("served_model_name")
            or Path(self.task_config.get("model_path", "").rstrip("/")).name
        )

        def ready():
            if not self._anchor_ready() or not self._healthy(self.etcd):
                return False
            for item, process in self.members:
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    return False
                if not self._healthy(process):
                    self.last_error = f"{item.name} process/log unhealthy"
                    return False
                if item.role != "ROUTER":
                    continue
                base = f"http://{item.host}:{item.http_server_port}"
                health = self._probe(
                    process, f"curl --noproxy '*' -fsS --max-time 5 {base}/readyz"
                )
                models = self._probe(
                    process, f"curl --noproxy '*' -fsS --max-time 5 {base}/v1/models"
                )
                try:
                    valid = (
                        health.success
                        and json.loads(health.stdout).get("status") == "ready"
                        and models.success
                        and model
                        in [v.get("id") for v in json.loads(models.stdout).get("data", [])]
                    )
                except (ValueError, TypeError, AttributeError):
                    valid = False
                if not valid:
                    self.last_error = (
                        f"Router {item.name} not ready or model {model!r} unavailable"
                    )
                    return False
            return True

        result = self._wait(ready, timeout)
        if result:
            self.last_error = ""
        return result

    def drain_after_benchmark(self):
        """Optional bounded grace, not an unsupported server-side drain API."""
        if self.shutdown_drain_seconds:
            logging.info("Benchmark client stopped; allow %ss request drain grace", self.shutdown_drain_seconds)
            time.sleep(self.shutdown_drain_seconds)

    def stop_members(self):
        """Stop owned Routers before Coordinators, keeping etcd for Engine exit."""
        if self.legacy:
            return  # Preserve the old single-process shutdown contract.
        errors = []
        grace = self.shutdown_grace
        for _, process in reversed(self.members):
            try:
                process.stop(grace=grace)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("; ".join(errors))

    def stop_etcd(self):
        """Only this cluster's fenced etcd child is owned by xTask."""
        if self.legacy:
            self.legacy.stop()
        else:
            self.etcd.stop(grace=self.shutdown_grace)

    def stop(self):
        errors = []
        for operation in (self.stop_members, self.stop_etcd):
            try:
                operation()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("; ".join(errors))

    def archive_logs(self, case_output, benchmarkpath):
        """Store each trial's current logs without altering earlier snapshots."""
        import shutil

        destination = Path(case_output) / benchmarkpath / "xllm_service"
        source = Path(self.local_root)
        if not source.exists():
            return
        paths = list(source.glob("*/xservice*.log")) + list(source.glob("etcd/etcd*.log"))
        paths += list(source.glob("topology.json"))
        paths += list(source.glob("remote/*/topology.json"))
        paths += list(source.glob("remote/*/*/xservice*.log"))
        paths += list(source.glob("remote/*/etcd/etcd*.log"))
        for path in paths:
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    def collect_logs(self):
        """Pull separate per-host trees; shared workspace paths never overwrite."""
        if not self.is_split:
            return
        Path(self.local_root).parent.mkdir(parents=True, exist_ok=True)
        hosting = {str(self.etcd_runtime.plan.host): self.etcd_runtime}
        hosting.update({str(p.runtime.plan.host): p.runtime for _, p in self.members})
        for index, runtime in enumerate(hosting.values()):
            access = ManagedServiceProcess(runtime, self.root, "access.log")
            permission = "import pathlib,os;p=pathlib.Path(" + repr(self.root) + ");"
            permission += "paths=[p]+[v for v in p.iterdir() if v.is_dir()];"
            permission += "paths += list(p.glob('*/xservice*.log'))+list(p.glob('etcd/etcd.log'))+list(p.glob('topology.json'));"
            permission += (
                "[os.chown(v,"
                + str(os.getuid())
                + ","
                + str(os.getgid())
                + ") for v in paths if v.exists()]"
            )
            access.execute("python3 -c " + shlex.quote(permission), timeout=30)
            if getattr(runtime.plan, "is_local", True):
                continue
            remote = self.local_root + "/archive"
            command = "import pathlib,shutil;src=pathlib.Path(" + repr(self.root) + ");"
            command += "dst=src/'archive';shutil.rmtree(dst,ignore_errors=True);dst.mkdir();"
            command += "[( (dst/p.relative_to(src)).parent.mkdir(parents=True,exist_ok=True),"
            command += (
                "shutil.copy2(p,dst/p.relative_to(src))) for p in src.glob('*/xservice*.log')];"
            )
            command += "p=src/'etcd'/'etcd.log';"
            command += "(dst/'etcd').mkdir(exist_ok=True);"
            command += "shutil.copy2(p,dst/'etcd'/'etcd.log') if p.exists() else None;"
            command += "p=src/'topology.json';shutil.copy2(p,dst/p.name) if p.exists() else None"
            if not access.execute("python3 -c " + shlex.quote(command)).success:
                logging.warning("Failed to stage Service logs on %s", runtime.plan.host)
                continue
            target = os.path.join(self.local_root, "remote", runtime.plan.host + "-" + str(index))
            access.execute("chmod -R a+rX " + shlex.quote(self.root + "/archive"), timeout=30)
            if not runtime.container_manager.executor.download(remote, target):
                logging.warning("Failed to archive Service logs from %s", runtime.plan.host)
