"""Lightweight host command executors for local and SSH remote ops."""

from __future__ import annotations

import logging
from xtask.utils.secrets import redact
import os
import shlex
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from xtask.utils.container_manager import CommandResult


class HostExecutor(ABC):
    """Execute commands and transfer files on one host."""

    @abstractmethod
    def run(
        self,
        command: str,
        timeout: Optional[int] = 300,
        capture_output: bool = True,
        abort_event=None,
        abort_poll_interval: float = 300.0,
        **kwargs,
    ) -> CommandResult:
        raise NotImplementedError

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> bool:
        raise NotImplementedError

    def sync_path(
        self,
        local_path: str,
        remote_path: str,
        excludes=(),
        timeout: int = 3600,
    ) -> bool:
        """Incrementally sync one path when supported by the executor."""
        return self.upload(local_path, remote_path)


class LocalHostExecutor(HostExecutor):
    """Run commands on the current machine via subprocess."""

    def run(
        self,
        command: str,
        timeout: Optional[int] = 300,
        capture_output: bool = True,
        abort_event=None,
        abort_poll_interval: float = 300.0,
        **kwargs,
    ) -> CommandResult:
        if abort_event is not None:
            return self._run_abortable(
                command,
                timeout=timeout,
                capture_output=capture_output,
                abort_event=abort_event,
                abort_poll_interval=abort_poll_interval,
                **kwargs,
            )

        start_time = time.time()
        try:
            process_kwargs = {
                "shell": True,
                "text": True,
                "timeout": timeout,
            }
            if capture_output:
                process_kwargs.update(
                    {
                        "stdout": subprocess.PIPE,
                        "stderr": subprocess.PIPE,
                    }
                )
            process_kwargs.update(kwargs)
            logging.debug("LocalHostExecutor running: %s", redact(command))
            result = subprocess.run(command, **process_kwargs)
            return CommandResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout if capture_output else "",
                stderr=result.stderr if capture_output else "",
                execution_time=time.time() - start_time,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Command timeout after {timeout} seconds",
                execution_time=time.time() - start_time,
            )
        except Exception as exc:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                execution_time=time.time() - start_time,
            )

    def _run_abortable(
        self,
        command: str,
        timeout: Optional[int],
        capture_output: bool,
        abort_event,
        abort_poll_interval: float,
        **kwargs,
    ) -> CommandResult:
        start_time = time.time()
        stdout_data = []
        stderr_data = []
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                start_new_session=True,
                **kwargs,
            )
            deadline = None if timeout is None else start_time + timeout
            while True:
                if abort_event is not None and abort_event.is_set():
                    try:
                        os.killpg(process.pid, 15)
                    except Exception:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except Exception:
                        try:
                            os.killpg(process.pid, 9)
                        except Exception:
                            process.kill()
                    return CommandResult(
                        success=False,
                        exit_code=-2,
                        stdout="".join(stdout_data),
                        stderr="Command aborted by abort_event",
                        execution_time=time.time() - start_time,
                    )

                ret = process.poll()
                if ret is not None:
                    if capture_output:
                        out, err = process.communicate()
                        stdout_data.append(out or "")
                        stderr_data.append(err or "")
                    return CommandResult(
                        success=ret == 0,
                        exit_code=ret,
                        stdout="".join(stdout_data),
                        stderr="".join(stderr_data),
                        execution_time=time.time() - start_time,
                    )

                if deadline is not None and time.time() >= deadline:
                    try:
                        os.killpg(process.pid, 15)
                    except Exception:
                        process.terminate()
                    return CommandResult(
                        success=False,
                        exit_code=-1,
                        stdout="".join(stdout_data),
                        stderr=f"Command timeout after {timeout} seconds",
                        execution_time=time.time() - start_time,
                    )
                time.sleep(min(abort_poll_interval, 1.0))
        except Exception as exc:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                execution_time=time.time() - start_time,
            )

    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            src = Path(local_path)
            dst = Path(remote_path)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return True
        except Exception as exc:
            logging.error("Local upload failed: %s", exc)
            return False

    def download(self, remote_path: str, local_path: str) -> bool:
        return self.upload(remote_path, local_path)


class SshHostExecutor(HostExecutor):
    """Run commands on a remote host over SSH/SCP.

    Supports two auth modes:
    - Passwordless (key-based): uses BatchMode=yes, no password needed.
    - Password-based: uses a PTY wrapper to auto-fill the password prompt,
      so it works without sshpass/paramiko/pexpect.
    """

    def __init__(
        self,
        host: str,
        user: Optional[str] = None,
        port: int = 22,
        password: Optional[str] = None,
        identity_file: Optional[str] = None,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.identity_file = identity_file
        self._target = f"{user}@{host}" if user else host

    @property
    def _is_passwordless(self) -> bool:
        return not self.password

    def _ssh_base_args(self) -> list:
        args = ["ssh", "-p", str(self.port)]
        if self._is_passwordless:
            args += ["-o", "BatchMode=yes"]
        else:
            # Password mode: force password auth, avoid hanging on pubkey retries.
            args += [
                "-o", "PreferredAuthentications=password",
                "-o", "PubkeyAuthentication=no",
                "-o", "NumberOfPasswordPrompts=1",
            ]
        args += ["-o", "StrictHostKeyChecking=accept-new"]
        if self.identity_file:
            args += ["-i", self.identity_file]
        args.append(self._target)
        return args

    def _ssh_prefix(self) -> str:
        return " ".join(self._ssh_base_args())

    def _run_ssh(self, remote_command_argv: list, timeout: Optional[int]) -> CommandResult:
        argv = self._ssh_base_args() + remote_command_argv
        start_time = time.time()
        if not self._is_passwordless:
            from xtask.utils.ssh_askpass import run_ssh_with_password

            exit_code, stdout, stderr = run_ssh_with_password(
                argv, password=self.password, timeout=timeout or 300
            )
            return CommandResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=time.time() - start_time,
            )
        # Passwordless: plain subprocess.
        try:
            result = subprocess.run(
                argv,
                text=True,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return CommandResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=time.time() - start_time,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"SSH command timeout after {timeout} seconds",
                execution_time=time.time() - start_time,
            )
        except Exception as exc:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                execution_time=time.time() - start_time,
            )

    def run(
        self,
        command: str,
        timeout: Optional[int] = 300,
        capture_output: bool = True,
        abort_event=None,
        abort_poll_interval: float = 300.0,
        **kwargs,
    ) -> CommandResult:
        if abort_event is not None:
            logging.warning(
                "SshHostExecutor ignores abort_event for host %s in v1", self.host
            )
        logging.debug("SshHostExecutor(%s) running: %s", self.host, redact(command))
        # Pass a single remote command string so ssh runs it under the login shell.
        # Avoid ["bash","-lc",cmd] argv splitting issues across OpenSSH versions.
        return self._run_ssh([command], timeout)

    def _scp_args(self) -> list:
        args = ["scp", "-P", str(self.port)]
        if self._is_passwordless:
            args += ["-o", "BatchMode=yes"]
        else:
            args += [
                "-o", "PreferredAuthentications=password",
                "-o", "PubkeyAuthentication=no",
                "-o", "NumberOfPasswordPrompts=1",
            ]
        args += ["-o", "StrictHostKeyChecking=accept-new"]
        if self.identity_file:
            args += ["-i", self.identity_file]
        args.append("-r")
        return args

    def _run_scp(self, argv: list, timeout: int = 600) -> int:
        if not self._is_passwordless:
            from xtask.utils.ssh_askpass import run_ssh_with_password

            exit_code, _, stderr = run_ssh_with_password(
                argv, password=self.password, timeout=timeout
            )
            if exit_code != 0 and stderr:
                logging.error("SCP failed: %s", stderr[-500:])
            return exit_code
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            logging.error("SCP failed: %s", result.stderr[-500:])
        return result.returncode

    def upload(self, local_path: str, remote_path: str) -> bool:
        try:
            local = Path(local_path)
            if local.is_dir():
                return self._upload_dir_via_tar(str(local.resolve()), remote_path)
            parent = str(Path(remote_path).parent)
            mkdir_result = self.run(f"mkdir -p {parent}", timeout=60)
            if not mkdir_result.success:
                logging.error(
                    "Remote mkdir failed on %s: %s",
                    self.host,
                    mkdir_result.stderr or mkdir_result.stdout,
                )
                return False
            argv = self._scp_args() + [
                local_path,
                f"{self._target}:{remote_path}",
            ]
            exit_code = self._run_scp(argv, timeout=600)
            if exit_code != 0:
                logging.error("SCP upload to %s failed (exit %s)", self.host, exit_code)
                return False
            return True
        except Exception as exc:
            logging.error("SCP upload failed: %s", exc)
            return False

    def upload_dir_to_exact_path(self, local_dir: str, remote_path: str) -> bool:
        """Opt in to uploading directory contents under an arbitrary remote name.

        The ordinary ``upload`` method retains its existing directory behavior
        for other callers; SWE-bench uses this for its per-run work directory.
        """
        local = Path(local_dir)
        if not local.is_dir():
            logging.error("Directory upload source does not exist: %s", local)
            return False
        return self._upload_dir_via_tar(
            str(local.resolve()), remote_path, exact_remote_path=True
        )

    def sync_path(
        self,
        local_path: str,
        remote_path: str,
        excludes=(),
        timeout: int = 3600,
    ) -> bool:
        """Use rsync for fast incremental resource transfer over the same SSH auth."""
        local = Path(local_path)
        if not local.exists():
            logging.error("Rsync source does not exist: %s", local)
            return False

        remote_rsync = self.run("command -v rsync >/dev/null 2>&1", timeout=30)
        if shutil.which("rsync") is None or not remote_rsync.success:
            logging.warning(
                "rsync is unavailable for %s; falling back to tar upload of %s",
                self.host,
                local,
            )
            if local.is_dir():
                return self._sync_dir_via_tar(
                    str(local),
                    remote_path,
                    excludes=excludes,
                    timeout=timeout,
                )
            return self.upload(str(local), remote_path)

        remote_parent = remote_path if local.is_dir() else str(Path(remote_path).parent)
        prepared = self.run(f"mkdir -p {shlex.quote(remote_parent)}", timeout=60)
        if not prepared.success:
            logging.error(
                "Remote rsync directory prepare failed on %s: %s",
                self.host,
                prepared.stderr or prepared.stdout,
            )
            return False

        ssh_args = self._ssh_base_args()
        ssh_args.pop()  # rsync supplies user@host separately.
        argv = [
            "rsync",
            "-a",
            "--partial",
            "--size-only",
            "--human-readable",
            "--info=stats1,name0",
            "--timeout=300",
            "-e",
            shlex.join(ssh_args),
        ]
        for pattern in excludes or ():
            argv.extend(("--exclude", str(pattern)))

        source = str(local)
        destination = remote_path
        if local.is_dir():
            source = source.rstrip("/") + "/"
            destination = destination.rstrip("/") + "/"
        argv.extend((source, f"{self._target}:{destination}"))

        try:
            started = time.time()
            if self.password:
                from xtask.utils.ssh_askpass import run_ssh_with_password

                exit_code, stdout, stderr = run_ssh_with_password(
                    argv, password=self.password, timeout=timeout
                )
            else:
                result = subprocess.run(
                    argv,
                    text=True,
                    timeout=timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                exit_code, stdout, stderr = (
                    result.returncode,
                    result.stdout or "",
                    result.stderr or "",
                )
            if exit_code != 0:
                logging.error(
                    "Rsync to %s failed (exit %s): %s",
                    self.host,
                    exit_code,
                    (stderr or stdout)[-1000:],
                )
                return False
            logging.info(
                "Rsync completed to %s:%s in %.2fs: %s",
                self.host,
                remote_path,
                time.time() - started,
                (stdout or "no changes").strip()[-500:],
            )
            return True
        except subprocess.TimeoutExpired:
            logging.error("Rsync to %s timed out after %ss", self.host, timeout)
            return False
        except Exception as exc:
            logging.error("Rsync to %s failed: %s", self.host, exc)
            return False

    def _sync_dir_via_tar(
        self,
        local_dir: str,
        remote_path: str,
        excludes=(),
        timeout: int = 3600,
    ) -> bool:
        """Merge a directory through tar while preserving rsync exclusions."""
        import tempfile

        local = Path(local_dir)
        parent = str(local.parent)
        name = local.name
        tar_fd, tar_path = tempfile.mkstemp(
            prefix="xtask_resource_sync_", suffix=".tar.gz"
        )
        os.close(tar_fd)
        try:
            tar_command = ["tar", "czf", tar_path]
            for pattern in excludes or ():
                tar_command.extend(("--exclude", str(pattern)))
                tar_command.extend(("--exclude", f"{name}/{pattern}"))
            tar_command.extend(("-C", parent, name))
            subprocess.check_call(tar_command)

            remote_tar = f"/tmp/xtask_resource_sync_{os.getpid()}_{name}.tar.gz"
            remote_parent = str(Path(remote_path).parent)
            prepared = self.run(
                f"mkdir -p {shlex.quote(remote_parent)} && "
                f"chmod -R u+w {shlex.quote(remote_path)} 2>/dev/null || true",
                timeout=120,
            )
            if not prepared.success:
                return False
            argv = self._scp_args() + [tar_path, f"{self._target}:{remote_tar}"]
            if self._run_scp(argv, timeout=timeout) != 0:
                return False
            extracted = self.run(
                f"tar xzf {shlex.quote(remote_tar)} -C "
                f"{shlex.quote(remote_parent)} && "
                f"rm -f {shlex.quote(remote_tar)} && "
                f"test -d {shlex.quote(remote_path)}",
                timeout=timeout,
            )
            return extracted.success
        except Exception as exc:
            logging.error("Resource tar sync to %s failed: %s", self.host, exc)
            return False
        finally:
            try:
                os.remove(tar_path)
            except OSError:
                pass

    def _upload_dir_via_tar(
        self, local_dir: str, remote_path: str, *, exact_remote_path: bool = False
    ) -> bool:
        """Upload a directory via tar to avoid scp nesting / 'unexpected filename: .'."""
        import tempfile

        local = Path(local_dir)
        parent = str(local.parent)
        name = local.name
        tar_fd, tar_path = tempfile.mkstemp(prefix="xtask_scp_dir_", suffix=".tar.gz")
        os.close(tar_fd)
        try:
            subprocess.check_call(
                [
                    "tar", "czf", tar_path,
                    "--exclude=kernel_meta",
                    "--exclude=log",
                    "--exclude=__pycache__",
                    "-C", parent, name,
                ]
            )
            remote_tar = f"/tmp/xtask_dir_{os.getpid()}_{name}.tar.gz"
            remote_parent = str(Path(remote_path).parent)
            if exact_remote_path:
                prep_command = (
                    f"rm -rf {shlex.quote(remote_path)} && "
                    f"mkdir -p {shlex.quote(remote_path)}"
                )
            else:
                # Preserve the pre-existing upload semantics for all other users.
                prep_command = (
                    f"chmod -R u+w {remote_path} 2>/dev/null || true; "
                    f"rm -rf {remote_path}; mkdir -p {remote_parent}"
                )
            prep = self.run(prep_command, timeout=120)
            if not prep.success:
                logging.error(
                    "Remote dir prepare failed on %s: %s",
                    self.host,
                    prep.stderr or prep.stdout,
                )
                return False
            if not self.upload(tar_path, remote_tar):
                return False
            if exact_remote_path:
                # The archive contains local_dir/; strip that prefix when the
                # SWE-bench run has a differently named remote work directory.
                extract_command = (
                    f"tar xzf {shlex.quote(remote_tar)} "
                    f"-C {shlex.quote(remote_path)} --strip-components=1 && "
                    f"rm -f {shlex.quote(remote_tar)} && "
                    f"test -d {shlex.quote(remote_path)} && echo DIR_OK"
                )
            else:
                extract_command = (
                    f"tar xzf {remote_tar} -C {remote_parent} && "
                    f"rm -f {remote_tar} && test -d {remote_path} && echo DIR_OK"
                )
            extract = self.run(extract_command, timeout=300)
            if not extract.success or "DIR_OK" not in (extract.stdout or ""):
                logging.error(
                    "Remote dir extract failed on %s: %s",
                    self.host,
                    extract.stderr or extract.stdout,
                )
                return False
            return True
        except Exception as exc:
            logging.error("Directory tar upload failed: %s", exc)
            return False
        finally:
            try:
                os.remove(tar_path)
            except OSError:
                pass

    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            local = Path(local_path)
            local.parent.mkdir(parents=True, exist_ok=True)
            # Always download directory into a fresh sibling path. Downloading
            # remote/log to an existing local/log creates local/log/log.
            temporary_path = local.with_name(f".{local.name}.download-{os.getpid()}")
            if temporary_path.exists():
                shutil.rmtree(temporary_path)
            argv = self._scp_args() + [
                f"{self._target}:{remote_path}",
                str(temporary_path),
            ]
            exit_code = self._run_scp(argv, timeout=600)
            if exit_code != 0:
                logging.error("SCP download from %s failed (exit %s)", self.host, exit_code)
                return False
            if local.exists():
                if local.is_dir():
                    shutil.rmtree(local)
                else:
                    local.unlink()
            temporary_path.replace(local)
            return True
        except Exception as exc:
            logging.error("SCP download failed: %s", exc)
            return False


def is_local_host(host: str, local_ips=None) -> bool:
    """Return True when host should use LocalHostExecutor."""
    if not host:
        return True
    normalized = str(host).strip().lower()
    if normalized in {"local", "localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    if local_ips is None:
        local_ips = set()
        try:
            from xtask.utils import get_host_ip

            local_ips.add(get_host_ip())
        except Exception:
            pass
    return host in local_ips


def _load_dotenv_file(path: str) -> None:
    """Load KEY=VALUE pairs from a dotenv file into os.environ (no overwrite)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        logging.debug("Skip dotenv %s: %s", path, exc)


def load_ssh_credentials() -> dict:
    """Load SSH host credentials from config/ssh_hosts.json and/or env vars.

    Precedence (highest first for a given host):
      1. config/ssh_hosts.json — { "host_ip": {"user":..., "password":..., "port":...} }
      2. Environment variables XTASK_SSH_USER / XTASK_SSH_PASS / XTASK_SSH_PORT
         (also loaded from ./.env or ~/.xtask.env if present and not already set)
    """
    import json

    # Soft-load dotenv so Cursor agent shells can pick up credentials the user exported elsewhere.
    for dotenv_path in (
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.expanduser("~"), ".xtask.env"),
    ):
        if os.path.exists(dotenv_path):
            _load_dotenv_file(dotenv_path)

    creds: dict = {}
    config_path = os.path.join(os.getcwd(), "config", "ssh_hosts.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                file_creds = json.load(f)
            for host, cfg in (file_creds or {}).items():
                creds[host] = {
                    "user": cfg.get("user"),
                    "password": cfg.get("password"),
                    "port": int(cfg.get("port", 22)),
                    "identity_file": cfg.get("identity_file"),
                }
    except Exception as exc:
        logging.warning("Failed to load config/ssh_hosts.json: %s", exc)

    # Env-var fallback for hosts not in the config file.
    env_user = os.environ.get("XTASK_SSH_USER")
    env_pass = os.environ.get("XTASK_SSH_PASS")
    env_port = os.environ.get("XTASK_SSH_PORT")
    if env_pass and env_user:
        env_entry = {
            "user": env_user,
            "password": env_pass,
            "port": int(env_port) if env_port else 22,
            "identity_file": os.environ.get("XTASK_SSH_IDENTITY"),
        }
        # Only apply env defaults to hosts not explicitly in config file.
        if "_default" not in creds:
            creds["_default"] = env_entry
    return creds


def create_host_executor(host: str, user: Optional[str] = None, port: int = 22) -> HostExecutor:
    if is_local_host(host):
        return LocalHostExecutor()

    creds = load_ssh_credentials()
    host_cred = creds.get(host) or creds.get("_default") or {}
    ssh_user = user or host_cred.get("user")
    ssh_port = port if port != 22 else host_cred.get("port", 22)
    password = host_cred.get("password")
    identity_file = host_cred.get("identity_file")

    return SshHostExecutor(
        host=host,
        user=ssh_user,
        port=ssh_port,
        password=password,
        identity_file=identity_file,
    )
