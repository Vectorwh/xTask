# framework/container/container_manager.py
import os
import sys
import time
import shlex
import signal
import inspect
import logging
from xtask.utils.secrets import redact
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
import yaml

from xtask.utils.config_parser import get_xtask_config_data


@dataclass
class ContainerConfig:
    """Container configuration"""
    image_home: str
    image_name: str
    container_name: str
    runtime: str = "docker"  # docker or nerdctl
    gpu_ids: List[int] = None
    ports: Dict[str, str] = None  # host_port:container_port
    volumes: Dict[str, str] = None  # host_path:container_path
    environment: Dict[str, str] = None
    working_dir: str = "/workspace"
    entrypoint: str = None
    command: str = None
    network_mode: str = "host"  # host, bridge, none
    # Explicit IPC mode; when unset, preserve the historical network-mode
    # fallback used by the scheduler.
    ipc_mode: Optional[str] = None
    # Size for the container-local /dev/shm tmpfs.
    shm_size: Optional[str] = None
    privileged: bool = True
    auto_remove: bool = True
    detach: bool = True
    stdin_open: bool = True  # Keep STDIN open
    tty: bool = True  # Allocate pseudo-TTY
    
    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = []
        if self.ports is None:
            self.ports = {}
        if self.volumes is None:
            self.volumes = {}
        if self.environment is None:
            self.environment = {}


@dataclass
class CommandResult:
    """Command execution result"""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float


class ContainerManager:
    """Manager for Docker/nerdctl container operations"""
    
    def __init__(self, config: ContainerConfig = None, logger=None, executor=None):
        """
        Initialize container manager
        
        Args:
            config: Container configuration
            logger: Logger instance for logging
            executor: Optional HostExecutor; defaults to local subprocess execution
        """
        self.config = config
        self.logger = logger
        self.container_id = None
        self._stop_event = threading.Event()
        if executor is None:
            from xtask.utils.host_executor import LocalHostExecutor

            executor = LocalHostExecutor()
        self.executor = executor

    
    def run_command(self, cmd: str, capture_output: bool = True, 
                   timeout: int = 300, abort_event=None, 
                   abort_poll_interval: float = 300.0, **kwargs) -> CommandResult:
        """
        Run shell command and return result
        
        Args:
            cmd: Command to execute
            capture_output: Whether to capture stdout/stderr
            timeout: Command timeout in seconds
            abort_event: 可选的中断信号（threading.Event / multiprocessing.Event）。
                         传入后使用 Popen + 进程组轮询方式执行，一旦该事件被置位，
                         立即终止命令进程（及其子进程组），命令返回中断结果，主流程可继续。
            abort_poll_interval: 中断轮询间隔（秒），仅在 abort_event 传入时生效
            **kwargs: Additional arguments for subprocess.run / Popen
            
        Returns:
            CommandResult object
        """
        from xtask.utils.host_executor import LocalHostExecutor

        # Local abortable path keeps existing ContainerManager behavior.
        if abort_event is not None and isinstance(self.executor, LocalHostExecutor):
            return self._run_command_abortable(
                cmd, capture_output=capture_output, timeout=timeout,
                abort_event=abort_event, abort_poll_interval=abort_poll_interval, **kwargs
            )

        return self.executor.run(
            cmd,
            timeout=timeout,
            capture_output=capture_output,
            abort_event=abort_event,
            abort_poll_interval=abort_poll_interval,
            **kwargs,
        )

    def _run_command_abortable(self, cmd: str, capture_output: bool = True,
                               timeout: int = 300, abort_event=None,
                               abort_poll_interval: float = 300.0, **kwargs) -> CommandResult:
        """
        使用 Popen + 进程组轮询方式执行命令，支持通过 abort_event 外部中断。

        中断/超时时会终止整个进程组（SIGTERM -> 等待 -> SIGKILL），
        以确保 shell 派生的孙进程（如 docker exec）一并被清理，主流程随后可继续。
        """
        start_time = time.time()
        popen_kwargs = {'shell': True, 'text': True}
        if capture_output:
            popen_kwargs.update({'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE})
        # 独立进程组，便于整组终止（POSIX）
        if os.name == 'posix':
            popen_kwargs['start_new_session'] = True
        popen_kwargs.update(kwargs)

        logging.debug(f"Executing command (abortable): {cmd}")

        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Command execution failed: {str(e)}")
            return CommandResult(
                success=False, exit_code=-1, stdout="", stderr=str(e),
                execution_time=execution_time
            )

        aborted = False
        timed_out = False
        while True:
            try:
                proc.wait(timeout=abort_poll_interval)
                break  # 进程已退出
            except subprocess.TimeoutExpired:
                pass

            # 检查外部中断信号
            if abort_event is not None and abort_event.is_set():
                aborted = True
                logging.info("Received abort signal, terminating command: %s", cmd)
                self._terminate_process_group(proc)
                break

            # 检查超时
            if (time.time() - start_time) >= timeout:
                timed_out = True
                logging.warning(f"Command timeout after {timeout} seconds: {cmd}")
                self._terminate_process_group(proc)
                break

        stdout, stderr = "", ""
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except Exception:
            try:
                self._terminate_process_group(proc)
            except Exception:
                pass
        execution_time = time.time() - start_time

        if aborted:
            return CommandResult(
                success=False, exit_code=-1,
                stdout=stdout or "", stderr="Command aborted by monitor (NPU process abnormal)",
                execution_time=execution_time
            )
        if timed_out:
            return CommandResult(
                success=False, exit_code=-1,
                stdout=stdout or "", stderr=f"Command timeout after {timeout} seconds",
                execution_time=execution_time
            )
        return CommandResult(
            success=proc.returncode == 0, exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout if capture_output else "", stderr=stderr if capture_output else "",
            execution_time=execution_time
        )

    def _terminate_process_group(self, proc):
        """终止进程（组）：先 SIGTERM，超时后 SIGKILL。"""
        try:
            if os.name == 'posix':
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception as e:
            logging.debug(f"terminate process group failed: {e}")
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                if os.name == 'posix':
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception as e:
                logging.debug(f"kill process group failed: {e}")
    
    def check_runtime_available(self) -> bool:
        """
        Check if container runtime is available
        
        Returns:
            True if runtime is available, False otherwise
        """
        runtime = self.config.runtime if self.config else "docker"
        
        cmd = f"which {runtime}"
        result = self.run_command(cmd, capture_output=True)
        
        if result.success:
            logging.debug(f"{runtime} is available at: {result.stdout.strip()}")
            return True
        else:
            logging.error(f"{runtime} is not available")
            return False
    
    def image_exists(self, image_url: str = None) -> bool:
        """
        Check if image exists locally
        
        Args:
            image_name: Image name to check (uses config.image_name if not provided)
            
        Returns:
            True if image exists, False otherwise
        """
        if image_url is None:
            if self.config is None:
                raise ValueError("No image name provided and no config set")
            image_url = f"{self.config.image_home}:{self.config.image_name}"
        
        runtime = self.config.runtime if self.config else "docker"
        
        # Check if image exists locally
        cmd = f"{runtime} images -q {image_url}"
        result = self.run_command(cmd, capture_output=True)
        
        exists = bool(result.stdout.strip())
        
        if exists:
            logging.debug(f"Image '{image_url}' exists locally")
        else:
            logging.warning(f"Image '{image_url}' not found locally")
        
        return exists
    
    def pull_image(self, image_url: str = None, timeout: int = 1800) -> bool:
        """
        Pull image from registry
        
        Args:
            image_name: Image name to pull (uses config.image_name if not provided)
            timeout: Pull timeout in seconds
            
        Returns:
            True if pull successful, False otherwise
        """
        if image_url is None:
            if self.config is None:
                raise ValueError("No image name provided and no config set")
            image_url = f"{self.config.image_home}:{self.config.image_name}"
        runtime = self.config.runtime if self.config else "docker"
        
        logging.info(f"Pulling image: {image_url}")
        
        cmd = f"{runtime} pull {image_url}"
        result = self.run_command(cmd, capture_output=True, timeout=timeout)
        
        if result.success:
            logging.info(f"Successfully pulled image: {image_url}")
            return True
        else:
            logging.error(f"Failed to pull image {image_url}: {result.stderr}")
            return False
    
    def build_image(self, dockerfile_path: str, image_name: str = None, 
                   build_args: Dict[str, str] = None, timeout: int = 3600) -> bool:
        """
        Build image from Dockerfile
        
        Args:
            dockerfile_path: Path to Dockerfile
            image_name: Image name to build (uses config.image_name if not provided)
            build_args: Build arguments
            timeout: Build timeout in seconds
            
        Returns:
            True if build successful, False otherwise
        """
        if image_name is None:
            if self.config is None:
                raise ValueError("No image name provided and no config set")
            image_name = self.config.image_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        # Check if Dockerfile exists
        dockerfile = Path(dockerfile_path)
        if not dockerfile.exists():
            logging.error(f"Dockerfile not found: {dockerfile_path}")
            return False
        
        logging.info(f"Building image from: {dockerfile_path}")
        
        # Prepare build command
        cmd = f"{runtime} build -t {image_name} -f {dockerfile_path} {dockerfile.parent}"
        
        # Add build arguments
        if build_args:
            for key, value in build_args.items():
                cmd += f" --build-arg {key}={value}"
        
        result = self.run_command(cmd, capture_output=True, timeout=timeout)
        
        if result.success:
            logging.info(f"Successfully built image: {image_name}")
            return True
        else:
            logging.error(f"Failed to build image: {result.stderr}")
            return False
    
    def create_container(self, config: ContainerConfig = None) -> Tuple[bool, str]:
        """
        Create container from configuration
        
        Args:
            config: Container configuration (uses self.config if not provided)
            
        Returns:
            Tuple of (success, container_id or error_message)
        """
        if config is None:
            if self.config is None:
                raise ValueError("No container configuration provided")
            config = self.config
        
        runtime = config.runtime
        
        # Check if image exists, pull if not
        image_url = f"{config.image_home}:{config.image_name}"
        if not self.image_exists(image_url):
            logging.info(f"Image '{image_url}' not found, attempting to pull...")
            if not self.pull_image(image_url):
                return False, f"Failed to pull image: {image_url}"
        
        # Prepare docker/nerdctl run command
        cmd_parts = [runtime, "run"]
        
        # Add container name
        if config.container_name:
            cmd_parts.extend(["--name", config.container_name])

        # if config.root_run:
        #     cmd_parts.extend(["-u", os.getuid()])
        # Add detach flag
        if config.detach:
            cmd_parts.append("-d")
        
        # Add auto remove flag
        if config.auto_remove:
            cmd_parts.append("--rm")
        
        # Add privileged flag
        if config.privileged:
            cmd_parts.append("--privileged")
        
        # Add network mode
        if config.network_mode:
            cmd_parts.extend([f"--network={config.network_mode}"])
        ipc_mode = config.ipc_mode or config.network_mode
        if ipc_mode:
            cmd_parts.extend([f"--ipc={ipc_mode}"])
        if config.shm_size:
            cmd_parts.extend([f"--shm-size={config.shm_size}"])
        
        # Add environment variables.
        # Proxy vars keep the historical special-case handling (no_proxy
        # derivation below); every other configured env var is passed through
        # as-is so benchmark runners receive their injected paths
        # (VBENCH2_DATASETS / AIS_THIRD_PARTY / TORCH_HOME / ...).
        for key, value in config.environment.items():
            if key in ("http_proxy", "https_proxy", "no_proxy", "NO_PROXY"):
                continue
            cmd_parts.extend(["-e", f"{key}={value}"])

        # Add environment proxy
        for key, value in config.environment.items():
            if key == "http_proxy" or key == "https_proxy":
                cmd_parts.extend(["-e", f"{key}={value}"])

        # Add no_proxy to exclude local addresses when proxy is set
        if "http_proxy" in config.environment or "https_proxy" in config.environment:
            no_proxy = config.environment.get("no_proxy", "localhost,127.0.0.1,0.0.0.0,11.87.0.0/16,::1")
            cmd_parts.extend(["-e", f"no_proxy={no_proxy}"])
            cmd_parts.extend(["-e", f"NO_PROXY={no_proxy}"])

        # Add port mappings
        # for host_port, container_port in config.ports.items():
        #     cmd_parts.extend(["-p", f"{host_port}:{container_port}"])
        
        cmd_parts.extend(["--device ", "/dev/davinci_manager"])
        cmd_parts.extend(["--device ", "/dev/devmm_svm"])
        cmd_parts.extend(["--device ", "/dev/hisi_hdc"])

        # Add volume mounts
        for host_path, container_path in config.volumes.items():
            # Ensure host path exists
            host_path_obj = Path(host_path)
            if not host_path_obj.exists():
                logging.warning(f"Host path does not exist, creating: {host_path}")
                host_path_obj.mkdir(parents=True, exist_ok=True)
            cmd_parts.extend(["-v", f"{host_path}:{container_path}"])
        
        # if config.gpu_ids and runtime == "docker":
        #     # For docker with NVIDIA runtime
        #     cmd_parts.extend(["--gpus", f'"device={",".join(map(str, config.gpu_ids))}"'])
        # elif config.gpu_ids and runtime == "nerdctl":
        #     # For nerdctl with NVIDIA runtime (assuming nvidia-container-runtime is configured)
        #     cmd_parts.extend(["--gpus", f'{",".join(map(str, config.gpu_ids))}'])
        
        # Add working directory
        if config.working_dir:
            cmd_parts.extend(["-w", config.working_dir])
        
        # Add stdin and tty flags
        if config.stdin_open:
            cmd_parts.append("-i")
        if config.tty:
            cmd_parts.append("-t")
        
        # Add image name
        cmd_parts.append(f"{config.image_home}:{config.image_name}")
        
        # Add entrypoint if specified
        if config.entrypoint:
            cmd_parts.extend(["--entrypoint", config.entrypoint])
        
        # Add command if specified
        if config.command:
            cmd_parts.append(config.command)
        
        cmd_parts.extend(["bash"])
        # Build command string
        cmd = " ".join(cmd_parts)
        logging.info(f"Creating container with command: {cmd}")
        
        # Execute command
        result = self.run_command(cmd, capture_output=True, timeout=60)
        
        if result.success:
            container_id = result.stdout.strip()
            self.container_id = container_id
            logging.info(f"Container created successfully with ID: {container_id}")
            return True, container_id
        else:
            error_msg = result.stderr.strip()
            logging.error(f"Failed to create container: {error_msg}")
            return False, error_msg
    
    def start_container(self, container_name: str = None) -> bool:
        """
        Start existing container
        
        Args:
            container_name: Container name to start (uses config.container_name if not provided)
            
        Returns:
            True if start successful, False otherwise
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        cmd = f"{runtime} start {container_name}"
        result = self.run_command(cmd, capture_output=True)
        
        if result.success:
            logging.info(f"Container '{container_name}' started successfully")
            return True
        else:
            logging.error(f"Failed to start container '{container_name}': {result.stderr}")
            return False
    
    def stop_container(self, container_name: str = None, timeout: int = 30) -> bool:
        """
        Stop running container
        
        Args:
            container_name: Container name to stop (uses config.container_name if not provided)
            timeout: Stop timeout in seconds
            
        Returns:
            True if stop successful, False otherwise
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        cmd = f"{runtime} stop --time={timeout} {container_name}"
        result = self.run_command(cmd, capture_output=True)
        
        if result.success:
            logging.info(f"Container '{container_name}' stopped successfully")
            return True
        else:
            # Try force kill if normal stop fails
            logging.warning(f"Normal stop failed, trying force kill...")
            cmd = f"{runtime} kill {container_name}"
            result = self.run_command(cmd, capture_output=True)
            
            if result.success:
                logging.info(f"Container '{container_name}' force killed")
                return True
            else:
                logging.error(f"Failed to stop container '{container_name}': {result.stderr}")
                return False
    
    def remove_container(self, container_name: str = None, force: bool = False) -> bool:
        """
        Remove container
        
        Args:
            container_name: Container name to remove (uses config.container_name if not provided)
            force: Force removal even if container is running
            
        Returns:
            True if removal successful, False otherwise
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        cmd = f"{runtime} rm"
        if force:
            cmd += " -f"
        cmd += f" {container_name}"
        
        result = self.run_command(cmd, capture_output=True)
        
        if result.success:
            logging.info(f"Container '{container_name}' removed successfully")
            return True
        else:
            logging.error(f"Failed to remove container '{container_name}': {result.stderr}")
            return False
    
    def container_exists(self, container_name: str = None) -> bool:
        """
        Check if container exists
        
        Args:
            container_name: Container name to check (uses config.container_name if not provided)
            
        Returns:
            True if container exists, False otherwise
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        # Check if container exists (including stopped containers)
        cmd = f"{runtime} ps -a --filter name={container_name} --format '{{{{.Names}}}}'"
        result = self.run_command(cmd, capture_output=True)
        
        exists = container_name in result.stdout
        status = "exists" if exists else "does not exist"
        logging.debug(f"Container '{container_name}' {status}")
        
        return exists
    
    def container_is_running(self, container_name: str = None) -> bool:
        """
        Check if container is running
        
        Args:
            container_name: Container name to check (uses config.container_name if not provided)
            
        Returns:
            True if container is running, False otherwise
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        cmd = f"{runtime} ps --filter name={container_name} --filter status=running"
        result = self.run_command(cmd, capture_output=True)
        
        is_running = container_name in result.stdout
        status = "running" if is_running else "not running"
        logging.debug(f"Container '{container_name}' is {status}")
        
        return is_running
    
    def exec_in_container(self, container_name: str = None, cmd: str = "", 
                         capture_output: bool = True, timeout: int = 300,
                         workdir: str = None, user: str = None,
                         environment: Dict[str, str] = None,
                         abort_event=None, abort_poll_interval: float = 300.0) -> CommandResult:
        """
        Execute command in container
        
        Args:
            container_name: Container name (uses config.container_name if not provided)
            cmd: Command to execute
            capture_output: Whether to capture stdout/stderr
            timeout: Command timeout in seconds
            workdir: Working directory inside container
            user: User to run command as
            environment: Environment variables to set
            abort_event: 可选中断信号，置位后立即终止容器命令执行，主流程继续
            abort_poll_interval: 中断轮询间隔（秒）
            
        Returns:
            CommandResult object
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        # Check if container not exists or not running
        # NOTE: container_exists issues a remote docker query (SSH on remote
        # hosts).  A transient SSH/docker failure returns an empty stdout and
        # is indistinguishable from a truly deleted container, which would
        # abort perfectly valid exec calls (e.g. the readiness probe during
        # service startup) with "does not exist".  Retry once after a short
        # delay so only a stable negative verdict is honored.
        if not self.container_exists(container_name):
            import time as _time

            _time.sleep(2)
            if not self.container_exists(container_name):
                return CommandResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Container '{container_name}' does not exist",
                    execution_time=0.0
                )
            logging.info(
                "[%s] Container '%s' existence check recovered on retry "
                "(transient remote failure)",
                inspect.currentframe().f_code.co_name,
                container_name,
            )
        else:
            logging.debug("[%s] Check %s container is exist" % 
                          (inspect.currentframe().f_code.co_name, container_name))

        if not self.container_is_running(container_name):
            # Try to start the container
            logging.info(f"Container '{container_name}' is not running, attempting to start...")
            if not self.start_container(container_name):
                return CommandResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Container '{container_name}' is not running and could not be started",
                    execution_time=0.0
                )
        else:
            logging.debug("[%s] Check %s container is running" % 
                          (inspect.currentframe().f_code.co_name, container_name))
        
        # Build exec command
        exec_parts = [runtime, "exec"]
        
        # Add interactive flag if we want to capture output
        if capture_output:
            exec_parts.append("-i")
        
        # Add working directory
        if workdir:
            exec_parts.extend(["-w", workdir])
        
        # Add user
        if user:
            exec_parts.extend(["-u", user])
        
        # Add environment variables
        if environment:
            for key, value in environment.items():
                exec_parts.extend(["-e", f"{key}={value}"])
        
        # Add container name
        exec_parts.append(container_name)
        
        # Add command
        exec_parts.extend(["sh", "-c", f"\"{cmd}\""])
        
        # Build command string
        exec_cmd = " ".join(exec_parts)
        logging.info("exec_in_container exec_cmd is: %s"% redact(exec_cmd))
        logging.info("Executing in container '%s': %s", container_name, redact(cmd))
        
        # Execute command
        return self.run_command(exec_cmd, capture_output=capture_output, timeout=timeout,
                                abort_event=abort_event, abort_poll_interval=abort_poll_interval)
    
    def copy_to_container(self, container_name: str = None, src_path: str = "", 
                         dest_path: str = "") -> bool:
        """
        Copy file/directory from host to container
        
        Args:
            container_name: Container name (uses config.container_name if not provided)
            src_path: Source path on host
            dest_path: Destination path in container
            
        Returns:
            True if copy successful, False otherwise
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        # Check if source exists
        src = Path(src_path)
        if not src.exists():
            logging.error(f"Source path does not exist: {src_path}")
            return False
        
        cmd = f"{runtime} cp {src_path} {container_name}:{dest_path}"
        result = self.run_command(cmd, capture_output=True)
        
        if result.success:
            logging.info(f"Successfully copied {src_path} to {container_name}:{dest_path}")
            return True
        else:
            logging.error(f"Failed to copy: {result.stderr}")
            return False
    
    def copy_from_container(self, container_name: str = None, src_path: str = "", 
                           dest_path: str = "") -> bool:
        """
        Copy file/directory from container to host
        
        Args:
            container_name: Container name (uses config.container_name if not provided)
            src_path: Source path in container
            dest_path: Destination path on host
            
        Returns:
            True if copy successful, False otherwise
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        # Ensure destination directory exists
        dest = Path(dest_path)
        if dest.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = f"{runtime} cp {container_name}:{src_path} {dest_path}"
        result = self.run_command(cmd, capture_output=True)
        
        if result.success:
            logging.info(f"Successfully copied {container_name}:{src_path} to {dest_path}")
            return True
        else:
            logging.error(f"Failed to copy: {result.stderr}")
            return False
    
    def get_container_info(self, container_name: str = None) -> Dict[str, Any]:
        """
        Get container information
        
        Args:
            container_name: Container name (uses config.container_name if not provided)
            
        Returns:
            Dictionary with container information
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        # Get container info in JSON format
        cmd = f"{runtime} inspect {container_name}"
        result = self.run_command(cmd, capture_output=True)
        
        if not result.success:
            logging.error(f"Failed to get container info: {result.stderr}")
            return {}
        
        try:
            info_list = json.loads(result.stdout)
            if info_list:
                return info_list[0]
            else:
                return {}
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse container info JSON: {str(e)}")
            return {}
    
    def get_container_logs(self, container_name: str = None, tail: int = 100,
                          follow: bool = False, timeout: int = 30) -> str:
        """
        Get container logs
        
        Args:
            container_name: Container name (uses config.container_name if not provided)
            tail: Number of lines to show from the end
            follow: Follow log output
            timeout: Timeout for follow mode
            
        Returns:
            Log output as string
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        runtime = self.config.runtime if self.config else "docker"
        
        cmd = f"{runtime} logs --tail {tail}"
        if follow:
            cmd += " -f"
        cmd += f" {container_name}"
        
        if follow:
            # For follow mode, we need a different approach
            result = self.run_command(cmd, capture_output=True, timeout=timeout)
        else:
            result = self.run_command(cmd, capture_output=True)
        
        if result.success:
            return result.stdout
        else:
            logging.error(f"Failed to get container logs: {result.stderr}")
            return ""
    
    def wait_for_container(self, container_name: str = None, 
                          timeout: int = 300, check_interval: int = 5) -> bool:
        """
        Wait for container to be ready (by checking if it's running)
        
        Args:
            container_name: Container name (uses config.container_name if not provided)
            timeout: Maximum wait time in seconds
            check_interval: Interval between checks in seconds
            
        Returns:
            True if container is ready within timeout, False otherwise
        """
        if container_name is None:
            if self.config is None:
                raise ValueError("No container name provided and no config set")
            container_name = self.config.container_name
        
        logging.info(f"Waiting for container '{container_name}' to be ready (timeout: {timeout}s)")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.container_is_running(container_name):
                logging.info(f"Container '{container_name}' is ready")
                return True
            
            # Check if container exists but is not running
            if self.container_exists(container_name) and not self.container_is_running(container_name):
                logging.info(f"Container '{container_name}' exists but is not running, \
                             attempting to start...")
                if self.start_container(container_name):
                    continue
                else:
                    logging.error(f"Failed to start container '{container_name}'")
                    return False
            
            logging.debug(f"Container not ready yet, waiting {check_interval}s...")
            time.sleep(check_interval)
        
        logging.error(f"Timeout waiting for container '{container_name}' to be ready")
        return False
    
    def setup_container(self) -> Tuple[bool, str]:
        """
        Setup container based on configuration:
        1. Check if image exists, pull if not
        2. Create or start container
        
        Returns:
            Tuple of (success, container_id or error_message)
        """
        if self.config is None:
            raise ValueError("No container configuration provided")
        
        # Check if container already exists and is running
        if self.container_exists(self.config.container_name):
            if self.container_is_running(self.config.container_name):
                logging.info(f"Container '{self.config.container_name}' already running")
                return True, self.config.container_name
            else:
                # Container exists but not running, start it
                if self.start_container(self.config.container_name):
                    return True, self.config.container_name
                else:
                    return False, f"Failed to start existing container: {self.config.container_name}"
        
        # Container doesn't exist, create it
        return self.create_container()
    
    def run_in_container_and_wait(self, cmd: str, check_ready: bool = True,
                                 ready_timeout: int = 300, exec_timeout: int = 3600) -> CommandResult:
        """
        Setup container and execute command, waiting for completion
        
        Args:
            cmd: Command to execute
            check_ready: Whether to wait for container to be ready
            ready_timeout: Timeout for container readiness check
            exec_timeout: Timeout for command execution
            
        Returns:
            CommandResult object
        """
        # Setup container
        success, container_id = self.setup_container()
        if not success:
            return CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Failed to setup container: {container_id}",
                execution_time=0.0
            )
        
        # Wait for container to be ready if requested
        if check_ready:
            if not self.wait_for_container(container_id, timeout=ready_timeout):
                return CommandResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Container '{container_id}' not ready within {ready_timeout}s",
                    execution_time=0.0
                )
        
        # Execute command in container
        return self.exec_in_container(
            container_name=container_id,
            cmd=cmd,
            capture_output=True,
            timeout=exec_timeout
        )
    
    def cleanup(self, stop_timeout: int = 30):
        """Cleanup container resources"""
        logging.debug("entry function: %s" % inspect.currentframe().f_code.co_name)
        if self.config and self.config.container_name:
            logging.info(f"Cleaning up container: {self.config.container_name}")
            
            # Stop container
            self.stop_container(self.config.container_name, timeout=stop_timeout)
            
            # Remove container if not auto-remove
            if not self.config.auto_remove:
                self.remove_container(self.config.container_name, force=True)
        
        self.container_id = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.cleanup()


# Utility functions for common container operations
def create_inference_container(model_config: Dict[str, Any], 
                              inference_config: Dict[str, Any],
                              runtime: str = "docker") -> ContainerManager:
    """
    Create container manager for model inference
    
    Args:
        model_config: Model configuration
        inference_config: Inference configuration
        runtime: Container runtime (docker/nerdctl)
        
    Returns:
        ContainerManager instance
    """
    # Generate container name
    model_name = model_config.get('name', 'unknown').replace('/', '-')
    container_name = f"inference-{model_name}-{int(time.time())}"
    
    # Prepare container configuration
    container_config = ContainerConfig(
        image_home=inference_config.get('container_home', 
                                       f"quay.io/jd_xllm/xllm-ai"),
        image_name=inference_config.get('container_image', f"latest"),
        container_name=container_name,
        runtime=runtime,
        gpu_ids=inference_config.get('gpu_ids', []),
        ports={str(inference_config.get('port', 8000)): "8000"},
        volumes={
            model_config.get('path', '.'): "/models",
            "./results": "/results",
            "./logs": "/logs"
        },
        environment={
            "MODEL_PATH": "/models",
            "MODEL_NAME": model_config.get('name', ''),
            "MODEL_FRAMEWORK": model_config.get('framework', 'tensorrt'),
            "PORT": str(inference_config.get('port', 8000))
        },
        working_dir="/workspace",
        network_mode="host",
        privileged=True,
        auto_remove=True,
        detach=True
    )
    
    return ContainerManager(container_config)


def path_mapping_in_container(path: str):
    volumes = get_xtask_config_data().get("DOCKER_CONFIG").get("volumes")
    # Longest host prefix wins: a generic mount such as /export/home/models/
    # must not shadow a more specific one like the DEPENDENCE_PACKAGE_PATH
    # subtree, which maps to the fixed /xllm_test_060 container root.
    for outside_path, inside_path in sorted(
        volumes.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if outside_path in path:
            return path.replace(outside_path, inside_path + "/").replace("//", "/")
    return path


def ensure_host_bind_mount(volumes: Dict[str, str], host_path: str) -> None:
    """Bind-mount a host path into the container at the same absolute path.

    Configured ``DOCKER_CONFIG.volumes`` mappings take precedence.  Paths
    outside those mappings (for example models under ``/mnt/cfs``) must be
    added explicitly or the container cannot see them.
    """
    if not host_path:
        return

    normalized = os.path.abspath(os.path.normpath(str(host_path)))
    if not os.path.exists(normalized):
        return

    mapped = path_mapping_in_container(normalized)
    if mapped != normalized:
        return

    for existing_host in volumes:
        existing_host_norm = os.path.abspath(
            os.path.normpath(str(existing_host).rstrip(os.sep))
        )
        if normalized == existing_host_norm or normalized.startswith(
            existing_host_norm + os.sep
        ):
            return

    volumes[normalized] = normalized
