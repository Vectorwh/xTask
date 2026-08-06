# framework/container/container_watchdog.py
import os
import re
import sys
import time
import shlex
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from multiprocessing import Process, Event, Queue
import json
import psutil
import logging
from datetime import datetime
from enum import Enum


class ProcessStatus(Enum):
    """Process status enumeration"""
    RUNNING = "running"
    STOPPED = "stopped"
    ZOMBIE = "zombie"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class WatchdogConfig:
    """Watchdog configuration"""
    check_interval: int = 5  # seconds between checks
    max_restart_attempts: int = 3  # maximum restart attempts
    restart_delay: int = 10  # seconds to wait before restart
    health_check_timeout: int = 30  # timeout for health checks
    log_file: str = "watchdog.log"
    enable_restart: bool = True  # whether to restart failed processes
    notify_on_failure: bool = True  # whether to send notifications on failure
    max_memory_mb: int = 4096  # maximum memory limit in MB
    max_cpu_percent: int = 90  # maximum CPU usage percentage


@dataclass
class ProcessInfo:
    """Process information"""
    pid: int
    name: str
    cmdline: str
    status: ProcessStatus
    cpu_percent: float
    memory_mb: float
    create_time: float
    parent_pid: int = None
    children_pids: List[int] = None
    
    def __post_init__(self):
        if self.children_pids is None:
            self.children_pids = []


class ProcessWatchdog:
    """Process watchdog for monitoring and managing processes"""
    
    def __init__(self, container_manager, config: WatchdogConfig = None, logger=None):
        """
        Initialize process watchdog
        
        Args:
            container_manager: ContainerManager instance
            config: Watchdog configuration
            logger: Logger instance
        """
        self.container_manager = container_manager
        self.config = config or WatchdogConfig()
        self.logger = logger or logging.getLogger("ProcessWatchdog")
        
        # Internal state
        self._watchdog_thread = None
        self._stop_event = threading.Event()
        self._monitored_processes = {}  # cmd -> ProcessInfo
        self._restart_counts = {}  # cmd -> restart count
        self._health_check_callbacks = {}  # cmd -> health check function
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for watchdog"""
        if not self.logging.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logging.addHandler(console_handler)
            
            # File handler
            if self.config.log_file:
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setFormatter(formatter)
                self.logging.addHandler(file_handler)
            
            self.logging.setLevel(logging.INFO)
    
    def log(self, message: str, level: str = "info", exception=None):
        """Log message with specified level"""
        log_func = getattr(self.logger, level, self.logging.info)
        if exception:
            log_func(f"{message}: {exception}", exc_info=exception)
        else:
            log_func(message)
    
    def start_monitoring(self, container_name: str, cmd: str, 
                        health_check_func: Callable = None) -> bool:
        """
        Start monitoring a process in container
        
        Args:
            container_name: Container name
            cmd: Command to monitor
            health_check_func: Custom health check function
            
        Returns:
            True if monitoring started successfully
        """
        try:
            # Get container PID
            container_pid = self._get_container_pid(container_name)
            if not container_pid:
                self.log(f"Container '{container_name}' not found or not running", "error")
                return False
            
            # Start the process in container
            result = self.container_manager.exec_in_container(
                container_name=container_name,
                cmd=cmd,
                capture_output=True,
                timeout=60
            )
            
            if not result.success:
                self.log(f"Failed to start process in container: {result.stderr}", "error")
                return False
            
            # Parse PID from output (assuming the command prints its PID)
            pid = self._extract_pid(result.stdout)
            if not pid:
                # If PID not found in output, try to find it by command
                pid = self._find_process_by_cmd(container_pid, cmd)
            
            if not pid:
                self.log(f"Could not determine PID for command: {cmd}", "warning")
                # We'll still monitor, but will rely on container state
                pid = -1
            
            # Store process info
            process_info = ProcessInfo(
                pid=pid,
                name=self._extract_process_name(cmd),
                cmdline=cmd,
                status=ProcessStatus.RUNNING if pid > 0 else ProcessStatus.NOT_FOUND,
                cpu_percent=0.0,
                memory_mb=0.0,
                create_time=time.time(),
                parent_pid=container_pid
            )
            
            self._monitored_processes[cmd] = process_info
            self._restart_counts[cmd] = 0
            
            if health_check_func:
                self._health_check_callbacks[cmd] = health_check_func
            
            self.log(f"Started monitoring process: {cmd} (PID: {pid})")
            return True
            
        except Exception as e:
            self.log(f"Failed to start monitoring: {str(e)}", "error", e)
            return False
    
    def stop_monitoring(self, cmd: str) -> bool:
        """
        Stop monitoring a process
        
        Args:
            cmd: Command being monitored
            
        Returns:
            True if monitoring stopped successfully
        """
        if cmd not in self._monitored_processes:
            self.log(f"Process not being monitored: {cmd}", "warning")
            return False
        
        try:
            process_info = self._monitored_processes[cmd]
            
            # Stop the process if it's running
            if process_info.pid > 0 and self._is_process_running(process_info.pid):
                self._terminate_process_tree(process_info.pid)
                self.log(f"Stopped process: {cmd} (PID: {process_info.pid})")
            
            # Remove from monitoring
            del self._monitored_processes[cmd]
            if cmd in self._health_check_callbacks:
                del self._health_check_callbacks[cmd]
            
            return True
            
        except Exception as e:
            self.log(f"Failed to stop monitoring: {str(e)}", "error", e)
            return False
    
    def start_watchdog(self):
        """Start the watchdog monitoring thread"""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self.log("Watchdog already running", "warning")
            return
        
        self._stop_event.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="ProcessWatchdog",
            daemon=True
        )
        self._watchdog_thread.start()
        self.log("Watchdog started")
    
    def stop_watchdog(self):
        """Stop the watchdog monitoring thread"""
        self._stop_event.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=30)
            self._watchdog_thread = None
        self.log("Watchdog stopped")
    
    def _watchdog_loop(self):
        """Main watchdog monitoring loop"""
        self.log("Starting watchdog monitoring loop")
        
        while not self._stop_event.is_set():
            try:
                self._check_all_processes()
                self._check_container_health()
                
                # Wait for next check
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.log(f"Error in watchdog loop: {str(e)}", "error", e)
                time.sleep(1)  # Prevent tight loop on errors
    
    def _check_all_processes(self):
        """Check status of all monitored processes"""
        for cmd, process_info in list(self._monitored_processes.items()):
            try:
                self._check_process_health(cmd, process_info)
            except Exception as e:
                self.log(f"Error checking process {cmd}: {str(e)}", "error", e)
    
    def _check_process_health(self, cmd: str, process_info: ProcessInfo):
        """Check health of a single process"""
        # Check if process is running
        if process_info.pid <= 0:
            # Process with unknown PID, check by command
            container_pid = process_info.parent_pid
            new_pid = self._find_process_by_cmd(container_pid, cmd)
            
            if new_pid:
                # Found the process
                process_info.pid = new_pid
                process_info.status = ProcessStatus.RUNNING
                self.log(f"Found process for command {cmd} with PID: {new_pid}")
            else:
                # Process not found
                process_info.status = ProcessStatus.NOT_FOUND
                self._handle_process_failure(cmd, process_info)
                return
        
        # Check process status
        try:
            proc = psutil.Process(process_info.pid)
            
            # Update process info
            with proc.oneshot():
                process_info.status = ProcessStatus.RUNNING
                process_info.cpu_percent = proc.cpu_percent()
                process_info.memory_mb = proc.memory_info().rss / 1024 / 1024
                process_info.children_pids = [p.pid for p in proc.children(recursive=True)]
            
            # Check resource limits
            if self._check_resource_limits(process_info):
                self.log(f"Process {cmd} exceeds resource limits, terminating", "warning")
                self._terminate_process_tree(process_info.pid)
                self._handle_process_failure(cmd, process_info)
                return
            
            # Perform custom health check
            if cmd in self._health_check_callbacks:
                try:
                    health_ok = self._health_check_callbacks[cmd](process_info)
                    if not health_ok:
                        self.log(f"Health check failed for process {cmd}", "warning")
                        self._handle_process_failure(cmd, process_info)
                        return
                except Exception as e:
                    self.log(f"Health check error for {cmd}: {str(e)}", "error", e)
            
            # Check if process is still responding
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                process_info.status = ProcessStatus.ZOMBIE
                self._handle_process_failure(cmd, process_info)
                
        except psutil.NoSuchProcess:
            # Process has died
            process_info.status = ProcessStatus.STOPPED
            self._handle_process_failure(cmd, process_info)
            
        except Exception as e:
            process_info.status = ProcessStatus.ERROR
            self.log(f"Error checking process {cmd}: {str(e)}", "error", e)
            self._handle_process_failure(cmd, process_info)
    
    def _check_resource_limits(self, process_info: ProcessInfo) -> bool:
        """Check if process exceeds resource limits"""
        if process_info.memory_mb > self.config.max_memory_mb:
            self.log(f"Process memory {process_info.memory_mb:.1f}MB exceeds limit "
                    f"{self.config.max_memory_mb}MB", "warning")
            return True
        
        if process_info.cpu_percent > self.config.max_cpu_percent:
            self.log(f"Process CPU {process_info.cpu_percent}% exceeds limit "
                    f"{self.config.max_cpu_percent}%", "warning")
            return True
        
        return False
    
    def _handle_process_failure(self, cmd: str, process_info: ProcessInfo):
        """Handle process failure"""
        self.log(f"Process failure detected: {cmd} (Status: {process_info.status.value})", "warning")
        
        # Increment restart count
        self._restart_counts[cmd] = self._restart_counts.get(cmd, 0) + 1
        
        if self.config.enable_restart:
            if self._restart_counts[cmd] <= self.config.max_restart_attempts:
                self.log(f"Attempting to restart process ({self._restart_counts[cmd]}/"
                        f"{self.config.max_restart_attempts}): {cmd}")
                time.sleep(self.config.restart_delay)
                self._restart_process(cmd, process_info)
            else:
                self.log(f"Max restart attempts reached for process: {cmd}", "error")
                self._escalate_failure(cmd, process_info)
        else:
            self.log(f"Automatic restart disabled for process: {cmd}", "warning")
            self._escalate_failure(cmd, process_info)
    
    def _restart_process(self, cmd: str, old_process_info: ProcessInfo):
        """Restart a failed process"""
        try:
            container_name = self.container_manager.config.container_name
            
            # Stop old process if still running
            if old_process_info.pid > 0 and self._is_process_running(old_process_info.pid):
                self._terminate_process_tree(old_process_info.pid)
            
            # Start new process
            result = self.container_manager.exec_in_container(
                container_name=container_name,
                cmd=cmd,
                capture_output=True,
                timeout=60
            )
            
            if result.success:
                new_pid = self._extract_pid(result.stdout) or -1
                old_process_info.pid = new_pid
                old_process_info.status = ProcessStatus.RUNNING
                old_process_info.create_time = time.time()
                self.log(f"Successfully restarted process: {cmd} (New PID: {new_pid})")
            else:
                self.log(f"Failed to restart process: {result.stderr}", "error")
                self._escalate_failure(cmd, old_process_info)
                
        except Exception as e:
            self.log(f"Error restarting process {cmd}: {str(e)}", "error", e)
            self._escalate_failure(cmd, old_process_info)
    
    def _escalate_failure(self, cmd: str, process_info: ProcessInfo):
        """Escalate process failure (destroy container)"""
        self.log(f"Escalating failure for process: {cmd} - Destroying container", "critical")
        
        # Notify about failure
        if self.config.notify_on_failure:
            self._notify_failure(cmd, process_info)
        
        # Destroy container
        self._destroy_container()
    
    def _destroy_container(self):
        """Destroy the container"""
        try:
            container_name = self.container_manager.config.container_name
            self.log(f"Destroying container: {container_name}")
            
            # Stop watchdog first
            self.stop_watchdog()
            
            # Stop and remove container
            self.container_manager.stop_container(container_name)
            self.container_manager.remove_container(container_name, force=True)
            
            # Log destruction
            self.log(f"Container {container_name} destroyed successfully")
            
        except Exception as e:
            self.log(f"Failed to destroy container: {str(e)}", "error", e)
    
    def _check_container_health(self):
        """Check overall container health"""
        try:
            container_name = self.container_manager.config.container_name
            
            # Check if container is running
            if not self.container_manager.container_is_running(container_name):
                self.log(f"Container {container_name} is not running", "critical")
                self._destroy_container()
                return
            
            # Get container stats
            container_info = self.container_manager.get_container_info(container_name)
            if not container_info:
                self.log(f"Could not get container info for {container_name}", "warning")
                return
            
            # Check container state
            state = container_info.get("State", {})
            if state.get("Status") != "running":
                self.log(f"Container {container_name} state is {state.get('Status')}", "critical")
                self._destroy_container()
                return
            
            # Check exit code (if container previously exited)
            exit_code = state.get("ExitCode")
            if exit_code and exit_code != 0:
                self.log(f"Container {container_name} exited with code {exit_code}", "critical")
                self._destroy_container()
                return
            
        except Exception as e:
            self.log(f"Error checking container health: {str(e)}", "error", e)
    
    def _get_container_pid(self, container_name: str) -> Optional[int]:
        """Get the host PID of container's main process"""
        try:
            container_info = self.container_manager.get_container_info(container_name)
            if container_info:
                state = container_info.get("State", {})
                return state.get("Pid")
        except Exception as e:
            self.log(f"Error getting container PID: {str(e)}", "error", e)
        return None
    
    def _find_process_by_cmd(self, container_pid: int, cmd: str) -> Optional[int]:
        """Find process by command in container's process tree"""
        try:
            # Get container process
            container_proc = psutil.Process(container_pid)
            
            # Search through all processes in container's namespace
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
                try:
                    # Check if process is in container's subtree
                    if self._is_descendant(proc.ppid(), container_pid):
                        # Check if command matches
                        if cmd in ' '.join(proc.cmdline()):
                            return proc.pid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.log(f"Error finding process by cmd: {str(e)}", "error", e)
        
        return None
    
    def _is_descendant(self, pid: int, ancestor_pid: int) -> bool:
        """Check if process is descendant of ancestor"""
        if pid == ancestor_pid:
            return True
        
        try:
            proc = psutil.Process(pid)
            while True:
                parent = proc.parent()
                if not parent:
                    return False
                if parent.pid == ancestor_pid:
                    return True
                proc = parent
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if process is running"""
        try:
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except psutil.NoSuchProcess:
            return False
    
    def _terminate_process_tree(self, pid: int):
        """Terminate process and all its children"""
        try:
            # Get process tree
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Terminate children first
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Wait for children to terminate
            gone, alive = psutil.wait_procs(children, timeout=5)
            
            # Force kill any remaining children
            for child in alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Terminate parent
            try:
                parent.terminate()
                parent.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                # Force kill if still running
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                    
        except psutil.NoSuchProcess:
            pass  # Process already terminated
        except Exception as e:
            self.log(f"Error terminating process tree {pid}: {str(e)}", "error", e)
    
    def _extract_pid(self, output: str) -> Optional[int]:
        """Extract PID from command output"""
        try:
            lines = output.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.isdigit():
                    return int(line)
                # Look for patterns like "PID: 1234" or "Started with PID 1234"
                import re
                patterns = [
                    r'PID[:\s]+(\d+)',
                    r'pid[=\s]+(\d+)',
                    r'Process ID[:\s]+(\d+)',
                    r'Started with PID (\d+)'
                ]
                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
        except (ValueError, AttributeError):
            pass
        return None
    
    def _extract_process_name(self, cmd: str) -> str:
        """Extract process name from command"""
        # Extract first non-option word (usually the executable)
        parts = cmd.strip().split()
        for part in parts:
            if not part.startswith('-'):
                # Remove path, keep just the name
                return Path(part).name
        return "unknown"
    
    def _notify_failure(self, cmd: str, process_info: ProcessInfo):
        """Send notification about process failure"""
        try:
            notification = {
                "timestamp": datetime.now().isoformat(),
                "type": "process_failure",
                "container": self.container_manager.config.container_name,
                "process": {
                    "cmd": cmd,
                    "pid": process_info.pid,
                    "name": process_info.name,
                    "status": process_info.status.value,
                    "memory_mb": process_info.memory_mb,
                    "cpu_percent": process_info.cpu_percent
                },
                "action": "container_destroyed",
                "restart_attempts": self._restart_counts.get(cmd, 0)
            }
            
            # Write to notification log
            log_file = Path(self.config.log_file).parent / "notifications.json"
            with open(log_file, 'a') as f:
                f.write(json.dumps(notification) + '\n')
            
            # You could add other notification methods here:
            # - Send email
            # - Send to Slack/Teams
            # - Send to monitoring system
            
            self.log(f"Failure notification sent for process: {cmd}")
            
        except Exception as e:
            self.log(f"Error sending notification: {str(e)}", "error", e)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "watchdog_running": self._watchdog_thread and self._watchdog_thread.is_alive(),
            "monitored_processes": len(self._monitored_processes),
            "processes": [
                {
                    "cmd": info.cmdline,
                    "pid": info.pid,
                    "status": info.status.value,
                    "cpu_percent": info.cpu_percent,
                    "memory_mb": info.memory_mb,
                    "restart_count": self._restart_counts.get(info.cmdline, 0),
                    "uptime_seconds": time.time() - info.create_time
                }
                for info in self._monitored_processes.values()
            ],
            "config": asdict(self.config)
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.stop_watchdog()


# Updated ContainerManager with watchdog integration
class ContainerManagerWithWatchdog:
    """Enhanced Container Manager with Process Watchdog"""
    
    def __init__(self, config=None, logger=None):
        """Initialize with container and watchdog managers"""
        from .container_manager import ContainerManager
        
        self.container_manager = ContainerManager(config, logger)
        self.watchdog = ProcessWatchdog(self.container_manager, logger=logger)
        
        # Store original exec_in_container for reference
        self._original_exec = self.container_manager.exec_in_container
    
    def exec_in_container_with_watchdog(self, container_name: str = None, cmd: str = "",
                                       health_check_func: Callable = None,
                                       enable_watchdog: bool = True) -> Any:
        """
        Execute command in container with watchdog monitoring
        
        Args:
            container_name: Container name
            cmd: Command to execute
            health_check_func: Custom health check function
            enable_watchdog: Whether to enable watchdog monitoring
            
        Returns:
            CommandResult object
        """
        # Execute the command
        result = self.container_manager.exec_in_container(
            container_name=container_name,
            cmd=cmd,
            capture_output=True
        )
        
        # Start watchdog monitoring if enabled
        if enable_watchdog and result.success:
            container_name = container_name or self.container_manager.config.container_name
            self.watchdog.start_monitoring(container_name, cmd, health_check_func)
            
            # Start watchdog thread if not already running
            if not (self.watchdog._watchdog_thread and 
                   self.watchdog._watchdog_thread.is_alive()):
                self.watchdog.start_watchdog()
        
        return result
    
    def run_service_with_watchdog(self, container_name: str = None, 
                                 service_cmd: str = "",
                                 health_check_cmd: str = "",
                                 health_check_interval: int = 30) -> Any:
        """
        Run a service command with watchdog monitoring
        
        Args:
            container_name: Container name
            service_cmd: Service command to run
            health_check_cmd: Command to check service health
            health_check_interval: Health check interval in seconds
            
        Returns:
            CommandResult object
        """
        # Define health check function
        def health_check_func(process_info):
            if not health_check_cmd:
                # Default health check: check if process is running
                return self.watchdog._is_process_running(process_info.pid)
            
            # Custom health check command
            result = self.container_manager.exec_in_container(
                container_name=container_name,
                cmd=health_check_cmd,
                capture_output=True,
                timeout=self.watchdog.config.health_check_timeout
            )
            return result.success and result.exit_code == 0
        
        # Run service with watchdog
        return self.exec_in_container_with_watchdog(
            container_name=container_name,
            cmd=service_cmd,
            health_check_func=health_check_func,
            enable_watchdog=True
        )
    
    def start_inference_service(self, container_name: str = None,
                               model_path: str = "",
                               port: int = 8000,
                               framework: str = "tensorrt") -> Any:
        """
        Start inference service with watchdog monitoring
        
        Args:
            container_name: Container name
            model_path: Path to model
            port: Service port
            framework: Inference framework
            
        Returns:
            CommandResult object
        """
        # Build service command based on framework
        if framework == "tensorrt":
            service_cmd = f"trt_server --model {model_path} --port {port}"
            health_check_cmd = f"curl -f http://localhost:{port}/health"
        elif framework == "vllm":
            service_cmd = f"python -m vllm.entrypoints.api_server --model {model_path} --port {port}"
            health_check_cmd = f"curl -f http://localhost:{port}/health"
        elif framework == "tgi":
            service_cmd = f"text-generation-server --model-id {model_path} --port {port}"
            health_check_cmd = f"curl -f http://localhost:{port}/health"
        else:
            service_cmd = f"python inference_server.py --model {model_path} --port {port}"
            health_check_cmd = f"curl -f http://localhost:{port}/health"
        
        # Run service with watchdog
        return self.run_service_with_watchdog(
            container_name=container_name,
            service_cmd=service_cmd,
            health_check_cmd=health_check_cmd
        )
    
    # Delegate other methods to container_manager
    def __getattr__(self, name):
        """Delegate unknown attributes to container_manager"""
        if hasattr(self.container_manager, name):
            return getattr(self.container_manager, name)
        elif hasattr(self.watchdog, name):
            return getattr(self.watchdog, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def cleanup(self):
        """Cleanup both container and watchdog"""
        self.watchdog.stop_watchdog()
        self.container_manager.cleanup()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.cleanup()


@dataclass
class WatchdogConfig:
    """Watchdog 配置类，统一管理监控参数"""
    log_file: str  # 推理服务日志文件路径
    success_keywords: List[str]  # 服务启动成��的关键词（任意匹配一个即可）
    npu_card_ids: List[int]  # 要监控的NPU卡ID列表（如[0]、[0,1,2,3]）
    target_pids: Optional[List[int]] = None  # 要监控的推理服务进程PID列表（若未指定，从npu-smi解析）
    target_process_name: Optional[str] = None  # 备选：通过进程名匹配（优先级低于PID）
    timeout: int = 300  # 监控超时时间（秒），默认5分钟
    log_check_interval: float = 1.0  # 日志文件检查间隔（秒）
    npu_check_interval: float = 2.0  # NPU进程检查间隔（秒）

    def __post_init__(self):
        # 兼容单个int传入，自动包装为list
        if isinstance(self.npu_card_ids, int):
            self.npu_card_ids = [self.npu_card_ids]
        if isinstance(self.target_pids, int):
            self.target_pids = [self.target_pids]

class LogMonitorProcess(Process):
    """进程A：监控日志文件中是否出现启动成功关键词"""
    def __init__(
        self,
        config: WatchdogConfig,
        exit_event,
        success_event
    ):
        super().__init__(name="LogMonitor")
        self.config = config
        self.exit_event = exit_event if exit_event else Event()  # 全局退出信号
        self.success_event = success_event if success_event else Event()  # 服务启动成功信号

    def _check_log_(self, new_lines):
        if new_lines:
            for line in new_lines:
                line = line.strip()
                # if line.startswith("E") or "[ERROR]" in line:
                if "[ERROR]" in line:
                    logging.error(f"监控到异常日志：\n{line}")
                    self.exit_event.set()
                    return
                if any(keyword in line for keyword in self.config.success_keywords):
                    logging.info(f"检测到服务启动成功关键词：{line}")
                    self.success_event.set()
                    self.exit_event.set()  # 触发全局退出（仅日志进程完成使命，主进程仍会等NPU监控）
                    return

    def run(self):
        """进程主逻辑"""
        logging.info(f"启动日志监控进程，日志文件：{self.config.log_file}，关键词：{self.config.success_keywords}")
        file_offset = 0  # 记录日志文件读取偏移量，避免重复读取
        # 检查日志文件是否存在
        if not os.path.exists(self.config.log_file):
            logging.error(f"日志文件不存在：{self.config.log_file}")
            self.exit_event.set()
            return

        try:
            with open(self.config.log_file, "r", encoding="utf-8") as f:
                # 先读取已有日志
                new_lines = f.readlines()
                self._check_log_(new_lines)
                # 再将文件指针移到末尾，只监控新增日志
                f.seek(0, os.SEEK_END)
                file_offset = f.tell()

                while not self.exit_event.is_set():
                    # 检查文件是否被截断（如日志轮转）
                    current_file_size = os.path.getsize(self.config.log_file)
                    if current_file_size < file_offset:
                        print("日志文件被截断，重新定位到文件开头")
                        f.seek(0)
                        file_offset = 0

                    # 读取新增内容
                    f.seek(file_offset)
                    new_lines = f.readlines()
                    self._check_log_(new_lines)

                    # 等待检查间隔或退出信号
                    self.exit_event.wait(self.config.log_check_interval)

        except Exception as e:
            logging.error(f"日志监控异常：{str(e)}", exc_info=True)
            self.exit_event.set()

class NpuProcessMonitorProcess(Process):
    """进程B：监控NPU卡上的推理服务进程状态"""
    def __init__(
        self,
        config: WatchdogConfig,
        exit_event
    ):
        super().__init__(name="NpuProcessMonitor")
        self.config = config
        self.exit_event = exit_event if exit_event else Event()  # 全局退出信号

    def _get_npu_processes(self, card_id: int) -> List[dict]:
        """调用 npu-smi 获取指定 NPU 卡上的进程信息。

        实际执行命令：
            npu-smi info -t proc-mem -i <chip_id> -c <core_id>
        其中 chip_id = card_id // 2（对 2 取商），core_id = card_id % 2（对 2 取余）。

        命令回显有两种形态：
            1) 存在进程（可能多行）：
               Process id:1679419 Process name:xllm              Process memory(MB):9229
            2) 无进程：
               No process in device

        执行位置：
            - 固定在 host 上直接执行 npu-smi（本类不做容器适配）。

        Returns:
            进程信息列表，每个元素包含以下键（与调用方兼容）：
                - process_id:        进程号（字符串）
                - process_name:      进程名
                - process_memory_mb: 进程占用显存(MB)，解析失败时为空字符串
            无进程或异常时返回空列表 []。
        """
        try:
            safe_card_id = int(card_id)
            chip_id = safe_card_id // 2   # -i 参数：对 2 取商
            core_id = safe_card_id % 2    # -c 参数：对 2 取余
            cmd = self._build_npu_smi_command(chip_id, core_id)
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                timeout=10
            )
            if result.returncode != 0:
                logging.error(f"执行npu-smi失败（NPU卡{card_id}）：{result.stderr}")
                return []

            processes = []
            for raw_line in result.stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                # 无进程标志：直接返回空列表
                if "No process in device" in line:
                    return []
                # 仅解析包含进程信息的行
                if "Process id" not in line:
                    continue
                proc = self._parse_proc_mem_line(line)
                if proc.get("process_id"):
                    processes.append(proc)
            return processes

        except subprocess.TimeoutExpired:
            logging.error(f"npu-smi命令执行超时（NPU卡{card_id}）")
            return []
        except Exception as e:
            logging.error(f"解析npu-smi输出异常（NPU卡{card_id}）：{str(e)}", exc_info=True)
            return []

    def _build_npu_smi_command(self, chip_id: int, core_id: int) -> str:
        """构造在 host 上执行的 npu-smi 命令字符串。

        chip_id / core_id 已是 int，直接拼接。
        """
        return f"npu-smi info -t proc-mem -i {int(chip_id)} -c {int(core_id)}"

    @staticmethod
    def _parse_proc_mem_line(line: str) -> dict:
        """解析一行 proc-mem 回显，形如：
            Process id:1679419 Process name:xllm              Process memory(MB):9229
        字段之间以多个空格分隔，字段内部以冒号分隔 key/value。
        采用正则提取，兼容进程名含空格前后的多余空白。
        """
        proc = {"process_id": "", "process_name": "", "process_memory_mb": ""}
        pid_match = re.search(r"Process\s+id\s*:\s*(\d+)", line)
        if pid_match:
            proc["process_id"] = pid_match.group(1)
        # 进程名：截取到 "Process memory" 之前，去除首尾空白
        name_match = re.search(r"Process\s+name\s*:\s*(.*?)\s*(?:Process\s+memory|$)", line)
        if name_match:
            proc["process_name"] = name_match.group(1).strip()
        mem_match = re.search(r"Process\s+memory\s*\(MB\)\s*:\s*(\d+)", line)
        if mem_match:
            proc["process_memory_mb"] = mem_match.group(1)
        return proc

    def _is_target_process_alive(self) -> bool:
        """检查目标进程/NPU卡是否存活。
        - PID模式：所有target_pids都需存活，任一退出即返回False；
        - 进程名模式：任一NPU卡找到即存活，所有卡都没找到才返回False；
        - 默认模式：所有NPU卡都需有进程，任一卡为空即返回False。
        """
        # 1. 优先通过PID列表检查：所有PID都需存活
        if self.config.target_pids:
            for pid in self.config.target_pids:
                try:
                    # 向进程发送0信号，检查是否存在（Unix/Linux）
                    os.kill(pid, 0)
                except OSError:
                    logging.info(f"目标进程PID {pid} 已退出")
                    return False
            return True

        # 2. 备选：通过进程名+NPU卡匹配，任一卡找到即存活，所有卡都没找到才返回False
        if self.config.target_process_name:
            for card_id in self.config.npu_card_ids:
                npu_processes = self._get_npu_processes(card_id)
                found = any(
                    proc.get("process_name", "").lower() == self.config.target_process_name.lower()
                    for proc in npu_processes
                )
                if found:
                    return True
            logging.info(f"所有NPU卡{self.config.npu_card_ids}上均未找到进程名 {self.config.target_process_name}")
            return False

        # 3. 未指定PID/进程名，默认检查每张NPU卡是否有进程，任一卡为空即异常
        for card_id in self.config.npu_card_ids:
            npu_processes = self._get_npu_processes(card_id)
            if not npu_processes:
                logging.info(f"NPU卡{card_id}上无运行进程")
                return False
        return True

    def run(self):
        """进程主逻辑"""
        logging.info(f"启动NPU进程监控，NPU卡ID列表：{self.config.npu_card_ids}，目标PID列表：{self.config.target_pids}")
        while not self.exit_event.is_set():
            # 检查进程是否存活
            if not self._is_target_process_alive():
                logging.error("推理服务进程已退出，触发全局退出")
                self.exit_event.set()
                return

            # 等待检查间隔或退出信号
            self.exit_event.wait(self.config.npu_check_interval)

class WatchdogManager:
    """Watchdog 主管理器，协调两个监控进程和超时控制"""
    def __init__(self, config: WatchdogConfig, serving_wait_time: int):
        self.config = config
        self.serving_wait_time = serving_wait_time
        self.exit_event = Event()  # 全局退出信号（所有进程监听）
        self.success_event = Event()  # 服务启动成功信号
        self.log_monitor: Optional[Process] = None
        self.npu_monitor: Optional[Process] = None

    def start(self):
        """启动所有监控进程"""
        # 1. 创建并启动日志监控进程（A）
        self.log_monitor = LogMonitorProcess(
            config=self.config,
            exit_event=self.exit_event,
            success_event=self.success_event
        )
        self.log_monitor.start()

        # 2. 创建并启动NPU进程监控进程（B）
        #self.npu_monitor = NpuProcessMonitorProcess(
        #    config=self.config,
        #    exit_event=self.exit_event
        #)
        #self.npu_monitor.start()

        # 3. 主进程监控超时和退出信号
        logging.info(f"Watchdog启动完成，超时时间：{self.config.timeout}秒")
        elapsed_time = 0
        start_time = time.time()
        while not self.exit_event.is_set():
            # 检查超时
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.config.timeout:
                logging.error(f"监控超时（{self.config.timeout}秒），触发全局退出")
                self.exit_event.set()
                break

            # 每1秒检查一次状态
            time.sleep(1)

        # 4. 等待子进程退出
        self._join_processes()

        # 5. 输出最终状态
        self._print_final_status(elapsed_time + self.serving_wait_time)

    def _join_processes(self):
        """等待所有子进程退出"""
        if self.log_monitor and self.log_monitor.is_alive():
            logging.info("等待日志监控进程退出...")
            self.log_monitor.join(timeout=5)
            if self.log_monitor.is_alive():
                self.log_monitor.terminate()
                logging.warning("强制终止日志监控进程")

        if self.npu_monitor and self.npu_monitor.is_alive():
            logging.info("等待NPU进程监控进程退出...")
            self.npu_monitor.join(timeout=5)
            if self.npu_monitor.is_alive():
                self.npu_monitor.terminate()
                logging.warning("强制终止NPU进程监控进程")

    def _print_final_status(self, elapsed_time: float):
        """输出监控最终状态"""
        logging.info("="*50)
        logging.info(f"监控总时长：{elapsed_time:.2f}秒")
        logging.info(f"服务启动成功：{self.success_event.is_set()}")
        logging.info(f"是否超时：{elapsed_time >= self.config.timeout}")
        logging.info(f"进程异常退出：{not self.success_event.is_set() and not (elapsed_time >= self.config.timeout)}")
        logging.info("="*50)

    def stop(self):
        """手动停止所有监控进程"""
        logging.info("手动触发Watchdog退出")
        self.exit_event.set()
        self._join_processes()


class NpuProcessGuard:
    """轻量级 NPU 进程状态守护器（线程实现，仅监控进程状态，不监控日志）。

    典型用法：在执行一条长时间阻塞的容器命令（如 benchmark）期间，
    后台定时检查 NPU 卡上的进程状态，任一 NPU 进程异常即置位 abort_event，
    由外部命令执行逻辑感知该事件并中断执行、进入错误处理。

    - 复用 npu-smi 检查逻辑，遍历 npu_card_ids 中所有卡；
    - 检查语义：任一卡出现异常（无目标进程/无运行进程）即视为异常，触发中断。
    - 支持通过 command_executor 在指定执行上下文中运行 npu-smi，例如远程节点容器。
    """

    def __init__(
        self,
        npu_card_ids: List[int],
        abort_event: Optional[threading.Event] = None,
        target_process_name: Optional[str] = None,
        check_interval: float = 300.0,
        startup_grace: float = 0.0,
        container_name: Optional[str] = None,
        command_executor: Optional[Callable[[str], Any]] = None,
    ):
        """
        Args:
            npu_card_ids: 要监控的 NPU 卡 ID 列表（支持单个 int 自动包装）
            abort_event: 异常时置位的中断信号（不传则内部创建）
            target_process_name: 可选，按进程名匹配；不传则只检查卡上是否有进程
            check_interval: 检查间隔（秒）
            startup_grace: 启动宽限期（秒），期间不判定异常，避免进程尚未拉起误报
            container_name: 可选，容器名；传入则 npu-smi 通过
                `docker exec -i <container_name> sh -c '<npu-smi ...>'` 在容器内执行，
                不传则在 host 上直接执行。
            command_executor: 可选，外部命令执行器；传入后 npu-smi 命令由该执行器执行，
                返回对象需包含 exit_code / stdout / stderr（如 CommandResult）。
                用于在指定 host 或容器中执行 npu-smi，实现多机节点监控。
        """
        if isinstance(npu_card_ids, int):
            npu_card_ids = [npu_card_ids]
        self.npu_card_ids = npu_card_ids
        self.abort_event = abort_event if abort_event is not None else threading.Event()
        self.target_process_name = target_process_name
        self.check_interval = check_interval
        self.startup_grace = startup_grace
        self.container_name = container_name
        self.command_executor = command_executor
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.aborted_reason: Optional[str] = None

    def _run_command(self, cmd: str):
        """执行 npu-smi 命令：本地 subprocess 或外部 command_executor。"""
        if self.command_executor is not None:
            result = self.command_executor(cmd)
            exit_code = getattr(result, "exit_code", None)
            if exit_code is None:
                exit_code = 0 if getattr(result, "success", True) else 1
            return SimpleNamespace(
                returncode=exit_code,
                stdout=getattr(result, "stdout", "") or "",
                stderr=getattr(result, "stderr", "") or "",
            )
        return subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            timeout=10,
        )

    def _get_npu_processes(self, card_id: int) -> List[dict]:
        """调用 npu-smi 获取指定 NPU 卡上的进程信息。

        实际执行命令：
            npu-smi info -t proc-mem -i <chip_id> -c <core_id>
        其中 chip_id = card_id // 2（对 2 取商），core_id = card_id % 2（对 2 取余）。

        命令回显有两种形态：
            1) 存在进程（可能多行）：
               Process id:1679419 Process name:xllm              Process memory(MB):9229
            2) 无进程：
               No process in device

        执行位置：
            - 无 command_executor 且 container_name 为空时，在 host 上直接执行 npu-smi；
            - 无 command_executor 且 container_name 非空时，通过 docker exec 在容器内执行；
            - 传入 command_executor 时，由调用方决定具体 host 或容器执行位置。

        Returns:
            进程信息列表，每个元素包含以下键（与调用方兼容）：
                - process_id:        进程号（字符串）
                - process_name:      进程名
                - process_memory_mb: 进程占用显存(MB)，解析失败时为空字符串
            无进程或异常时返回空列表 []。
        """
        try:
            safe_card_id = int(card_id)
            chip_id = safe_card_id // 2   # -i 参数：对 2 取商
            core_id = safe_card_id % 2    # -c 参数：对 2 取余
            cmd = self._build_npu_smi_command(chip_id, core_id)
            result = self._run_command(cmd)
            if result.returncode != 0:
                logging.error(f"执行npu-smi失败（NPU卡{card_id}）：{result.stderr}")
                return []

            processes = []
            for raw_line in result.stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                # 无进程标志：直接返回空列表
                if "No process in device" in line:
                    return []
                # 仅解析包含进程信息的行
                if "Process id" not in line:
                    continue
                proc = self._parse_proc_mem_line(line)
                if proc.get("process_id"):
                    processes.append(proc)
            return processes

        except subprocess.TimeoutExpired:
            logging.error(f"npu-smi命令执行超时（NPU卡{card_id}）")
            return []
        except Exception as e:
            logging.error(f"解析npu-smi输出异常（NPU卡{card_id}）：{str(e)}", exc_info=True)
            return []

    def _build_npu_smi_command(self, chip_id: int, core_id: int) -> str:
        """构造最终要执行的 shell 命令字符串。

        - 无 command_executor 且 无 container_name：直接在 host 执行 npu-smi；
        - 无 command_executor 且 有 container_name：包装成 docker exec 在容器内执行；
        - 有 command_executor：由执行器执行，这里只返回裸 npu-smi 命令。

        chip_id / core_id 已是 int，容器名用 shlex.quote 转义，避免注入。
        """
        npu_smi_cmd = (
            f"npu-smi info -t proc-mem -i {int(chip_id)} -c {int(core_id)}"
        )
        if self.command_executor is not None:
            return npu_smi_cmd
        if self.container_name:
            return (
                f"docker exec -i {shlex.quote(str(self.container_name))} "
                f"sh -c {shlex.quote(npu_smi_cmd)}"
            )
        return npu_smi_cmd

    @staticmethod
    def _parse_proc_mem_line(line: str) -> dict:
        """解析一行 proc-mem 回显，形如：
            Process id:1679419 Process name:xllm              Process memory(MB):9229
        字段之间以多个空格分隔，字段内部以冒号分隔 key/value。
        采用正则提取，兼容进程名含空格前后的多余空白。
        """
        proc = {"process_id": "", "process_name": "", "process_memory_mb": ""}
        pid_match = re.search(r"Process\s+id\s*:\s*(\d+)", line)
        if pid_match:
            proc["process_id"] = pid_match.group(1)
        # 进程名：截取到 "Process memory" 之前，去除首尾空白
        name_match = re.search(r"Process\s+name\s*:\s*(.*?)\s*(?:Process\s+memory|$)", line)
        if name_match:
            proc["process_name"] = name_match.group(1).strip()
        mem_match = re.search(r"Process\s+memory\s*\(MB\)\s*:\s*(\d+)", line)
        if mem_match:
            proc["process_memory_mb"] = mem_match.group(1)
        return proc

    def _check_abnormal_card(self) -> Optional[str]:
        """检查是否存在异常的 NPU 卡。任一卡异常即返回异常描述，正常返回 None。"""
        for card_id in self.npu_card_ids:
            npu_processes = self._get_npu_processes(card_id)
            if self.target_process_name:
                found = any(
                    proc.get("process_name", "").lower() == self.target_process_name.lower()
                    for proc in npu_processes
                )
                if not found:
                    return f"NPU卡{card_id}未找到目标进程 {self.target_process_name}"
            else:
                if not npu_processes:
                    return f"NPU卡{card_id}上无运行进程"
        return None

    def _run(self):
        logging.info(f"启动NPU进程守护，NPU卡ID列表：{self.npu_card_ids}，检查间隔：{self.check_interval}s")
        if self.startup_grace > 0:
            self._stop_event.wait(self.startup_grace)
        while not self._stop_event.is_set() and not self.abort_event.is_set():
            reason = self._check_abnormal_card()
            if reason:
                self.aborted_reason = reason
                logging.error(f"NPU进程状态异常，触发中断：{reason}")
                self.abort_event.set()
                return
            self._stop_event.wait(self.check_interval)

    def start(self):
        """启动后台守护线程"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="NpuProcessGuard", daemon=True)
        self._thread.start()

    def stop(self):
        """停止守护线程"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.check_interval + 2)

    def is_aborted(self) -> bool:
        return self.abort_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
