# framework/container/container_watchdog.py
import os
import sys
import time
import signal
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
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
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            if self.config.log_file:
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            
            self.logger.setLevel(logging.INFO)
    
    def log(self, message: str, level: str = "info", exception=None):
        """Log message with specified level"""
        log_func = getattr(self.logger, level, self.logger.info)
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


# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ContainerManagerWithWatchdog")
    
    print("=" * 60)
    print("Example: Inference Service with Watchdog")
    print("=" * 60)
    
    # Create container configuration
    from .container_manager import ContainerConfig
    
    config = ContainerConfig(
        image_name="tensorrt-inference:latest",
        container_name="inference-service-test",
        runtime="docker",
        ports={"8000": "8000"},
        volumes={"./models": "/models", "./logs": "/logs"},
        environment={
            "MODEL_PATH": "/models/gpt2",
            "PORT": "8000"
        }
    )
    
    # Create container manager with watchdog
    with ContainerManagerWithWatchdog(config, logger) as manager:
        # Check if runtime is available
        if not manager.check_runtime_available():
            print("Container runtime not available")
            sys.exit(1)
        
        # Setup container
        print("Setting up container...")
        success, container_id = manager.setup_container()
        
        if not success:
            print(f"Failed to setup container: {container_id}")
            sys.exit(1)
        
        # Wait for container to be ready
        print("Waiting for container to be ready...")
        if not manager.wait_for_container(container_id, timeout=60):
            print("Container not ready within timeout")
            sys.exit(1)
        
        # Start inference service with watchdog
        print("Starting inference service with watchdog...")
        result = manager.start_inference_service(
            container_name=container_id,
            model_path="/models/gpt2",
            port=8000,
            framework="tensorrt"
        )
        
        if result.success:
            print(f"Inference service started successfully: {result.stdout}")
            
            # Get monitoring status
            status = manager.get_monitoring_status()
            print(f"Watchdog status: {status['watchdog_running']}")
            print(f"Monitored processes: {status['monitored_processes']}")
            
            # Simulate running for a while
            print("Service running with watchdog monitoring...")
            print("Press Ctrl+C to stop")
            
            try:
                # Keep the main thread alive
                while True:
                    time.sleep(1)
                    
                    # Periodically check status
                    status = manager.get_monitoring_status()
                    for process in status['processes']:
                        print(f"Process {process['cmd']}: "
                              f"PID={process['pid']}, "
                              f"Status={process['status']}, "
                              f"CPU={process['cpu_percent']:.1f}%, "
                              f"Memory={process['memory_mb']:.1f}MB")
                    
                    time.sleep(10)
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                
        else:
            print(f"Failed to start inference service: {result.stderr}")
        
        print("Cleaning up...")
