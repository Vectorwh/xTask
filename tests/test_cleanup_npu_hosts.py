"""Cleanup ordering and sudo fallback, without touching real workloads.

Host operations are mocked. Shell-flow tests replace /bin/kill and sudo with
local recording stubs before execution; no real signals or sudo calls occur.
"""

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools/cleanup_npu_hosts.py"
SPEC = importlib.util.spec_from_file_location("cleanup_npu_hosts_under_test", SCRIPT)
cleanup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cleanup
SPEC.loader.exec_module(cleanup)


def result(success=True, stdout="", stderr=""):
    return SimpleNamespace(
        success=success, exit_code=0 if success else 1, stdout=stdout, stderr=stderr
    )


def process(pid):
    return cleanup.NpuProcess(logic_id=0, npu_id=0, chip_id=0, pid=pid, name="fixture")


class CleanupTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(cleanup, "print", create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def patch(self, name, **kwargs):
        patcher = mock.patch.object(cleanup, name, **kwargs)
        patched = patcher.start()
        self.addCleanup(patcher.stop)
        return patched


class CleanupOrderingTests(CleanupTestCase):
    def setUp(self):
        super().setUp()
        self.executor = mock.Mock()
        self.executor.run.side_effect = AssertionError("No real host commands allowed")
        self.patch("load_ssh_credentials", return_value={})
        self.patch("create_host_executor", return_value=self.executor)
        self.before = cleanup.HostReport(
            host="fixture", containers=["container"], processes=[process(101)]
        )
        self.after = cleanup.HostReport(host="fixture")
        self.patch("collect_host_report", side_effect=[self.before, self.after])
        self.patch("print_report")
        self.patch("discover_devices", return_value={0: (0, 0)})

    def run_host(self, **kwargs):
        options = dict(query_only=False, dry_run=False, kill_processes=True, stop_containers=True)
        options.update(kwargs)
        return cleanup.process_host("fixture", **options)

    def test_stop_then_requery_then_signal_only_fresh_pids(self):
        events = []
        stop = self.patch(
            "stop_running_containers",
            side_effect=lambda *args, **kwargs: (events.append("stop") or (["container"], [])),
        )
        self.patch(
            "query_npu_processes",
            side_effect=lambda *args: (events.append("query") or [process(202)]),
        )
        kill = self.patch(
            "kill_npu_processes",
            side_effect=lambda *args, **kwargs: (events.append("kill") or ([202], [])),
        )
        report = self.run_host()
        self.assertEqual(events, ["stop", "query", "kill"])
        stop.assert_called_once()
        self.assertEqual([p.pid for p in kill.call_args.args[2]], [202])
        self.assertEqual(report.cleanup_failures, [])

    def test_container_shutdown_releases_all_processes(self):
        self.patch("stop_running_containers", return_value=(["container"], []))
        self.patch("query_npu_processes", return_value=[])
        kill = self.patch("kill_npu_processes")
        self.run_host()
        kill.assert_not_called()

    def test_requery_even_when_initial_process_list_was_empty(self):
        self.before.processes = []
        self.patch("stop_running_containers", return_value=(["container"], []))
        self.patch("query_npu_processes", return_value=[process(303)])
        kill = self.patch("kill_npu_processes", return_value=([303], []))
        self.run_host()
        self.assertEqual([p.pid for p in kill.call_args.args[2]], [303])

    def test_stop_exception_does_not_skip_residual_process_cleanup(self):
        self.patch("stop_running_containers", side_effect=RuntimeError("Docker unavailable"))
        self.patch("query_npu_processes", return_value=[process(202)])
        kill = self.patch("kill_npu_processes", return_value=([202], []))
        report = self.run_host()
        kill.assert_called_once()
        self.assertIn("Docker unavailable", report.cleanup_failures[0])

    def test_no_stop_option_still_requeries_processes(self):
        stop = self.patch("stop_running_containers")
        query = self.patch("query_npu_processes", return_value=[process(202)])
        self.patch("kill_npu_processes", return_value=([202], []))
        self.run_host(stop_containers=False)
        stop.assert_not_called()
        query.assert_called_once()

    def test_no_kill_option_only_stops_containers(self):
        stop = self.patch("stop_running_containers", return_value=(["container"], []))
        query = self.patch("query_npu_processes")
        kill = self.patch("kill_npu_processes")
        self.run_host(kill_processes=False)
        stop.assert_called_once()
        query.assert_not_called()
        kill.assert_not_called()

    def test_dry_run_does_not_execute_stop_kill_or_sudo(self):
        query = self.patch("query_npu_processes")
        self.run_host(dry_run=True)
        self.executor.run.assert_not_called()
        query.assert_not_called()

    def test_query_only_does_not_execute_actions(self):
        stop = self.patch("stop_running_containers")
        kill = self.patch("kill_npu_processes")
        self.assertIs(self.run_host(query_only=True), self.before)
        stop.assert_not_called()
        kill.assert_not_called()
        self.executor.run.assert_not_called()


class SignalShellTests(CleanupTestCase):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="cleanup-signal-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.kill_log = self.base / "kill.log"
        self.sudo_log = self.base / "sudo.log"
        self.kill_stub = self.base / "kill-stub"
        self.kill_stub.write_text('''#!/bin/sh
printf '%s %s\\n' "${IN_SUDO:-0}" "$*" >> "$KILL_LOG"
for target do :; done
if [ "$target" = "${MISSING_PID:-}" ] ||
   { [ "${IN_SUDO:-0}" = 1 ] && [ "$target" = "${SUDO_MISSING_PID:-}" ]; }; then
    echo "${MISSING_PREFIX:-kill}: sending signal to $target failed: No such process" >&2
    exit 1
fi
if [ "${IN_SUDO:-0}" = 1 ]; then exit "${SUDO_KILL_STATUS:-0}"; fi
if [ "${NORMAL_KILL_STATUS:-0}" != 0 ]; then
    echo 'kill: Operation not permitted' >&2
fi
exit "${NORMAL_KILL_STATUS:-0}"
''')
        sudo = self.base / "sudo"
        sudo.write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$SUDO_LOG"
if [ "${SUDO_DENIED:-0}" = 1 ]; then
    echo "${SUDO_ERROR:-sudo: a password is required}" >&2
    exit 1
fi
[ "$1" = -n ] || exit 99
shift
[ "$1" = -u ] && [ "$2" = root ] || exit 99
shift 2
[ "$1" = -- ] || exit 99
shift
IN_SUDO=1
export IN_SUDO
exec "$@"
''')
        self.kill_stub.chmod(0o755)
        sudo.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            PATH=f"{self.base}:{self.env['PATH']}",
            KILL_LOG=str(self.kill_log), SUDO_LOG=str(self.sudo_log),
            NORMAL_KILL_STATUS="0", SUDO_KILL_STATUS="0", SUDO_DENIED="0", IN_SUDO="0",
            MISSING_PID="", SUDO_MISSING_PID="", MISSING_PREFIX="kill",
            SUDO_ERROR="sudo: a password is required",
        )
        self.executor = mock.Mock()
        self.executor.run.side_effect = self._execute_stubs

    def _execute_stubs(self, command, **kwargs):
        # Fail closed if the implementation adds another destructive command.
        self.assertNotIn("pkill", command)
        self.assertNotIn("/usr/bin/kill", command)
        self.assertNotIn("/usr/bin/sudo", command)
        self.assertNotIn("/bin/sudo", command)
        command = command.replace("/bin/kill", shlex.quote(str(self.kill_stub)))
        self.assertNotIn("/bin/kill", command)
        completed = subprocess.run(
            ["/bin/sh", "-c", command], env=self.env,
            capture_output=True, text=True, timeout=10,
        )
        return result(completed.returncode == 0, completed.stdout, completed.stderr)

    def test_normal_permission_does_not_use_sudo(self):
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertTrue(response.success)
        self.assertFalse(self.sudo_log.exists())
        self.assertIn(f"0 -TERM -- {os.getpid()}", self.kill_log.read_text())

    def test_permission_failure_retries_exact_pid_with_noninteractive_sudo(self):
        self.env["NORMAL_KILL_STATUS"] = "1"
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertTrue(response.success, response.stderr)
        self.assertIn("Operation not permitted", response.stderr)
        self.assertIn("[SUDO]", response.stdout)
        self.assertIn("-n -u root --", self.sudo_log.read_text())
        self.assertIn(f"1 -TERM -- {os.getpid()}", self.kill_log.read_text())

    def test_sudo_without_noninteractive_authorization_returns_failure(self):
        self.env.update(NORMAL_KILL_STATUS="1", SUDO_DENIED="1")
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertFalse(response.success)
        self.assertIn("a password is required", response.stderr)
        self.assertIn("[FAILED]", response.stderr)
        self.assertEqual(len(self.kill_log.read_text().splitlines()), 1)

    def test_kill_phase_can_use_sudo_too(self):
        self.env["NORMAL_KILL_STATUS"] = "1"
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "KILL")
        self.assertTrue(response.success, response.stderr)
        self.assertIn(f"1 -KILL -- {os.getpid()}", self.kill_log.read_text())

    def test_already_exited_pid_does_not_request_sudo(self):
        pid = 999999999
        self.assertFalse(Path(f"/proc/{pid}").exists())
        self.env.update(NORMAL_KILL_STATUS="1", MISSING_PID=str(pid))
        response = cleanup._signal_npu_pids(self.executor, [pid], "TERM")
        self.assertTrue(response.success)
        self.assertFalse(self.sudo_log.exists())

    def test_mixed_batch_requests_sudo_only_for_still_existing_pid(self):
        missing = 999999999
        self.assertFalse(Path(f"/proc/{missing}").exists())
        self.env.update(NORMAL_KILL_STATUS="1", MISSING_PID=str(missing))
        response = cleanup._signal_npu_pids(self.executor, [os.getpid(), missing], "TERM")
        self.assertTrue(response.success, response.stderr)
        sudo_calls = self.sudo_log.read_text().splitlines()
        self.assertEqual(len(sudo_calls), 1)
        self.assertNotIn(str(missing), sudo_calls[0])

    def test_authorized_sudo_with_failed_kill_is_reported_as_failure(self):
        self.env.update(NORMAL_KILL_STATUS="1", SUDO_KILL_STATUS="1")
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertFalse(response.success)
        self.assertIn("[FAILED]", response.stderr)
        self.assertIn(f"1 -TERM -- {os.getpid()}", self.kill_log.read_text())

    def _hide_proc_from_signal_command(self):
        hidden_proc = self.base / "hidden-proc"
        hidden_proc.mkdir()

        def execute(command, **kwargs):
            command = command.replace("/proc/", str(hidden_proc) + "/")
            return self._execute_stubs(command, **kwargs)

        self.executor.run.side_effect = execute

    def test_hidden_proc_does_not_skip_authorized_sudo(self):
        self._hide_proc_from_signal_command()
        self.env["NORMAL_KILL_STATUS"] = "1"
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertTrue(response.success, response.stderr)
        self.assertTrue(self.sudo_log.exists())
        self.assertIn(f"1 -TERM -- {os.getpid()}", self.kill_log.read_text())

    def test_hidden_proc_does_not_hide_sudo_denial(self):
        self._hide_proc_from_signal_command()
        self.env.update(NORMAL_KILL_STATUS="1", SUDO_DENIED="1")
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertFalse(response.success)
        self.assertIn("a password is required", response.stderr)
        self.assertIn("[FAILED]", response.stderr)

    def test_process_exiting_before_sudo_is_not_a_cleanup_failure(self):
        self.env.update(NORMAL_KILL_STATUS="1", SUDO_MISSING_PID=str(os.getpid()))
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertTrue(response.success, response.stderr)
        self.assertTrue(self.sudo_log.exists())
        self.assertIn("No such process", response.stderr)

    def test_sudo_error_containing_missing_process_text_is_not_ignored(self):
        self.env.update(
            NORMAL_KILL_STATUS="1", SUDO_DENIED="1",
            SUDO_ERROR="sudo: unable to launch /bin/kill: No such process",
        )
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertFalse(response.success)
        self.assertIn("[FAILED]", response.stderr)

    def test_missing_process_diagnostic_with_absolute_kill_path(self):
        self.env.update(MISSING_PID=str(os.getpid()), MISSING_PREFIX="/bin/kill")
        response = cleanup._signal_npu_pids(self.executor, [os.getpid()], "TERM")
        self.assertTrue(response.success, response.stderr)
        self.assertFalse(self.sudo_log.exists())

    def test_invalid_or_protected_targets_never_execute_commands(self):
        for pids in [[], [0], [1], [-1], [2, 0], ["123"]]:
            with self.subTest(pids=pids):
                with self.assertRaises(ValueError):
                    cleanup._signal_npu_pids(self.executor, pids, "TERM")
        with self.assertRaises(ValueError):
            cleanup._signal_npu_pids(self.executor, [os.getpid()], "INVALID")
        self.executor.run.assert_not_called()


class ProcessSignalPhasesTests(CleanupTestCase):
    def setUp(self):
        super().setUp()
        self.executor = mock.Mock()
        self.signal = self.patch("_signal_npu_pids", return_value=result())
        self.query = self.patch("query_npu_processes", return_value=[])
        self.patch("discover_devices", return_value={0: (0, 0)})
        sleep = mock.patch.object(cleanup.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def test_term_only_when_no_processes_remain(self):
        targeted, failures = cleanup.kill_npu_processes(
            self.executor, "fixture", [process(202), process(101), process(101)], False
        )
        self.assertEqual(targeted, [101, 202])
        self.assertEqual(failures, [])
        self.signal.assert_called_once_with(self.executor, [101, 202], "TERM")

    def test_kill_only_the_requeried_remaining_pids(self):
        self.query.side_effect = [[process(202)], []]
        _, failures = cleanup.kill_npu_processes(
            self.executor, "fixture", [process(101), process(202)], False
        )
        self.assertEqual(failures, [])
        self.assertEqual(self.signal.call_args_list, [
            mock.call(self.executor, [101, 202], "TERM"),
            mock.call(self.executor, [202], "KILL"),
        ])
        self.assertEqual(self.sleep.call_count, 2)

    def test_persistent_processes_are_reported(self):
        self.query.return_value = [process(202)]
        _, failures = cleanup.kill_npu_processes(
            self.executor, "fixture", [process(202)], False
        )
        self.assertTrue(any("still alive" in error for error in failures))

    def test_signal_failure_preserves_sudo_diagnostic(self):
        self.signal.return_value = result(False, stderr="sudo: a password is required")
        _, failures = cleanup.kill_npu_processes(
            self.executor, "fixture", [process(202)], False
        )
        self.assertIn("a password is required", failures[0])

    def test_dry_run_and_protected_pids_do_not_send_signals(self):
        cleanup.kill_npu_processes(self.executor, "fixture", [process(202)], True)
        targets, failures = cleanup.kill_npu_processes(
            self.executor, "fixture", [process(0), process(1)], False
        )
        self.assertEqual(targets, [])
        self.assertEqual(len(failures), 2)
        self.signal.assert_not_called()
        self.query.assert_not_called()
        self.sleep.assert_not_called()


class ContainerStopTests(CleanupTestCase):
    def test_stop_success(self):
        executor = mock.Mock()
        executor.run.return_value = result()
        stopped, failures = cleanup.stop_running_containers(executor, "fixture", ["demo"], False)
        self.assertEqual((stopped, failures), (["demo"], []))
        self.assertIn("docker stop", executor.run.call_args.args[0])

    def test_stop_failure_uses_docker_kill_fallback(self):
        executor = mock.Mock()
        executor.run.side_effect = [result(False, stderr="timeout"), result()]
        stopped, failures = cleanup.stop_running_containers(executor, "fixture", ["demo"], False)
        self.assertEqual((stopped, failures), (["demo"], []))
        self.assertIn("docker kill", executor.run.call_args.args[0])

    def test_both_docker_commands_fail(self):
        executor = mock.Mock()
        executor.run.return_value = result(False, stderr="denied")
        stopped, failures = cleanup.stop_running_containers(executor, "fixture", ["demo"], False)
        self.assertEqual(stopped, [])
        self.assertIn("denied", failures[0])

    def test_dry_run_does_not_call_docker(self):
        executor = mock.Mock()
        self.assertEqual(cleanup.stop_running_containers(executor, "fixture", ["demo"], True), (["demo"], []))
        executor.run.assert_not_called()


class ExistingExitPolicyTests(CleanupTestCase):
    def test_cleanup_failure_still_uses_existing_continue_policy(self):
        self.patch("parse_args", return_value=SimpleNamespace(
            hosts=["fixture"], query_only=False, dry_run=False,
            no_kill_processes=False, no_stop_containers=False,
        ))
        self.patch("load_ssh_credentials", return_value={"fixture": {}})
        self.patch("process_host", return_value=cleanup.HostReport(
            host="fixture", cleanup_failures=["fixture failure"]
        ))
        self.assertEqual(cleanup.main(), 0)


if __name__ == "__main__":
    unittest.main()
