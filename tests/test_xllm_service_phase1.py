"""Phase-1 split Service tests: fake Engine containers, real CPU process owner."""

import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import yaml

from xtask.core import scheduler
from xtask.core.service_group import build_service_plans, get_benchmark_endpoint
from xtask.core.xllm_service_config import (
    build_xllm_service_topology,
    service_container_nodes,
    service_port_claims,
)
from xtask.core.xllm_service_cluster import ManagedServiceProcess, XllmServiceCluster

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/manage_xllm_service_process.py"


def task(explicit=True):
    return {
        "engine_type": "xllm",
        "case_name": "phase1",
        "model_path": "/models/test",
        "served_model_name": "test",
        "benchmark": {"benchmark_type": "aisbench"},
        "deployment": {
            "mode": "pd_disaggregated",
            "master_host": "10.0.0.1",
            "benchmark": (
                {"runner_instance": "p-node", "router_instance": "r0"}
                if explicit
                else {"runner_instance": "p-node"}
            ),
            "xllm_service": {
                "host": "10.0.0.1",
                "etcd_client_port": 3389,
                "etcd_peer_port": 3390,
                "startup_wait_seconds": 0,
                "startup_attempts": 2,
                "service_binary_profile": "refactor",
                **(
                    {
                        "coordinators": [
                            {
                                "name": "c0",
                                "host": "10.0.0.1",
                                "http_server_port": 48888,
                                "rpc_server_port": 38889,
                            }
                        ],
                        "routers": [
                            {
                                "name": "r0",
                                "host": "10.0.0.2",
                                "http_server_port": 38888,
                                "rpc_server_port": 48889,
                            }
                        ],
                    }
                    if explicit
                    else {"http_server_port": 38888, "rpc_server_port": 38889}
                ),
            },
            "pd_instances": [
                {
                    "name": "p",
                    "role": "prefill",
                    "nodes": [
                        {
                            "name": "p-node",
                            "host": "10.0.0.1",
                            "device_ids": [0],
                            "local_port": 48000,
                        }
                    ],
                },
                {
                    "name": "d",
                    "role": "decode",
                    "nodes": [
                        {
                            "name": "d-node",
                            "host": "10.0.0.2",
                            "device_ids": [0],
                            "local_port": 48000,
                        }
                    ],
                },
            ],
        },
    }


def runtimes():
    return [
        SimpleNamespace(
            plan=SimpleNamespace(
                host=host, instance_name=name, is_local=True, is_benchmark_runner=index == 0
            ),
            container_config=SimpleNamespace(container_name=name + "-container"),
            container_manager=mock.Mock(),
        )
        for index, (host, name) in enumerate((("10.0.0.1", "p-node"), ("10.0.0.2", "d-node")))
    ]


def success(text=""):
    return SimpleNamespace(success=True, stdout=text, stderr="")


class TopologyTests(unittest.TestCase):
    def test_flat_auto_keeps_business_port_and_reserves_two_extra_listeners(self):
        value = task(False)
        topo = build_xllm_service_topology(value)
        self.assertEqual(topo.router.http_server_port, 38888)
        self.assertEqual(
            [(p.role, p.http_server_port, p.rpc_server_port) for p in topo.instances],
            [("COORDINATOR", 48888, 38889), ("ROUTER", 38888, 48889)],
        )
        self.assertEqual(get_benchmark_endpoint(value, []), "http://10.0.0.1:38888")

    def test_explicit_remote_router_does_not_change_engine_ranks_or_devices(self):
        value = task()
        with tempfile.TemporaryDirectory() as out:
            plans = build_service_plans(value, out)
        self.assertEqual(len(plans), 2)
        self.assertEqual([p.rank_start for p in plans], [0, 0])
        self.assertEqual([p.device_ids for p in plans], [[0], [0]])
        self.assertEqual(get_benchmark_endpoint(value, plans), "http://10.0.0.2:38888")
        self.assertEqual(len(service_container_nodes(value, plans)), 2)

    def test_counts_and_router_selection_are_not_one_to_one(self):
        value = task()
        control = value["deployment"]["xllm_service"]
        control["coordinators"].append(
            {"name": "c1", "host": "10.0.0.2", "http_server_port": 58888, "rpc_server_port": 58889}
        )
        control["routers"].append(
            {"name": "r1", "host": "10.0.0.1", "http_server_port": 39888, "rpc_server_port": 39889}
        )
        value["deployment"]["benchmark"]["router_instance"] = "r1"
        self.assertEqual(len(build_xllm_service_topology(value).instances), 4)
        self.assertEqual(get_benchmark_endpoint(value, []), "http://10.0.0.1:39888")
        del value["deployment"]["benchmark"]["router_instance"]
        with self.assertRaisesRegex(ValueError, "Multiple Routers"):
            build_xllm_service_topology(value)

    def test_missing_host_and_ambiguous_container_are_rejected(self):
        value = task()
        control = value["deployment"]["xllm_service"]
        control["routers"][0]["host"] = "10.0.0.9"
        with self.assertRaisesRegex(ValueError, "existing Engine container"):
            service_container_nodes(value, runtimes())
        value = task()
        choices = runtimes()
        choices.append(copy.copy(choices[1]))
        choices[-1].plan = SimpleNamespace(host="10.0.0.2", instance_name="d-node-2")
        with self.assertRaisesRegex(ValueError, "container_instance"):
            service_container_nodes(value, choices)
        value["deployment"]["xllm_service"]["routers"][0]["container_instance"] = "d-node-2"
        self.assertIs(service_container_nodes(value, choices)[-1], choices[-1])

    def test_invalid_lists_names_ports_role_or_mixed_configuration_fail(self):
        unsupported = task()
        unsupported["deployment"].pop("mode")
        unsupported["deployment"].pop("pd_instances")
        with self.assertRaisesRegex(ValueError, "require PD or service_routed"):
            build_xllm_service_topology(unsupported)
        for field, replacement in (
            ("routers", []),
            ("routers", "bad"),
            ("coordinators", None),
            ("http_server_port", 1234),
            ("service_deployment_role", "ROUTER"),
        ):
            with self.subTest(field=field):
                value = task()
                value["deployment"]["xllm_service"][field] = replacement
                with self.assertRaises((ValueError, TypeError)):
                    build_xllm_service_topology(value)
        for port in (True, -1, 65536, 1.5, "bad"):
            value = task()
            value["deployment"]["xllm_service"]["routers"][0]["http_server_port"] = port
            with self.assertRaises(ValueError):
                build_xllm_service_topology(value)
        for name in ("../bad", "", "c0"):
            value = task()
            value["deployment"]["xllm_service"]["routers"][0]["name"] = name
            with self.assertRaises(ValueError):
                build_xllm_service_topology(value)

    def test_service_engine_and_etcd_port_collisions_are_rejected(self):
        for port in (48000, 3389, 38889):
            with self.subTest(port=port):
                value = task()
                value["deployment"]["xllm_service"]["routers"][0]["host"] = "10.0.0.1"
                value["deployment"]["xllm_service"]["routers"][0]["http_server_port"] = port
                with tempfile.TemporaryDirectory() as out, self.assertRaisesRegex(
                    ValueError, "[Pp]ort"
                ):
                    build_service_plans(value, out)

    def test_etcd_host_and_namespace_propagate_to_both_engines(self):
        value = task()
        control = value["deployment"]["xllm_service"]
        control.pop("etcd_client_port")
        control.pop("etcd_peer_port")
        control["etcd"] = {"host": "10.0.0.2", "client_port": 2389, "peer_port": 2390}
        control["etcd_namespace"] = "phase1"
        with tempfile.TemporaryDirectory() as out:
            plans = build_service_plans(value, out)
        for plan in plans:
            self.assertEqual(
                (plan.instance_config["pd_master_host"], plan.instance_config["pd_etcd_port"]),
                ("10.0.0.2", 2389),
            )
            self.assertEqual(
                plan.instance_config["serving_extensiable_param"]["etcd_namespace"], "phase1"
            )

    def test_legacy_still_has_only_original_listeners(self):
        value = task(False)
        value["deployment"]["xllm_service"]["service_binary_profile"] = "legacy"
        self.assertEqual(
            {port for _, port, _ in service_port_claims(value)}, {3389, 3390, 38888, 38889}
        )
        self.assertEqual(len(service_container_nodes(value, runtimes())), 1)


class ManagedProcessTests(unittest.TestCase):
    def test_encoded_container_commands_do_not_expand_host_dollars_or_quotes(self):
        runtime = runtimes()[0]
        runtime.container_manager.exec_in_container.return_value = success()
        process = ManagedServiceProcess(runtime, "/tmp/case", "service.log")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "literal.txt"
            script = (
                "import pathlib;pathlib.Path("
                + repr(str(path))
                + ").write_text('$HOME quote\\\"')"
            )
            process.execute("python3 -c " + shlex.quote(script))
            sent = runtime.container_manager.exec_in_container.call_args.args[1]
            subprocess.run('sh -c "' + sent + '"', shell=True, check=True, timeout=10)
            self.assertEqual(path.read_text(), '$HOME quote"')

    def test_cpu_owner_stops_only_recorded_process_and_fences_stale_pid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pid = root / "owned.json"
            log = root / "service.log"
            argv = [sys.executable, str(HELPER)]
            unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
            try:
                start = [
                    *argv,
                    "start",
                    "--pid-file",
                    str(pid),
                    "--log-file",
                    str(log),
                    "--",
                    sys.executable,
                    "-c",
                    "import time;time.sleep(60)",
                ]
                subprocess.run(start, check=True, capture_output=True, timeout=10)
                record = json.loads(pid.read_text())
                self.assertNotEqual(record["pid"], unrelated.pid)
                subprocess.run([*argv, "status", "--pid-file", str(pid)], check=True, timeout=10)
                repeat = subprocess.run(start, capture_output=True, timeout=10)
                self.assertNotEqual(repeat.returncode, 0)
                subprocess.run(
                    [*argv, "stop", "--pid-file", str(pid), "--grace", "0.1"],
                    check=True,
                    timeout=10,
                )
                self.assertIsNone(unrelated.poll())
                spec = importlib.util.spec_from_file_location("owner", HELPER)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                stale = module.identity(unrelated.pid)
                stale["start_time"] = "0"
                pid.write_text(json.dumps(stale))
                subprocess.run([*argv, "stop", "--pid-file", str(pid)], check=True, timeout=10)
                self.assertIsNone(unrelated.poll())
            finally:
                if pid.exists():
                    subprocess.run(
                        [*argv, "stop", "--pid-file", str(pid), "--grace", "0"], timeout=10
                    )
                unrelated.terminate()
                unrelated.wait(timeout=5)


class ClusterTests(unittest.TestCase):
    def make_cluster(self, value=None):
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container", side_effect=lambda p: p
        ):
            return XllmServiceCluster(value or task(), runtimes(), "/tmp/phase1-output")

    def test_startup_order_and_leader_check_use_configured_members(self):
        cluster = self.make_cluster()
        events = []

        def execute(process, command, timeout=30):
            if "etcdctl" in command:
                return success("10.0.0.1:38889\n")
            if "sha256sum" in command:
                return success("same-digest binary\n")
            return success()

        with mock.patch.object(
            ManagedServiceProcess, "execute", autospec=True, side_effect=execute
        ), mock.patch.object(
            ManagedServiceProcess, "healthy", return_value=True
        ), mock.patch.object(
            ManagedServiceProcess,
            "start",
            autospec=True,
            side_effect=lambda p, a, e: events.append(a[0:2]),
        ), mock.patch.object(
            ManagedServiceProcess, "stop"
        ), mock.patch(
            "xtask.core.xllm_service_cluster.time.sleep"
        ):
            cluster.start_with_retry()
        self.assertEqual(events[0][0], "etcd")
        self.assertEqual(events[1][0], "xllm_master_serving")
        self.assertIn(
            "--service_deployment_role=COORDINATOR", cluster._service_argv(cluster.members[0][0])
        )
        self.assertIn(
            "--service_deployment_role=ROUTER", cluster._service_argv(cluster.members[1][0])
        )
        self.assertEqual(cluster.benchmark_endpoint, "http://10.0.0.2:38888")

    def test_service_retry_does_not_restart_etcd_or_another_member(self):
        cluster = self.make_cluster()
        item, process = cluster.members[1]
        with mock.patch.object(process, "start") as start, mock.patch.object(
            process, "stop"
        ) as stop, mock.patch.object(
            process, "healthy", side_effect=[False, True]
        ), mock.patch.object(
            cluster.etcd, "stop"
        ) as etcd_stop, mock.patch(
            "xtask.core.xllm_service_cluster.time.sleep"
        ):
            cluster._start_member(item, process)
        self.assertEqual(start.call_count, 2)
        self.assertEqual(stop.call_count, 2)
        etcd_stop.assert_not_called()

    def test_router_readiness_requires_public_model_route_not_generic_health(self):
        cluster = self.make_cluster()
        cluster.is_split = True

        def execute(process, command, timeout=30):
            if "/readyz" in command:
                return success('{"status":"ready"}')
            if "/v1/models" in command:
                return success('{"data":[{"id":"test"}]}')
            return success()

        with mock.patch.object(cluster, "_anchor_ready", return_value=True), mock.patch.object(
            ManagedServiceProcess, "healthy", return_value=True
        ), mock.patch.object(ManagedServiceProcess, "execute", autospec=True, side_effect=execute):
            self.assertTrue(cluster.wait_for_http_endpoint(1))

        def fail(process, command, timeout=30):
            return success('{"status":"ready"}')  # missing model list despite /readyz success

        with mock.patch.object(cluster, "_anchor_ready", return_value=True), mock.patch.object(
            ManagedServiceProcess, "healthy", return_value=True
        ), mock.patch.object(ManagedServiceProcess, "execute", autospec=True, side_effect=fail):
            self.assertFalse(cluster.wait_for_http_endpoint(0.02))
            self.assertIn("not ready", cluster.last_error)

    def test_legacy_detection_keeps_single_service_without_split_launch(self):
        value = task(False)
        value["deployment"]["xllm_service"]["service_binary_profile"] = "auto"
        cluster = self.make_cluster(value)
        with mock.patch(
            "xtask.core.pd_control_service.XllmControlService._resolve_binary_capabilities",
            return_value=SimpleNamespace(supports_deployment_role=False),
        ), mock.patch(
            "xtask.core.pd_control_service.XllmControlService.start_with_retry"
        ) as legacy, mock.patch.object(
            cluster.etcd, "start"
        ) as etcd:
            cluster.start_with_retry()
        legacy.assert_called_once()
        etcd.assert_not_called()
        self.assertFalse(cluster.is_split)

    def test_shutdown_order_and_errors_do_not_skip_other_owned_members(self):
        cluster = self.make_cluster()
        calls = []
        for item, process in cluster.members:
            process.stop = mock.Mock(side_effect=lambda grace=15, name=item.name: calls.append(name))
        cluster.etcd.stop = mock.Mock(side_effect=lambda grace=15: calls.append("etcd"))
        cluster.stop()
        self.assertEqual(calls, ["r0", "c0", "etcd"])
        cluster.members[-1][1].stop.side_effect = RuntimeError("router-stop")
        with self.assertRaisesRegex(RuntimeError, "router-stop"):
            cluster.stop()
        self.assertEqual(cluster.etcd.stop.call_count, 2)

    def test_stop_etcd_restores_owned_root_for_host_user(self):
        cluster = self.make_cluster()
        cluster.etcd.stop = mock.Mock()
        commands = []

        def execute(command, timeout=30):
            commands.append(command)
            return success()

        with mock.patch.object(cluster.etcd, "execute", side_effect=execute):
            cluster.stop_etcd()
        self.assertEqual(len(commands), 1)
        self.assertIn(repr(cluster.root), commands[0])
        self.assertIn(f"{os.getuid()!r},{os.getgid()!r}", commands[0])

    def test_stop_etcd_still_restores_ownership_after_stop_failure(self):
        cluster = self.make_cluster()
        cluster.etcd.stop = mock.Mock(side_effect=RuntimeError("etcd-stop"))
        execute = mock.Mock(return_value=success())
        with mock.patch.object(cluster.etcd, "execute", autospec=True, side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, "etcd-stop"):
                cluster.stop_etcd()
        self.assertEqual(execute.call_count, 1)

    def test_stop_etcd_ownership_failure_is_warning_not_cleanup_failure(self):
        cluster = self.make_cluster()
        cluster.etcd.stop = mock.Mock()
        execute = mock.Mock(
            return_value=SimpleNamespace(success=False, stdout="", stderr="permission denied")
        )
        with mock.patch.object(cluster.etcd, "execute", side_effect=execute), self.assertLogs(
            level="WARNING"
        ) as logs:
            cluster.stop_etcd()
        self.assertIn("Could not restore xLLM Service artifact ownership", "\n".join(logs.output))

    def test_logs_are_per_trial_and_do_not_overwrite_previous_results(self):
        with tempfile.TemporaryDirectory() as root:
            cluster = self.make_cluster()
            cluster.local_root = str(Path(root) / "live")
            source = Path(cluster.local_root) / "r0" / "xservice.log"
            source.parent.mkdir(parents=True)
            source.write_text("first")
            cluster.archive_logs(root, "trial1")
            source.write_text("second")
            cluster.archive_logs(root, "trial1")
            cluster.archive_logs(root, "trial2")
            self.assertEqual(
                (Path(root) / "trial1/xllm_service/r0/xservice.log").read_text(), "first"
            )
            self.assertEqual(
                (Path(root) / "trial2/xllm_service/r0/xservice.log").read_text(), "second"
            )


class CpuClusterSmokeTests(unittest.TestCase):
    """Real subprocess/HTTP/PID smoke with fake binaries, no Docker or NPU."""

    def test_split_start_readiness_stop_in_reused_container_shell(self):
        import socket

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            master = root / "master"
            # Reserve unique localhost ports while assembling the case.
            sockets = []
            ports = []
            for _ in range(6):
                sock = socket.socket()
                sock.bind(("127.0.0.1", 0))
                sockets.append(sock)
                ports.append(sock.getsockname()[1])
            executable = f"#!{sys.executable}\n"
            common = """import json,os,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
args={}
for i,a in enumerate(sys.argv[1:]):
 if a.startswith('--') and '=' in a:args.update([a[2:].split('=',1)])
 elif a.startswith('--') and i+2<len(sys.argv):args[a[2:]]=sys.argv[i+2]
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  body={'status':'ready'} if self.path=='/readyz' else {'data':[{'id':'test'}]} if self.path=='/v1/models' else {'health':'true'}
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(body).encode())
 def log_message(self,*a):pass
"""
            service = (
                executable
                + common
                + """
if '--help' in sys.argv:
 print('service_deployment_role readyz');sys.exit(0)
if args.get('service_deployment_role')=='COORDINATOR':
 open(os.environ['FAKE_MASTER'],'w').write(args['server_host']+':'+args['rpc_server_port'])
HTTPServer((args['server_host'],int(args['http_server_port'])),Handler).serve_forever()
"""
            )
            etcd = (
                executable
                + common
                + """
from urllib.parse import urlparse
url=urlparse(args['listen-client-urls'])
HTTPServer(('127.0.0.1',url.port),Handler).serve_forever()
"""
            )
            ctl = executable + "import os;print(open(os.environ['FAKE_MASTER']).read())\n"
            for name, text in (("xllm_master_serving", service), ("etcd", etcd), ("etcdctl", ctl)):
                path = bin_dir / name
                path.write_text(text)
                path.chmod(0o755)
            value = task()
            control = value["deployment"]["xllm_service"]
            control.update(
                host="127.0.0.1",
                etcd_client_port=ports[0],
                etcd_peer_port=ports[1],
                master_key_wait_seconds=5,
                startup_wait_seconds=0,
            )
            control["coordinators"] = [
                {
                    "name": "c0",
                    "host": "127.0.0.1",
                    "http_server_port": ports[2],
                    "rpc_server_port": ports[3],
                    "container_instance": "p-node",
                }
            ]
            control["routers"] = [
                {
                    "name": "r0",
                    "host": "127.0.0.1",
                    "http_server_port": ports[4],
                    "rpc_server_port": ports[5],
                    "container_instance": "p-node",
                }
            ]
            control["container_instance"] = "p-node"
            env = dict(
                os.environ, PATH=str(bin_dir) + ":" + os.environ["PATH"], FAKE_MASTER=str(master)
            )

            class FakeContainerManager:
                def exec_in_container(self, name, command, timeout=30):
                    ret = subprocess.run(
                        'sh -c "' + command + '"',
                        shell=True,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    return SimpleNamespace(
                        success=ret.returncode == 0, stdout=ret.stdout, stderr=ret.stderr
                    )

            runtime = SimpleNamespace(
                plan=SimpleNamespace(
                    host="127.0.0.1",
                    instance_name="p-node",
                    is_local=True,
                    is_benchmark_runner=True,
                ),
                container_config=SimpleNamespace(container_name="reused"),
                container_manager=FakeContainerManager(),
            )
            for sock in sockets:
                sock.close()
            with mock.patch(
                "xtask.utils.container_manager.path_mapping_in_container", side_effect=lambda p: p
            ):
                cluster = XllmServiceCluster(value, [runtime], str(root / "out"))
            original_helper = ManagedServiceProcess._helper

            def local_helper(process, action, arguments=()):
                return original_helper(process, action, arguments).replace(
                    "/workspace/tools/manage_xllm_service_process.py", str(HELPER)
                )

            with mock.patch.object(ManagedServiceProcess, "_helper", new=local_helper):
                try:
                    cluster.start_with_retry()
                    self.assertTrue(cluster.wait_for_http_endpoint(timeout=5), cluster.last_error)
                    self.assertEqual(cluster.benchmark_endpoint, f"http://127.0.0.1:{ports[4]}")
                    self.assertEqual(len(list(Path(cluster.local_root).glob("*/process.json"))), 3)
                finally:
                    cluster.stop()
            self.assertFalse(list(Path(cluster.local_root).glob("*/process.json")))


class SchedulerIntegrationTests(unittest.TestCase):
    def test_installs_service_only_in_etcd_and_configured_role_containers(self):
        value = task()
        nodes = runtimes()
        nodes.append(
            SimpleNamespace(
                plan=SimpleNamespace(
                    host="10.0.0.3",
                    instance_name="other-node",
                    is_local=True,
                    is_benchmark_runner=False,
                ),
                container_config=SimpleNamespace(container_name="other-container"),
                container_manager=mock.Mock(),
            )
        )
        for node in nodes:
            node.container_manager.exec_in_container.return_value = success()
        with tempfile.TemporaryDirectory() as out:
            model = Path(out) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            value["model_path"] = str(model)
            context = SimpleNamespace(
                state={
                    "task_config": value,
                    "engine_config": {},
                    "service_runtimes": nodes,
                    "service_plans": [],
                },
                output_file_path=out,
                result={},
                complete=False,
            )
            with mock.patch(
                "xtask.core.ranktable_manager.prepare_multihost_ranktables", return_value={}
            ), mock.patch.object(scheduler, "_clear_pre_benchmark_rst"):
                scheduler._prepare_runtime_dependencies_multi_host(context)
        for node, enabled in zip(nodes, (True, True, False)):
            commands = [c.args[1] for c in node.container_manager.exec_in_container.call_args_list]
            installs = [c for c in commands if "download_xllm.sh" in c]
            self.assertEqual(len(installs), 1)
            self.assertIn(f"XLLM_INSTALL_SERVICE={int(enabled)}", installs[0])

    def test_scheduler_stores_cluster_before_partial_start_failure(self):
        value = task()
        nodes = runtimes()
        cluster = mock.Mock()
        cluster.start_with_retry.side_effect = RuntimeError("partial-start")
        context = SimpleNamespace(
            state={"task_config": value, "service_runtimes": nodes, "pd_disaggregated": True},
            output_file_path="/tmp/phase1",
            result={},
            complete=False,
        )
        with mock.patch.object(scheduler, "_reserve_xllm_ports_before_start"), mock.patch(
            "xtask.core.xllm_service_cluster.XllmServiceCluster", return_value=cluster
        ):
            scheduler._start_and_wait_service_multi_host(context)
        self.assertIs(context.state["xllm_control_service"], cluster)
        self.assertTrue(context.complete)
        self.assertIn("partial-start", context.result["message"])

    def test_reused_session_checks_router_models_not_health_substring(self):
        cluster = SimpleNamespace(
            is_split=True, wait_for_http_endpoint=mock.Mock(return_value=True), last_error=""
        )
        context = SimpleNamespace(
            state={
                "task_config": task(),
                "multi_host": True,
                "container_manager": mock.Mock(),
                "container_config": SimpleNamespace(container_name="p"),
                "xllm_control_service": cluster,
            },
            result={},
        )
        self.assertTrue(scheduler._check_session_service_health(context))
        cluster.wait_for_http_endpoint.assert_called_once_with(timeout=90)
        context.state["container_manager"].exec_in_container.assert_not_called()
        cluster.wait_for_http_endpoint.return_value = False
        cluster.last_error = "model not ready"
        self.assertFalse(scheduler._check_session_service_health(context))
        self.assertIn("model not ready", context.result["message"])

    def test_trial_archive_calls_split_service_snapshot_before_cleanup(self):
        cluster = SimpleNamespace(
            is_split=True,
            local_root="/tmp/case/xllm_service/id",
            collect_logs=mock.Mock(),
            archive_logs=mock.Mock(),
        )
        state = {"service_runtimes": [object()], "xllm_control_service": cluster}
        scheduler._record_service_group_archive_path(state, "trial1")
        self.assertEqual(state["service_group_archive_paths"], ["trial1"])
        cluster.collect_logs.assert_called_once()
        cluster.archive_logs.assert_called_once_with("/tmp/case", "trial1")


class SafetyRegressionTests(unittest.TestCase):
    def test_unsafe_or_wildcard_host_is_rejected_before_shell_generation(self):
        for host in ("0.0.0.0", "10.0.0.1;false", "$(false)", "bad host", "::1"):
            value = task()
            value["deployment"]["xllm_service"]["routers"][0]["host"] = host
            with self.subTest(host=host), self.assertRaises(ValueError):
                build_xllm_service_topology(value)

    def test_old_failure_log_does_not_poison_owned_process_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = root / "owner.json"
            log = root / "xservice.log"
            log.write_text("FATAL from a failed attempt")
            argv = [sys.executable, str(HELPER)]
            try:
                subprocess.run(
                    [
                        *argv,
                        "start",
                        "--pid-file",
                        str(record),
                        "--log-file",
                        str(log),
                        "--",
                        sys.executable,
                        "-c",
                        "import time;time.sleep(30)",
                    ],
                    check=True,
                    timeout=10,
                )
                self.assertNotIn("FATAL", log.read_text())
                history = list(root.glob("xservice.*.log"))
                self.assertEqual(len(history), 1)
                self.assertIn("FATAL", history[0].read_text())
            finally:
                subprocess.run(
                    [*argv, "stop", "--pid-file", str(record), "--grace", "0"], timeout=10
                )

    def test_preflight_rejects_mismatched_binaries_before_any_start(self):
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container", side_effect=lambda p: p
        ):
            cluster = XllmServiceCluster(task(), runtimes(), "/tmp/phase1")

        def execute(process, command, timeout=30):
            if "sha256sum" in command:
                return success(process.runtime.plan.instance_name + " binary")
            return success()

        with mock.patch.object(
            ManagedServiceProcess, "execute", autospec=True, side_effect=execute
        ), mock.patch.object(cluster.etcd, "start") as start:
            with self.assertRaisesRegex(RuntimeError, "different Service binaries"):
                cluster.start_with_retry()
        start.assert_not_called()

    def test_failed_cleanup_still_attempts_every_member_and_shared_etcd(self):
        with mock.patch(
            "xtask.utils.container_manager.path_mapping_in_container", side_effect=lambda p: p
        ):
            cluster = XllmServiceCluster(task(), runtimes(), "/tmp/phase1")
        first = cluster.members[-1][1]
        first.stop = mock.Mock(side_effect=RuntimeError("bad-stop"))
        cluster.members[0][1].stop = mock.Mock()
        cluster.etcd.stop = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "bad-stop"):
            cluster.stop()
        cluster.members[0][1].stop.assert_called_once()
        cluster.etcd.stop.assert_called_once()
