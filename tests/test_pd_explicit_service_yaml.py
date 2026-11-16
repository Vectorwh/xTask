"""Keep authored PD cases aligned with the split Service deployment example."""

import copy
import itertools
from pathlib import Path
import unittest

import yaml

from xtask.core.scheduler import TaskConfig
from xtask.core.service_group import build_service_plans, get_benchmark_endpoint
from xtask.core.xllm_service_config import (
    build_xllm_service_topology,
    service_container_nodes,
    service_etcd_location,
    service_port_claims,
)
from xtask.tasks.parameter_search import build_parameter_candidate
from xtask.tasks.task_parser import _normalize_assess_parameters
from xtask.tasks.topology_search import normalize_assess_topologies


ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = ROOT / "tests/e2e"
REFERENCE = (
    E2E_ROOT
    / "xllm/examples/yaml_authoring"
    / "34_glm5.3_pd_split_service_reuse_containers_perf.yaml"
)


class PdExplicitServiceYamlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
        cls.cases = []
        for path in sorted(E2E_ROOT.rglob("*")):
            if path.suffix not in {".yaml", ".yml"} or not path.is_file():
                continue
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                continue
            deployment = config.get("deployment") or {}
            if deployment.get("mode") == "pd_disaggregated" or deployment.get(
                "pd_instances"
            ):
                cls.cases.append((path, config))

    def _assert_runtime_bindings(self, authored_config):
        config = copy.deepcopy(authored_config)
        plans = build_service_plans(config, "/tmp/pd_yaml_validation")
        topology = build_xllm_service_topology(config)
        owners = service_container_nodes(config, plans)
        self.assertTrue(topology.explicit)
        nodes = {plan.instance_name: plan for plan in plans}
        self.assertEqual(
            len(plans),
            sum(
                len(instance["nodes"])
                for instance in authored_config["deployment"]["pd_instances"]
            ),
        )
        for item in topology.instances:
            self.assertEqual(nodes[item.container_instance].host, item.host)
            self.assertIn(nodes[item.container_instance], owners)
        self.assertEqual(
            nodes[topology.etcd_container_instance].host, topology.etcd_host
        )
        etcd_host, etcd_port = service_etcd_location(config)
        for plan in plans:
            self.assertEqual(plan.instance_config["pd_master_host"], etcd_host)
            self.assertEqual(plan.instance_config["pd_etcd_port"], etcd_port)
        self.assertEqual(sum(plan.is_benchmark_runner for plan in plans), 1)
        self.assertEqual(
            get_benchmark_endpoint(config, plans),
            f"http://{topology.router.host}:{topology.router.http_server_port}",
        )
        claims = [(host, port) for host, port, _ in service_port_claims(config)]
        self.assertEqual(len(claims), len(set(claims)))

    def test_all_pd_cases_use_explicit_coordinator_and_router(self):
        self.assertTrue(self.cases)
        reference = self.reference["deployment"]["xllm_service"]
        for path, config in self.cases:
            with self.subTest(path=str(path.relative_to(ROOT))):
                deployment = config["deployment"]
                self.assertNotIn("pd_control", deployment)
                self.assertNotIn("service_control", deployment)
                control = deployment["xllm_service"]
                for obsolete in (
                    "http_server_port", "rpc_server_port", "service_deployment_role"
                ):
                    self.assertNotIn(obsolete, control)
                for key in (
                    "service_binary_profile", "startup_attempts",
                    "startup_wait_seconds", "master_key_wait_seconds"
                ):
                    self.assertEqual(control[key], reference[key])
                for role in ("coordinators", "routers"):
                    self.assertTrue(control[role])
                    for instance in control[role]:
                        self.assertTrue(instance["container_instance"])
                        for port in ("http_server_port", "rpc_server_port"):
                            self.assertEqual(instance[port], reference[role][0][port])
                self.assertTrue(control["etcd"]["container_instance"])
                benchmark = deployment["benchmark"]
                self.assertEqual(benchmark["health_path"], "/readyz")
                self.assertIn(
                    benchmark["router_instance"],
                    [instance["name"] for instance in control["routers"]],
                )

    def test_all_pd_cases_load_and_bind_to_existing_engine_containers(self):
        for path, config in self.cases:
            with self.subTest(path=str(path.relative_to(ROOT))):
                loaded = TaskConfig.load_task_config(str(path))
                self.assertTrue(loaded)
                self.assertEqual(
                    loaded["deployment"]["xllm_service"],
                    config["deployment"]["xllm_service"],
                )
                self._assert_runtime_bindings(config)

    def test_all_pd_parameter_and_topology_candidates_keep_valid_bindings(self):
        for path, config in self.cases:
            space = _normalize_assess_parameters(config.get("assess_test_param"))
            if not space:
                continue
            topologies = normalize_assess_topologies(config, space)
            names = list(space)
            for index, values in enumerate(itertools.product(*space.values())):
                parameters = dict(zip(names, values))
                with self.subTest(
                    path=str(path.relative_to(ROOT)), parameters=parameters
                ):
                    candidate = build_parameter_candidate(
                        config, parameters, f"pd_yaml_candidate_{index}", topologies
                    )
                    self._assert_runtime_bindings(candidate)
