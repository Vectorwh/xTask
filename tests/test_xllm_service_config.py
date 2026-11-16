"""Regression tests for the Service authoring-key migration (no live services)."""

import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from xtask.core import scheduler
from xtask.core.external_service import (
    topology_fields_ignored,
    validate_external_service_assess_params,
)
from xtask.core.pd_control_service import (
    XllmControlService,
    build_pd_control_config,
    build_service_control_config,
    build_xllm_service_config,
)
from xtask.core.result_processor import _resolve_host_addr
from xtask.core.service_group import (
    build_service_plans,
    get_benchmark_endpoint,
    normalize_pd_deployment,
    normalize_service_routed_deployment,
    validate_pd_deployment_authoring,
)
from xtask.core.xllm_service_config import (
    canonical_xllm_service_parameter_path,
    build_xllm_service_topology,
    has_service_instance_lists,
    get_xllm_service_config,
    normalize_xllm_service_config,
)
from xtask.tasks import parameter_search, task_parser
from xtask.tasks.task_parser import (
    _normalize_assess_parameters,
    _update_config_by_path,
)


CONTROL_KEYS = ("xllm_service", "pd_control", "service_control")


def _task(mode="pd_disaggregated", key="xllm_service", **control_overrides):
    control = {
        "host": "10.0.0.1",
        "etcd_client_port": 13389,
        "etcd_peer_port": 13390,
        "http_server_port": 13888,
        "rpc_server_port": 13889,
        "service_binary_profile": "legacy",
        "transfer_port_base": 26000,
    }
    control.update(control_overrides)
    deployment = {
        "mode": mode,
        "master_host": "10.0.0.1",
        key: control,
        "benchmark": {"runner_instance": "engine-0"},
    }
    if mode == "pd_disaggregated":
        deployment["pd_instances"] = [
            {
                "name": "prefill-0",
                "role": "prefill",
                "group_master_host": "10.0.0.1",
                "nodes": [
                    {
                        "name": "engine-0",
                        "host": "10.0.0.1",
                        "device_ids": [0, 1],
                        "local_port": 48000,
                    }
                ],
            },
            {
                "name": "decode-0",
                "role": "decode",
                "group_master_host": "10.0.0.2",
                "nodes": [
                    {
                        "name": "engine-1",
                        "host": "10.0.0.2",
                        "device_ids": [0, 1],
                        "local_port": 48000,
                    }
                ],
            },
        ]
    else:
        deployment["service_instances"] = [
            {
                "name": "default-0",
                "nodes": [
                    {
                        "name": "engine-0",
                        "host": "10.0.0.1",
                        "device_ids": [0, 1],
                        "local_port": 48000,
                    }
                ],
            }
        ]
    return {"engine_type": "xllm", "model_path": "/models/test", "deployment": deployment}


def _command(config):
    runtime = SimpleNamespace(
        container_manager=mock.Mock(),
        container_config=SimpleNamespace(container_name="service-test"),
    )
    with mock.patch(
        "xtask.utils.container_manager.path_mapping_in_container",
        side_effect=lambda path: path,
    ):
        service = XllmControlService(runtime, config, "/models/test", "/tmp/case")
        return service._build_start_command()


class XllmServiceNormalizationTests(unittest.TestCase):
    def test_legacy_mapping_moves_once_without_copying_or_flattening(self):
        for key in ("pd_control", "service_control"):
            with self.subTest(key=key):
                task = _task(key=key)
                control = task["deployment"][key]
                normalize_xllm_service_config(task)
                self.assertIs(task["deployment"]["xllm_service"], control)
                self.assertNotIn(key, task["deployment"])
                self.assertIn("pd_instances", task["deployment"])
                self.assertNotIn("instances", task["deployment"])
                normalized = copy.deepcopy(task)
                self.assertIs(normalize_xllm_service_config(task), task)
                self.assertEqual(task, normalized)

    def test_canonical_mapping_and_absent_block_are_unchanged(self):
        for task in ({}, {"deployment": {}}, _task()):
            with self.subTest(task=task):
                before = copy.deepcopy(task)
                normalize_xllm_service_config(task)
                self.assertEqual(task, before)

    def test_null_block_keeps_legacy_default_semantics(self):
        for key in CONTROL_KEYS:
            with self.subTest(key=key):
                task = {"deployment": {key: None}}
                self.assertEqual(get_xllm_service_config(task), {})
                self.assertIsNone(task["deployment"]["xllm_service"])

    def test_conflicting_spellings_fail_before_mutating(self):
        for pair in itertools.combinations(CONTROL_KEYS, 2):
            with self.subTest(pair=pair):
                task = {"deployment": {key: {"host": "10.0.0.1"} for key in pair}}
                before = copy.deepcopy(task)
                with self.assertRaisesRegex(ValueError, "Conflicting xLLM-Service"):
                    normalize_xllm_service_config(task)
                self.assertEqual(task, before)

    def test_invalid_mapping_fails_early(self):
        for key in CONTROL_KEYS:
            for value in ([], True, "10.0.0.1", 123):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(TypeError, "must be a mapping"):
                        normalize_xllm_service_config({"deployment": {key: value}})

    def test_non_mapping_task_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "task_config must be a mapping"):
            normalize_xllm_service_config(None)

    def test_authoring_validation_normalizes_without_expanding_engine_nodes(self):
        for mode, legacy in (
            ("pd_disaggregated", "pd_control"),
            ("service_routed", "service_control"),
        ):
            with self.subTest(mode=mode):
                task = _task(mode, legacy)
                validate_pd_deployment_authoring(task)
                self.assertIn("xllm_service", task["deployment"])
                self.assertNotIn("instances", task["deployment"])

    def test_both_engine_normalizers_use_the_canonical_block(self):
        for mode, legacy, normalizer in (
            ("pd_disaggregated", "pd_control", normalize_pd_deployment),
            ("service_routed", "service_control", normalize_service_routed_deployment),
        ):
            with self.subTest(mode=mode):
                task = _task(mode, legacy)
                normalizer(task)
                self.assertIn("instances", task["deployment"])
                self.assertIn("xllm_service", task["deployment"])
                self.assertNotIn(legacy, task["deployment"])


class XllmServiceBehaviorTests(unittest.TestCase):
    def test_legacy_and_canonical_builders_match_all_fields_and_commands(self):
        for mode, legacy, wrapper in (
            ("pd_disaggregated", "pd_control", build_pd_control_config),
            ("service_routed", "service_control", build_service_control_config),
        ):
            for profile in ("legacy", "refactor"):
                with self.subTest(mode=mode, profile=profile):
                    old_task = _task(mode, legacy, service_binary_profile=profile)
                    new_task = _task(mode, service_binary_profile=profile)
                    old_task["deployment"][legacy]["internal_api_token"] = "test-shared-secret"
                    new_task["deployment"]["xllm_service"]["internal_api_token"] = "test-shared-secret"
                    old = wrapper(old_task)
                    new = build_xllm_service_config(new_task)
                    self.assertEqual(old, new)
                    self.assertEqual(_command(old), _command(new))
                    self.assertEqual(new.output_subdir, legacy)
                    self.assertEqual(
                        "ENABLE_DECODE_RESPONSE_TO_SERVICE=true" in _command(new),
                        mode == "pd_disaggregated",
                    )

    def test_all_existing_defaults_remain_unchanged(self):
        task = _task()
        task["deployment"]["xllm_service"] = {"host": "10.0.0.1"}
        with mock.patch.dict("os.environ", {"XLLM_INTERNAL_TOKEN": ""}):
            config = build_xllm_service_config(task)
        self.assertEqual((config.etcd_client_port, config.etcd_peer_port), (3389, 3390))
        self.assertEqual((config.http_server_port, config.rpc_server_port), (38888, 38889))
        self.assertEqual((config.startup_attempts, config.startup_wait_seconds), (2, 3))
        self.assertEqual(config.master_key_wait_seconds, 30)
        self.assertEqual(config.master_service_key, "XLLM:SERVICE:MASTER")
        self.assertEqual(config.service_deployment_role, "COORDINATOR")
        self.assertEqual(config.service_binary_profile, "auto")
        self.assertIn("10.0.0.2", config.no_proxy_hosts)

    def test_host_fallback_and_token_precedence_are_preserved(self):
        task = _task()
        del task["deployment"]["xllm_service"]["host"]
        with mock.patch.dict("os.environ", {"XLLM_INTERNAL_TOKEN": "environment-secret"}):
            config = build_xllm_service_config(copy.deepcopy(task))
            self.assertEqual(config.host, "10.0.0.1")
            self.assertEqual(config.internal_api_token, "environment-secret")
            task["deployment"]["xllm_service"]["internal_api_token"] = "yaml-secret"
            self.assertEqual(build_xllm_service_config(task).internal_api_token, "yaml-secret")

    def test_endpoints_preserve_mode_specific_validation(self):
        for mode, legacy in (
            ("pd_disaggregated", "pd_control"),
            ("service_routed", "service_control"),
        ):
            with self.subTest(mode=mode):
                for key in ("xllm_service", legacy):
                    task = _task(mode, key)
                    task["deployment"]["benchmark"]["endpoint"] = "http://wrong:9999"
                    self.assertEqual(
                        get_benchmark_endpoint(task, []), "http://10.0.0.1:13888"
                    )
        task = _task()
        del task["deployment"]["xllm_service"]["http_server_port"]
        with self.assertRaisesRegex(ValueError, "xllm_service.http_server_port"):
            get_benchmark_endpoint(task, [])
        task = _task("service_routed")
        task["deployment"]["xllm_service"] = {}
        self.assertEqual(get_benchmark_endpoint(task, []), "http://10.0.0.1:38888")

    def test_engine_plan_connections_and_transfer_defaults_are_preserved(self):
        for mode, legacy in (
            ("pd_disaggregated", "pd_control"),
            ("service_routed", "service_control"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as output:
                old = build_service_plans(_task(mode, legacy), output)
                new = build_service_plans(_task(mode), output)
                self.assertEqual(
                    [p.instance_config for p in old],
                    [p.instance_config for p in new],
                )
                for plan in new:
                    cfg = plan.instance_config
                    self.assertNotIn(legacy, cfg["deployment"])
                    if mode == "pd_disaggregated":
                        self.assertEqual(cfg["pd_master_host"], "10.0.0.1")
                        self.assertEqual(cfg["pd_etcd_port"], 13389)
                        self.assertEqual(cfg["transfer_port_base"], 26000)
                    else:
                        self.assertEqual(cfg["service_control_host"], "10.0.0.1")
                        self.assertEqual(cfg["service_etcd_port"], 13389)

    def test_control_host_and_reserved_ports_accept_both_spellings(self):
        for mode, legacy in (
            ("pd_disaggregated", "pd_control"),
            ("service_routed", "service_control"),
        ):
            for key in ("xllm_service", legacy):
                with self.subTest(mode=mode, key=key):
                    task = _task(mode, key)
                    self.assertEqual(scheduler._xllm_control_host(task), "10.0.0.1")
                    self.assertEqual(
                        scheduler._control_plane_ports(task),
                        {13389, 13390, 13888, 13889},
                    )
                    self.assertEqual(_resolve_host_addr(task), "10.0.0.1")

    def test_port_collisions_are_still_rejected(self):
        for mode in ("pd_disaggregated", "service_routed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as output:
                with self.assertRaisesRegex(ValueError, "[Pp]ort"):
                    build_service_plans(_task(mode, http_server_port=48000), output)

    def test_external_mode_does_not_take_over_service_startup(self):
        task = _task()
        task["external_service"] = True
        task["deployment"]["benchmark"]["endpoint"] = "http://external:8800"
        self.assertEqual(get_benchmark_endpoint(task, []), "http://external:8800")
        context = SimpleNamespace(state={"external_service": True})
        with mock.patch(
            "xtask.core.scheduler._start_and_wait_service_multi_host"
        ) as start, mock.patch("xtask.core.scheduler._probe_external_service") as probe:
            scheduler._start_and_wait_service(context)
        start.assert_not_called()
        probe.assert_called_once_with(context)
        self.assertIn("deployment.xllm_service", topology_fields_ignored(task))
        task["assess_test_param"] = {"deployment.xllm_service.http_server_port": [13888, 14888]}
        with self.assertRaisesRegex(ValueError, "external_service does not support"):
            validate_external_service_assess_params(task)

    def test_auto_assess_reuse_signature_normalizes_without_mutating(self):
        old = _task(key="pd_control")
        new = _task()
        before = copy.deepcopy(old)
        self.assertEqual(
            scheduler._benchmark_only_service_config(old),
            scheduler._benchmark_only_service_config(new),
        )
        self.assertEqual(old, before)
        new["deployment"]["xllm_service"]["rpc_server_port"] += 1
        self.assertNotEqual(
            scheduler._benchmark_only_service_config(old),
            scheduler._benchmark_only_service_config(new),
        )


class XllmServiceSearchPathTests(unittest.TestCase):
    def test_legacy_explicit_paths_update_the_canonical_block(self):
        for block in CONTROL_KEYS:
            for path_key in CONTROL_KEYS:
                with self.subTest(block=block, path_key=path_key):
                    task = _task(key=block)
                    changed = _update_config_by_path(
                        task, f"deployment.{path_key}.http_server_port", 14888
                    )
                    self.assertEqual(changed, ["deployment.xllm_service.http_server_port"])
                    self.assertEqual(task["deployment"]["xllm_service"]["http_server_port"], 14888)

    def test_nested_and_dotted_search_spaces_use_canonical_paths(self):
        for key in CONTROL_KEYS:
            with self.subTest(key=key):
                expected = {"deployment.xllm_service.http_server_port": [13888, 14888]}
                self.assertEqual(
                    _normalize_assess_parameters({
                        "deployment": {key: {"http_server_port": [13888, 14888]}}
                    }),
                    expected,
                )
                self.assertEqual(
                    _normalize_assess_parameters({
                        f"deployment.{key}.http_server_port": [13888, 14888]
                    }),
                    expected,
                )

    def test_alias_search_path_collision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate assess_test_param path"):
            _normalize_assess_parameters({
                "deployment.pd_control.http_server_port": [13888],
                "deployment.xllm_service.http_server_port": [14888],
            })

    def test_other_parameter_paths_and_typo_validation_are_unchanged(self):
        self.assertEqual(
            canonical_xllm_service_parameter_path("benchmark.perf.batch_size"),
            "benchmark.perf.batch_size",
        )
        self.assertEqual(
            canonical_xllm_service_parameter_path("deployment.pd_control_extra.port"),
            "deployment.pd_control_extra.port",
        )
        with self.assertRaises(KeyError):
            _update_config_by_path(_task(), "deployment.xllm_service.typo", 10)


class XllmServiceGeneratedTaskTests(unittest.TestCase):
    """Exercise persisted candidates and real loader/session paths, not mocks."""

    @staticmethod
    def _source_case(mode, key):
        task = _task(mode, key)
        task["case_name"] = "service_review_case"
        task["metrics"] = ["perf"]
        task["benchmark"] = {
            "benchmark_type": "aisbench",
            "perf": {"batch_size": 1, "num_prompts": 17},
        }
        return task

    def test_auto_assess_default_round_normalizes_without_changing_prompts(self):
        for mode in ("pd_disaggregated", "service_routed"):
            for key in CONTROL_KEYS:
                with self.subTest(mode=mode, key=key), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = self._source_case(mode, key)
                    case = root / "source.yaml"
                    case.write_text(yaml.safe_dump(source), encoding="utf-8")
                    original = case.read_bytes()
                    tasks = {}
                    with mock.patch.object(task_parser, "find_case_file", return_value=str(case)):
                        self.assertTrue(task_parser.gen_group_autoassess_task(
                            "service_review_case", str(root / "out"), tasks, {}
                        ))
                    generated = Path(next(iter(tasks)))
                    authored = yaml.safe_load(generated.read_text(encoding="utf-8"))
                    self.assertIn("xllm_service", authored["deployment"])
                    self.assertNotIn("pd_control", authored["deployment"])
                    self.assertNotIn("service_control", authored["deployment"])
                    self.assertNotIn("instances", authored["deployment"])
                    self.assertEqual(authored["benchmark"], source["benchmark"])
                    self.assertEqual(case.read_bytes(), original)
                    self.assertTrue(scheduler.TaskConfig.load_task_config(str(generated)))

    def test_auto_assess_repeated_rounds_reuse_the_real_loaded_service_config(self):
        for mode, key in (
            ("pd_disaggregated", "pd_control"),
            ("service_routed", "service_control"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                case = root / "source.yaml"
                source = self._source_case(mode, key)
                case.write_text(yaml.safe_dump(source), encoding="utf-8")
                before = case.read_bytes()
                context = SimpleNamespace(args=None, state={})
                for index, params in enumerate(({}, {"batch_size": 2}, {"batch_size": 4})):
                    tasks = {}
                    with mock.patch.object(task_parser, "find_case_file", return_value=str(case)):
                        self.assertTrue(task_parser.gen_group_autoassess_task(
                            "service_review_case", str(root / "out"), tasks, params
                        ))
                    generated = next(iter(tasks))
                    if index == 0:
                        context.state["source_task_config"] = (
                            scheduler.TaskConfig.load_task_config(generated)
                        )
                    loaded = scheduler._load_session_trial_config(context, generated)
                    perf = loaded["benchmark"]["perf"]
                    self.assertEqual(perf["batch_size"], params.get("batch_size", 1))
                    self.assertEqual(
                        perf["num_prompts"], 17 if not params else params["batch_size"] * 4
                    )
                    self.assertEqual(get_benchmark_endpoint(loaded, []), "http://10.0.0.1:13888")
                self.assertEqual(case.read_bytes(), before)

    def test_auto_assess_conflicting_blocks_fail_before_creating_candidates(self):
        for params in ({}, {"batch_size": 2}):
            with self.subTest(params=params), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = self._source_case("pd_disaggregated", "pd_control")
                source["deployment"]["xllm_service"] = {"host": "10.0.0.9"}
                case = root / "source.yaml"
                case.write_text(yaml.safe_dump(source), encoding="utf-8")
                tasks = {}
                with mock.patch.object(task_parser, "find_case_file", return_value=str(case)):
                    self.assertFalse(task_parser.gen_group_autoassess_task(
                        "service_review_case", str(root / "out"), tasks, params
                    ))
                self.assertFalse(tasks)
                self.assertFalse((root / "out").exists())

    def test_exhaustive_search_generated_endpoints_follow_each_candidate(self):
        for mode, key in (
            ("pd_disaggregated", "pd_control"),
            ("service_routed", "service_control"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = self._source_case(mode, key)
                source["assess_test_param"] = {
                    "deployment": {key: {"http_server_port": [13888, 14888]}},
                    "benchmark.perf.batch_size": [1, 2],
                }
                case = root / "source.yaml"
                case.write_text(yaml.safe_dump(source), encoding="utf-8")
                original = case.read_bytes()
                tasks = {}
                with mock.patch.object(task_parser, "find_case_file", return_value=str(case)):
                    self.assertTrue(task_parser.gen_group_paraset_task(
                        "service_review_case", str(root / "out"), tasks
                    ))
                self.assertEqual(len(tasks), 4)
                observed = set()
                for generated in tasks:
                    authored = yaml.safe_load(Path(generated).read_text(encoding="utf-8"))
                    self.assertNotIn(key, authored["deployment"])
                    loaded = scheduler.TaskConfig.load_task_config(generated)
                    config = build_xllm_service_config(loaded)
                    endpoint = get_benchmark_endpoint(loaded, [])
                    self.assertEqual(endpoint, f"http://10.0.0.1:{config.http_server_port}")
                    perf = loaded["benchmark"]["perf"]
                    observed.add((
                        config.http_server_port, perf["batch_size"], perf["num_prompts"]
                    ))
                    metadata = yaml.safe_load(
                        (Path(generated).parent / "assess_test_param.yaml").read_text()
                    )
                    self.assertEqual(metadata["assess_test_param"], source["assess_test_param"])
                    self.assertIn(
                        "deployment.xllm_service.http_server_port", metadata["current_combination"]
                    )
                self.assertEqual(
                    observed, {(13888, 1, 4), (13888, 2, 8), (14888, 1, 4), (14888, 2, 8)}
                )
                self.assertEqual(case.read_bytes(), original)

    def test_tpe_load_write_and_scheduler_load_keep_canonical_control(self):
        for mode, key in (
            ("pd_disaggregated", "pd_control"),
            ("service_routed", "service_control"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = self._source_case(mode, key)
                source["assess_test_param"] = {
                    f"deployment.{key}.http_server_port": [13888, 14888],
                    "benchmark.perf.batch_size": [1, 2],
                }
                source["assess_search"] = {"strategy": "tpe", "max_trials": 2}
                case = root / "source.yaml"
                case.write_text(yaml.safe_dump(source), encoding="utf-8")
                with mock.patch.object(parameter_search, "find_case_file", return_value=str(case)):
                    search = parameter_search.load_search_case("service_review_case")
                self.assertIn("deployment.xllm_service.http_server_port", search.parameter_space)
                before = copy.deepcopy(search.base_config)
                for index, port in enumerate((13888, 14888), 1):
                    candidate = parameter_search.SearchCandidate(
                        trial=None,
                        sg_index=index,
                        parameters={
                            "deployment.xllm_service.http_server_port": port,
                            "benchmark.perf.batch_size": index,
                        },
                        combination_key=str(index),
                    )
                    generated, _ = parameter_search.write_parameter_candidate(
                        search, candidate, root / "out"
                    )
                    authored = yaml.safe_load(Path(generated).read_text(encoding="utf-8"))
                    self.assertNotIn(key, authored["deployment"])
                    loaded = scheduler.TaskConfig.load_task_config(generated)
                    self.assertEqual(get_benchmark_endpoint(loaded, []), f"http://10.0.0.1:{port}")
                    self.assertEqual(loaded["benchmark"]["perf"]["num_prompts"], index * 4)
                self.assertEqual(search.base_config, before)

    def test_real_session_accepts_alias_migration_but_rejects_control_changes(self):
        for mode, key in (
            ("pd_disaggregated", "pd_control"),
            ("service_routed", "service_control"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = self._source_case(mode, key)
                base = root / "base.yaml"
                base.write_text(yaml.safe_dump(source), encoding="utf-8")
                context = SimpleNamespace(
                    args=None,
                    state={"source_task_config": scheduler.TaskConfig.load_task_config(str(base))},
                )
                trial_config = copy.deepcopy(source)
                normalize_xllm_service_config(trial_config)
                trial_config["benchmark"]["perf"]["batch_size"] = 2
                trial = root / "trial.yaml"
                trial.write_text(yaml.safe_dump(trial_config), encoding="utf-8")
                self.assertTrue(scheduler._load_session_trial_config(context, str(trial)))
                for field, value in (
                    ("host", "10.0.0.2"),
                    ("http_server_port", 14888),
                    ("rpc_server_port", 14889),
                    ("service_deployment_role", "ROUTER"),
                ):
                    changed = copy.deepcopy(trial_config)
                    changed["deployment"]["xllm_service"][field] = value
                    trial.write_text(yaml.safe_dump(changed), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "service or deployment configuration changed"
                    ):
                        scheduler._load_session_trial_config(context, str(trial))


class AuthoredXllmServiceCaseTests(unittest.TestCase):
    def test_all_authored_cases_use_only_the_canonical_control_key(self):
        root = Path(__file__).resolve().parent / "e2e" / "xllm"
        managed = []
        for path in root.rglob("*.yaml"):
            task = yaml.safe_load(path.read_text(encoding="utf-8"))
            deployment = task.get("deployment") or {}
            self.assertNotIn("pd_control", deployment, str(path))
            self.assertNotIn("service_control", deployment, str(path))
            if "xllm_service" not in deployment:
                continue
            managed.append(path)
            validate_pd_deployment_authoring(task)
            config = build_xllm_service_config(task)
            if has_service_instance_lists(task):
                router = build_xllm_service_topology(task).router
                expected = f"http://{router.host}:{router.http_server_port}"
            else:
                expected = f"http://{config.host}:{config.http_server_port}"
            self.assertEqual(get_benchmark_endpoint(task, []), expected)
        self.assertTrue(managed)
