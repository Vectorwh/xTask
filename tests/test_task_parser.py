import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from xtask.tasks.task_parser import (
    _normalize_assess_parameters,
    _update_config_by_path,
    dynamic_loops_from_dict,
    gen_group_paraset_task,
)
from xtask.tasks.topology_search import normalize_assess_topologies


def _pd_config(assess_test_param):
    return {
        "case_name": "multi_host_case",
        "benchmark": {
            "benchmark_type": "aisbench",
            "perf": {
                "batch_size": 1,
                "num_prompts": 99,
                "input_len": 1024,
            },
        },
        "deployment": {
            "mode": "pd_disaggregated",
            "pd_instances": [
                {
                    "name": "prefill-0",
                    "role": "prefill",
                    "serving_extensiable_param": {
                        "max_seqs_per_batch": 10,
                        "enable_graph": False,
                    },
                    "nodes": [
                        {
                            "name": "prefill-0-node-0",
                            "host": "10.0.0.1",
                            "device_ids": [0, 1],
                        }
                    ],
                },
                {
                    "name": "decode-0",
                    "role": "decode",
                    "serving_extensiable_param": {
                        "max_seqs_per_batch": 10,
                        "enable_graph": True,
                    },
                    "nodes": [
                        {
                            "name": "decode-0-node-0",
                            "host": "10.0.0.2",
                            "device_ids": [2, 3],
                        }
                    ],
                },
            ]
        },
        "assess_test_param": assess_test_param,
    }


class TaskParserTests(unittest.TestCase):
    def test_dynamic_loops_cartesian_product_and_stable_set_order(self):
        combinations = list(
            dynamic_loops_from_dict(
                {"batch_size": {2, 1}, "max_seqs_per_batch": [8, 16]}
            )
        )
        self.assertEqual(
            combinations,
            [
                {"batch_size": 1, "max_seqs_per_batch": 8},
                {"batch_size": 1, "max_seqs_per_batch": 16},
                {"batch_size": 2, "max_seqs_per_batch": 8},
                {"batch_size": 2, "max_seqs_per_batch": 16},
            ],
        )

    def test_normalize_assess_parameters_accepts_nested_paths(self):
        normalized = _normalize_assess_parameters(
            {
                "benchmark": {"perf": {"batch_size": [1, 2]}},
                "deployment": {
                    "pd_instances[name=prefill-0]": {
                        "serving_extensiable_param": {
                            "max_seqs_per_batch": [8, 16]
                        }
                    }
                },
            }
        )
        self.assertEqual(
            normalized,
            {
                "benchmark.perf.batch_size": [1, 2],
                "deployment.pd_instances[name=prefill-0].serving_extensiable_param.max_seqs_per_batch": [
                    8,
                    16,
                ],
            },
        )

    def test_explicit_benchmark_path_and_batch_size_default(self):
        config = {"benchmark": {"perf": {"batch_size": 1, "num_prompts": 2}}}
        changed = _update_config_by_path(config, "benchmark.perf.batch_size", 3)
        self.assertEqual(changed, ["benchmark.perf.batch_size"])
        self.assertEqual(config["benchmark"]["perf"]["batch_size"], 3)
        self.assertEqual(config["benchmark"]["perf"]["num_prompts"], 12)

    def test_named_topologies_validate_references_and_pd_layouts(self):
        config = _pd_config({"deployment_topology": ["topo_1p1d"]})
        config["assess_topologies"] = {
            "topo_1p1d": {
                "pd_instances": copy.deepcopy(
                    config["deployment"]["pd_instances"]
                )
            }
        }
        normalized = _normalize_assess_parameters(config["assess_test_param"])
        topologies = normalize_assess_topologies(config, normalized)
        self.assertEqual(list(topologies), ["topo_1p1d"])

        config["assess_test_param"]["deployment_topology"] = ["missing"]
        normalized = _normalize_assess_parameters(config["assess_test_param"])
        with self.assertRaisesRegex(KeyError, "Unknown deployment topology"):
            normalize_assess_topologies(config, normalized)

    def test_exhaustive_para_set_expands_named_topologies_and_metadata(self):
        config = _pd_config(
            {
                "deployment.pd_instances[role=decode].serving_extensiable_param.max_seqs_per_batch": [
                    20
                ],
                "deployment_topology": ["topo_1p1d", "topo_1p2node_d"],
            }
        )
        one_node = copy.deepcopy(config["deployment"]["pd_instances"])
        two_decode_nodes = copy.deepcopy(one_node)
        two_decode_nodes[1]["nodes"] = [
            {
                "name": "decode-0-node-0",
                "host": "10.0.0.2",
                "device_ids": [0, 1],
            },
            {
                "name": "decode-0-node-1",
                "host": "10.0.0.3",
                "device_ids": [0, 1],
            },
        ]
        config["assess_topologies"] = {
            "topo_1p1d": {"pd_instances": one_node},
            "topo_1p2node_d": {"pd_instances": two_decode_nodes},
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "multi_host_case.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            tasks = {}
            with mock.patch(
                "xtask.tasks.task_parser.Path.cwd", return_value=root
            ):
                self.assertTrue(
                    gen_group_paraset_task(
                        "multi_host_case", str(root / "out"), tasks
                    )
                )

            self.assertEqual(len(tasks), 2)
            decode_node_counts = set()
            resolved_names = set()
            for task_file in tasks:
                task_path = Path(task_file)
                generated = yaml.safe_load(task_path.read_text())
                decode = generated["deployment"]["pd_instances"][1]
                decode_node_counts.add(len(decode["nodes"]))
                self.assertEqual(
                    decode["serving_extensiable_param"]["max_seqs_per_batch"],
                    20,
                )
                metadata = yaml.safe_load(
                    (task_path.parent / "assess_test_param.yaml").read_text()
                )
                resolved_names.add(metadata["resolved_topology"]["name"])
                self.assertEqual(
                    metadata["resolved_topology"]["pd_instances"],
                    generated["deployment"]["pd_instances"],
                )

            self.assertEqual(decode_node_counts, {1, 2})
            self.assertEqual(
                resolved_names, {"topo_1p1d", "topo_1p2node_d"}
            )

    def test_pd_instance_selectors_only_change_selected_logical_instances(self):
        config = _pd_config({})
        original = copy.deepcopy(config)

        _update_config_by_path(
            config,
            "deployment.pd_instances[name=prefill-0].serving_extensiable_param.max_seqs_per_batch",
            20,
        )
        self.assertEqual(
            config["deployment"]["pd_instances"][0]["serving_extensiable_param"][
                "max_seqs_per_batch"
            ],
            20,
        )
        self.assertEqual(
            config["deployment"]["pd_instances"][1]["serving_extensiable_param"][
                "max_seqs_per_batch"
            ],
            10,
        )

        _update_config_by_path(
            config,
            "deployment.pd_instances[*].serving_extensiable_param.max_seqs_per_batch",
            30,
        )
        self.assertEqual(
            [
                instance["serving_extensiable_param"]["max_seqs_per_batch"]
                for instance in config["deployment"]["pd_instances"]
            ],
            [30, 30],
        )
        self.assertEqual(
            config["deployment"]["pd_instances"][0]["nodes"],
            original["deployment"]["pd_instances"][0]["nodes"],
        )

        # Role selection may target all logical instances of one PD role.
        _update_config_by_path(
            config,
            "deployment.pd_instances[role=decode].serving_extensiable_param.enable_graph",
            False,
        )
        self.assertEqual(
            config["deployment"]["pd_instances"][0]["serving_extensiable_param"][
                "enable_graph"
            ],
            False,
        )
        self.assertEqual(
            config["deployment"]["pd_instances"][1]["serving_extensiable_param"][
                "enable_graph"
            ],
            False,
        )

    def test_non_pd_flat_instance_selectors_remain_supported(self):
        config = {
            "deployment": {
                "instances": [
                    {"name": "master-0", "role": "master", "local_port": 8800},
                    {"name": "worker-0", "role": "worker", "local_port": 8800},
                ]
            }
        }

        _update_config_by_path(
            config,
            "deployment.instances[name=worker-0].local_port",
            8900,
        )

        self.assertEqual(config["deployment"]["instances"][0]["local_port"], 8800)
        self.assertEqual(config["deployment"]["instances"][1]["local_port"], 8900)

    def test_unknown_explicit_parameter_path_is_rejected(self):
        config = _pd_config({})
        with self.assertRaises(KeyError):
            _update_config_by_path(
                config,
                "deployment.pd_instances[name=prefill-0].serving_extensiable_param.not_a_real_parameter",
                1,
            )

    def test_para_set_deep_copies_combinations_and_applies_batch_defaults(self):
        config = _pd_config(
            {
                "batch_size": [1, 2],
                "deployment.pd_instances[name=prefill-0].serving_extensiable_param.max_seqs_per_batch": [
                    8,
                    16,
                ],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case_dir = root / "tests" / "multi_host"
            case_dir.mkdir(parents=True)
            case_file = case_dir / "multi_host_case.yaml"
            case_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            output_dir = root / "outputs"
            tasks = {}

            with mock.patch(
                "xtask.tasks.task_parser.Path.cwd", return_value=root
            ):
                self.assertTrue(
                    gen_group_paraset_task("multi_host_case", str(output_dir), tasks)
                )

            self.assertEqual(len(tasks), 4)
            generated = []
            metadata_by_case = {}
            for task_file in tasks:
                task_path = Path(task_file)
                generated.append(yaml.safe_load(task_path.read_text()))
                metadata_path = task_path.parent / "assess_test_param.yaml"
                metadata_by_case[task_path.stem] = yaml.safe_load(
                    metadata_path.read_text()
                )

            self.assertEqual(
                sorted(metadata_by_case),
                [
                    "multi_host_case_sg1",
                    "multi_host_case_sg2",
                    "multi_host_case_sg3",
                    "multi_host_case_sg4",
                ],
            )
            for metadata in metadata_by_case.values():
                self.assertEqual(
                    set(metadata), {"current_combination", "assess_test_param"}
                )
                self.assertEqual(
                    metadata["assess_test_param"],
                    config["assess_test_param"],
                )

            values = {
                (
                    item["benchmark"]["perf"]["batch_size"],
                    item["benchmark"]["perf"]["num_prompts"],
                    item["deployment"]["pd_instances"][0][
                        "serving_extensiable_param"
                    ]["max_seqs_per_batch"],
                    item["deployment"]["pd_instances"][1][
                        "serving_extensiable_param"
                    ]["max_seqs_per_batch"],
                )
                for item in generated
            }
            self.assertEqual(
                values,
                {(1, 4, 8, 10), (1, 4, 16, 10), (2, 8, 8, 10), (2, 8, 16, 10)},
            )

    def test_para_set_explicit_num_prompts_wins_over_batch_default(self):
        config = _pd_config(
            {"batch_size": [2], "num_prompts": [7]}
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case_dir = root / "tests"
            case_dir.mkdir()
            (case_dir / "multi_host_case.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            tasks = {}
            with mock.patch(
                "xtask.tasks.task_parser.Path.cwd", return_value=root
            ):
                self.assertTrue(
                    gen_group_paraset_task("multi_host_case", str(root / "out"), tasks)
                )
            self.assertEqual(len(tasks), 1)
            generated_path = Path(next(iter(tasks)))
            generated = yaml.safe_load(generated_path.read_text())
            self.assertEqual(generated["benchmark"]["perf"]["batch_size"], 2)
            self.assertEqual(generated["benchmark"]["perf"]["num_prompts"], 7)
            self.assertEqual(generated_path.stem, "multi_host_case_sg1")
            metadata = yaml.safe_load(
                (generated_path.parent / "assess_test_param.yaml").read_text()
            )
            self.assertEqual(
                metadata["current_combination"],
                {"batch_size": 2, "num_prompts": 7},
            )

    def test_para_set_uses_short_names_for_values_that_need_sanitizing(self):
        config = _pd_config(
            {"input_len": ["a/b", "a b"]}
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case_dir = root / "tests"
            case_dir.mkdir()
            (case_dir / "multi_host_case.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            tasks = {}
            output_dir = root / "out"
            with mock.patch(
                "xtask.tasks.task_parser.Path.cwd", return_value=root
            ):
                self.assertTrue(
                    gen_group_paraset_task("multi_host_case", str(output_dir), tasks)
                )
            self.assertEqual(len(tasks), 2)
            self.assertEqual(
                sorted(Path(task).stem for task in tasks),
                ["multi_host_case_sg1", "multi_host_case_sg2"],
            )
            combinations = []
            for task in tasks:
                metadata = yaml.safe_load(
                    (Path(task).parent / "assess_test_param.yaml").read_text()
                )
                combinations.append(metadata["current_combination"]["input_len"])
            self.assertEqual(combinations, ["a/b", "a b"])


if __name__ == "__main__":
    unittest.main()
