import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from xtask.tasks import parameter_search


def _search_config(strategy="tpe"):
    return {
        "case_name": "search_case",
        "metrics": ["perf"],
        "benchmark": {
            "benchmark_type": "aisbench",
            "perf": {
                "batch_size": 1,
                "num_prompts": 99,
                "input_len": 1024,
            },
        },
        "assess_test_param": {
            "benchmark.perf.batch_size": [1, 2],
            "benchmark.perf.input_len": [1024, 2048],
        },
        "assess_search": {
            "strategy": strategy,
            "max_trials": 3,
            "startup_trials": 1,
            "seed": 7,
            "objective": {
                "metric": "output_tps_total",
                "direction": "maximize",
            },
            "constraints": {
                "ttft_p90": {"max": 800},
                "tpot_p90": {"max": 25},
            },
        },
    }


class _FakeTrial:
    def __init__(self, number, planned_choices, running_state):
        self.number = number
        self._planned_choices = list(planned_choices)
        self.params = {}
        self.user_attrs = {}
        self.state = running_state
        self.value = None

    def suggest_categorical(self, name, choices):
        value = self._planned_choices.pop(0)
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class _FakeStudy:
    def __init__(self, planned_trials, states):
        self._planned_trials = list(planned_trials)
        self._states = states
        self.trials = []

    def ask(self):
        trial = _FakeTrial(
            len(self.trials),
            self._planned_trials.pop(0),
            self._states.RUNNING,
        )
        self.trials.append(trial)
        return trial

    def tell(self, trial, value=None, state=None):
        if isinstance(trial, int):
            trial = self.trials[trial]
        trial.value = value
        trial.state = state or self._states.COMPLETE

    def get_trials(self, deepcopy=False):
        return self.trials


def _fake_optuna(planned_trials):
    states = SimpleNamespace(
        RUNNING="running",
        PRUNED="pruned",
        FAIL="fail",
        COMPLETE="complete",
    )
    study = _FakeStudy(planned_trials, states)
    logging_api = SimpleNamespace(WARNING=30, set_verbosity=lambda value: None)
    created_study = {}

    def create_study(**kwargs):
        created_study.update(kwargs)
        return study

    optuna = SimpleNamespace(
        logging=logging_api,
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: object()),
        trial=SimpleNamespace(TrialState=states),
        create_study=create_study,
        created_study=created_study,
    )
    return optuna, study


class ParameterSearchTests(unittest.TestCase):
    def test_search_config_defaults_to_exhaustive(self):
        self.assertEqual(
            parameter_search.normalize_assess_search_config(None),
            {
                "strategy": "exhaustive",
                "objective": {
                    "metric": "output_tps_total",
                    "direction": "maximize",
                },
            },
        )
        self.assertEqual(
            parameter_search.normalize_assess_search_config(
                {"strategy": "exhaustive", "max_trials": 1}
            )["strategy"],
            "exhaustive",
        )

    def test_exhaustive_search_accepts_optional_objective_and_constraints(self):
        normalized = parameter_search.normalize_assess_search_config(
            {
                "strategy": "exhaustive",
                "objective": {
                    "metric": "ttft_p90",
                    "direction": "minimize",
                },
                "constraints": {"tpot_p90": {"max": 50}},
            }
        )
        self.assertEqual(
            normalized["objective"],
            {"metric": "ttft_p90", "direction": "minimize"},
        )
        self.assertEqual(
            normalized["constraints"], {"tpot_p90": {"max": 50.0}}
        )

    def test_tpe_search_config_is_validated(self):
        normalized = parameter_search.normalize_assess_search_config(
            _search_config()["assess_search"]
        )
        self.assertEqual(normalized["strategy"], "tpe")
        self.assertEqual(normalized["max_trials"], 3)
        self.assertEqual(normalized["startup_trials"], 1)
        self.assertEqual(normalized["constraints"]["ttft_p90"], {"max": 800.0})

        invalid = _search_config()["assess_search"]
        invalid["startup_trials"] = 4
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            parameter_search.normalize_assess_search_config(invalid)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parameter_search.normalize_assess_search_config(
                {"strategy": "pso"}
            )

    def test_candidate_uses_existing_parameter_path_and_batch_defaults(self):
        generated = parameter_search.build_parameter_candidate(
            _search_config(),
            {
                "benchmark.perf.batch_size": 2,
                "benchmark.perf.input_len": 2048,
            },
            "search_case_sg1",
        )
        self.assertEqual(generated["case_name"], "search_case_sg1")
        self.assertEqual(generated["benchmark"]["perf"]["batch_size"], 2)
        self.assertEqual(generated["benchmark"]["perf"]["num_prompts"], 8)
        self.assertEqual(generated["benchmark"]["perf"]["input_len"], 2048)

    def test_candidate_expands_named_topology_before_instance_parameters(self):
        config = {
            "case_name": "pd_search_case",
            "deployment": {
                "mode": "pd_disaggregated",
                "master_host": "10.0.0.1",
                "pd_instances": [
                    {
                        "name": "prefill-0",
                        "role": "prefill",
                        "nodes": [{"host": "10.0.0.1", "device_ids": [0]}],
                    },
                    {
                        "name": "decode-0",
                        "role": "decode",
                        "serving_extensiable_param": {"max_seqs_per_batch": 8},
                        "nodes": [{"host": "10.0.0.2", "device_ids": [0]}],
                    },
                ],
            },
            "assess_topologies": {
                "topo_1p1d": {
                    "pd_instances": [
                        {
                            "name": "prefill-0",
                            "role": "prefill",
                            "nodes": [
                                {"host": "10.0.0.1", "device_ids": [0, 1]}
                            ],
                        },
                        {
                            "name": "decode-0",
                            "role": "decode",
                            "serving_extensiable_param": {
                                "max_seqs_per_batch": 10
                            },
                            "nodes": [
                                {"host": "10.0.0.2", "device_ids": [0, 1]}
                            ],
                        },
                    ]
                }
            },
        }
        generated = parameter_search.build_parameter_candidate(
            config,
            {
                "deployment.pd_instances[role=decode].serving_extensiable_param.max_seqs_per_batch": 20,
                "deployment_topology": "topo_1p1d",
            },
            "pd_search_case_sg1",
        )
        self.assertEqual(
            generated["deployment"]["pd_instances"][1][
                "serving_extensiable_param"
            ]["max_seqs_per_batch"],
            20,
        )
        self.assertEqual(
            generated["deployment"]["pd_instances"][0]["nodes"][0][
                "device_ids"
            ],
            [0, 1],
        )

    def test_observation_applies_hard_constraints(self):
        result = {
            "benchmark_perf_result": SimpleNamespace(
                summary=SimpleNamespace(
                    output_tps_total=120.0,
                    ttft_p90=900.0,
                    tpot_p90=20.0,
                )
            )
        }
        config = parameter_search.normalize_assess_search_config(
            _search_config()["assess_search"]
        )
        observation = parameter_search.build_search_observation(result, config)
        self.assertEqual(observation["status"], "constraint_failed")
        self.assertFalse(observation["feasible"])
        self.assertEqual(observation["objective_value"], 120.0)
        self.assertLess(observation["score"], 0)

    def test_controller_skips_duplicate_and_writes_reports(self):
        search_case = parameter_search.SearchCase(
            case_name="search_case",
            base_config=_search_config(),
            raw_parameter_space={"benchmark.perf.batch_size": [1, 2]},
            parameter_space={"benchmark.perf.batch_size": [1, 2]},
            search_config=parameter_search.normalize_assess_search_config(
                {
                    "strategy": "tpe",
                    "max_trials": 2,
                    "startup_trials": 1,
                    "seed": 7,
                }
            ),
            topologies={},
        )
        fake_optuna, study = _fake_optuna([[0], [0], [1]])
        result = {
            "benchmark_perf_result": SimpleNamespace(
                summary=SimpleNamespace(output_tps_total=100.0)
            )
        }

        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            parameter_search, "_import_optuna", return_value=fake_optuna
        ):
            controller = parameter_search.TpeSearchController(
                search_case, Path(temporary_directory)
            )
            self.assertEqual(
                fake_optuna.created_study["study_name"],
                "search_case_tpe_output_tps_total",
            )
            first = controller.ask()
            controller.tell_result(first, result)
            second = controller.ask()
            controller.tell_result(second, result)

            self.assertEqual(first.parameters, {"benchmark.perf.batch_size": 1})
            self.assertEqual(second.parameters, {"benchmark.perf.batch_size": 2})
            self.assertEqual([first.sg_index, second.sg_index], [1, 2])
            self.assertEqual(study.trials[1].state, "pruned")
            self.assertIsNone(controller.ask())
            self.assertTrue(
                (Path(temporary_directory) / "search_summary.yaml").is_file()
            )
            self.assertTrue(
                (Path(temporary_directory) / "search_trials.csv").is_file()
            )

    def test_tpe_treats_topology_name_as_a_categorical_choice(self):
        search_case = parameter_search.SearchCase(
            case_name="pd_search_case",
            base_config={},
            raw_parameter_space={
                "deployment_topology": ["topo_1p1d", "topo_2p1d"]
            },
            parameter_space={
                "deployment_topology": ["topo_1p1d", "topo_2p1d"]
            },
            search_config=parameter_search.normalize_assess_search_config(
                {
                    "strategy": "tpe",
                    "max_trials": 1,
                    "startup_trials": 1,
                    "seed": 7,
                }
            ),
            topologies={
                "topo_1p1d": {"pd_instances": []},
                "topo_2p1d": {"pd_instances": []},
            },
        )
        fake_optuna, _ = _fake_optuna([[1]])

        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(
            parameter_search, "_import_optuna", return_value=fake_optuna
        ):
            controller = parameter_search.TpeSearchController(
                search_case, Path(temporary_directory)
            )
            candidate = controller.ask()

        self.assertEqual(
            candidate.parameters, {"deployment_topology": "topo_2p1d"}
        )

    def test_dynamic_runner_generates_one_case_per_asked_trial(self):
        config = _search_config()

        class FakeController:
            instances = []

            def __init__(self, search_case, state_dir):
                self.search_case = search_case
                self.total_combinations = 4
                self.candidates = [
                    parameter_search.SearchCandidate(
                        trial=object(),
                        sg_index=1,
                        parameters={
                            "benchmark.perf.batch_size": 1,
                            "benchmark.perf.input_len": 1024,
                        },
                        combination_key="one",
                    ),
                    parameter_search.SearchCandidate(
                        trial=object(),
                        sg_index=2,
                        parameters={
                            "benchmark.perf.batch_size": 2,
                            "benchmark.perf.input_len": 2048,
                        },
                        combination_key="two",
                    ),
                ]
                self.results = []
                self.failures = []
                self.__class__.instances.append(self)

            def ask(self):
                return self.candidates.pop(0) if self.candidates else None

            def tell_result(self, candidate, result):
                self.results.append((candidate, result))

            def tell_failure(self, candidate, error):
                self.failures.append((candidate, error))

            def write_reports(self):
                pass

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "search_case.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            args = SimpleNamespace(
                case_group=None,
                case_name="search_case",
                template=None,
                model_name=None,
                assess_test_type="para_set",
            )
            executed = []

            def run_trial(task_file, output_path, current_args):
                task_config = yaml.safe_load(Path(task_file).read_text())
                executed.append((task_config, output_path, current_args))
                return {
                    "success": True,
                    "task_config": task_config,
                    "benchmark_perf_result": SimpleNamespace(
                        summary=SimpleNamespace(output_tps_total=100.0)
                    ),
                }

            openme = mock.Mock()
            with mock.patch.object(
                parameter_search.Path, "cwd", return_value=root
            ), mock.patch.object(
                parameter_search, "TpeSearchController", FakeController
            ):
                results = parameter_search.run_tpe_parameter_search(
                    args,
                    openme,
                    run_trial,
                    output_root=root / "outputs",
                )

            self.assertEqual(len(results), 2)
            self.assertEqual(len(executed), 2)
            self.assertEqual(openme.AddCaseResult.call_count, 2)
            self.assertEqual(
                [item[0]["case_name"] for item in executed],
                ["search_case_sg1", "search_case_sg2"],
            )
            for index in (1, 2):
                metadata = yaml.safe_load(
                    (
                        root
                        / "outputs"
                        / f"search_case_sg{index}"
                        / "assess_test_param.yaml"
                    ).read_text()
                )
                self.assertEqual(metadata["assess_search"]["strategy"], "tpe")
                self.assertEqual(metadata["assess_search"]["trial_number"], index)


if __name__ == "__main__":
    unittest.main()
