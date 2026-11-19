"""Adaptive parameter search for ``para_set`` tasks.

The existing exhaustive Cartesian-product implementation remains in
``task_parser``.  This module is imported by the main entry point, but Optuna
is loaded lazily only when a case explicitly selects the TPE strategy.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from xtask.tasks.task_parser import (
    _apply_batch_size_defaults,
    _normalize_assess_parameters,
    _update_config_by_path,
    find_case_file,
    get_case_list_from_group,
    init_task_path,
)
from xtask.tasks.topology_search import (
    ASSESS_TOPOLOGY_PARAMETER,
    apply_assess_topology,
    normalize_assess_topologies,
    resolved_topology_metadata,
)


SUPPORTED_SEARCH_STRATEGIES = {"exhaustive", "tpe"}
DEFAULT_OBJECTIVE_METRIC = "output_tps_total"
DEFAULT_MAX_TRIALS = 20
DEFAULT_STARTUP_TRIALS = 5


@dataclass
class SearchCase:
    case_name: str
    base_config: Dict[str, Any]
    raw_parameter_space: Dict[str, Any]
    parameter_space: Dict[str, list]
    search_config: Dict[str, Any]
    topologies: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SearchCandidate:
    trial: Any
    sg_index: int
    parameters: Dict[str, Any]
    combination_key: str


def search_config_for_task(task_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Use an ASR-native objective; keep LLM search configuration unchanged."""
    from xtask.benchmarks.asr.config import is_asr_task

    configured = task_config.get("assess_search")
    if configured is not None and not isinstance(configured, dict):
        raise TypeError("assess_search must be a mapping")
    raw = copy.deepcopy(configured or {})
    if is_asr_task(task_config) and not raw.get("objective"):
        raw["objective"] = {"metric": "audio_throughput", "direction": "maximize"}
    return normalize_assess_search_config(raw)


def normalize_assess_search_config(raw_config: Any) -> Dict[str, Any]:
    """Validate ``assess_search`` while preserving exhaustive compatibility."""
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise TypeError("assess_search must be a mapping")

    strategy = str(raw_config.get("strategy", "exhaustive")).strip().lower()
    if strategy not in SUPPORTED_SEARCH_STRATEGIES:
        raise ValueError(
            "Unsupported assess_search.strategy: %s; expected one of %s"
            % (strategy, sorted(SUPPORTED_SEARCH_STRATEGIES))
        )

    normalized = copy.deepcopy(raw_config)
    normalized["strategy"] = strategy
    if strategy == "exhaustive":
        # Exhaustive search also needs a default objective so its report can
        # select the best feasible combination when the YAML omits
        # assess_search.objective.  An explicitly configured metric remains
        # authoritative for backward compatibility.
        normalized["objective"] = _normalize_objective(
            raw_config.get("objective"), required=True
        )
        if "constraints" in raw_config:
            normalized["constraints"] = _normalize_constraints(
                raw_config.get("constraints")
            )
        return normalized

    max_trials = _positive_int(
        raw_config.get("max_trials", DEFAULT_MAX_TRIALS),
        "assess_search.max_trials",
    )
    startup_trials = _non_negative_int(
        raw_config.get("startup_trials", min(DEFAULT_STARTUP_TRIALS, max_trials)),
        "assess_search.startup_trials",
    )
    if startup_trials > max_trials:
        raise ValueError(
            "assess_search.startup_trials cannot exceed assess_search.max_trials"
        )

    seed = raw_config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("assess_search.seed must be an integer")

    objective = _normalize_objective(raw_config.get("objective"), required=True)
    constraints = _normalize_constraints(raw_config.get("constraints"))
    normalized.update(
        {
            "max_trials": max_trials,
            "startup_trials": startup_trials,
            "seed": seed,
            "objective": objective,
            "constraints": constraints,
        }
    )
    return normalized


def _normalize_objective(raw_objective: Any, *, required: bool) -> Optional[Dict[str, str]]:
    if raw_objective is None and not required:
        return None
    objective = raw_objective or {}
    if not isinstance(objective, dict):
        raise TypeError("assess_search.objective must be a mapping")
    metric = str(objective.get("metric", DEFAULT_OBJECTIVE_METRIC)).strip()
    if not metric:
        raise ValueError("assess_search.objective.metric cannot be empty")
    direction = str(objective.get("direction", "maximize")).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError(
            "assess_search.objective.direction must be maximize or minimize"
        )
    return {"metric": metric, "direction": direction}


def _normalize_constraints(raw_constraints: Any) -> Dict[str, Dict[str, float]]:
    constraints = raw_constraints or {}
    if not isinstance(constraints, dict):
        raise TypeError("assess_search.constraints must be a mapping")
    normalized_constraints = {}
    for constraint_metric, constraint in constraints.items():
        if not isinstance(constraint_metric, str) or not constraint_metric.strip():
            raise ValueError("assess_search constraint metric names cannot be empty")
        if not isinstance(constraint, dict):
            raise TypeError(
                f"Constraint for '{constraint_metric}' must be a mapping"
            )
        keys = set(constraint)
        if not keys or not keys.issubset({"min", "max"}):
            raise ValueError(
                f"Constraint for '{constraint_metric}' must define min and/or max"
            )
        bounds = {}
        for bound_name, bound_value in constraint.items():
            if isinstance(bound_value, bool) or not isinstance(
                bound_value, (int, float)
            ):
                raise TypeError(
                    f"Constraint {constraint_metric}.{bound_name} must be numeric"
                )
            bounds[bound_name] = float(bound_value)
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise ValueError(
                f"Constraint min exceeds max for '{constraint_metric}'"
            )
        normalized_constraints[constraint_metric.strip()] = bounds
    return normalized_constraints


def resolve_para_set_strategy(args) -> str:
    """Return the strategy shared by all cases in one para_set invocation."""
    cases = load_search_cases(args)
    strategies = {case.search_config["strategy"] for case in cases}
    if len(strategies) != 1:
        raise ValueError(
            "A para_set case group cannot mix exhaustive and TPE strategies"
        )
    return next(iter(strategies))


def load_search_cases(args) -> list[SearchCase]:
    case_names = _get_case_names(args)
    return [load_search_case(case_name) for case_name in case_names]


def load_search_case(case_name: str) -> SearchCase:
    basecase_file = find_case_file(case_name, str(Path.cwd() / "tests"))
    if basecase_file is None:
        raise FileNotFoundError(f"case file not found: {case_name}")
    with open(basecase_file, "r", encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    if not isinstance(base_config, dict):
        raise ValueError(f"case YAML is empty: {basecase_file}")

    from xtask.utils.resource_sync import expand_dependence_placeholders

    base_config = expand_dependence_placeholders(base_config)

    from xtask.core.service_group import validate_pd_deployment_authoring

    validate_pd_deployment_authoring(base_config)
    raw_parameter_space = base_config.get("assess_test_param", {})
    parameter_space = _normalize_assess_parameters(raw_parameter_space)
    if not parameter_space:
        raise ValueError(f"case {case_name} has no assess_test_param values")
    topologies = normalize_assess_topologies(base_config, parameter_space)
    search_config = search_config_for_task(base_config)
    return SearchCase(
        case_name=case_name,
        base_config=base_config,
        raw_parameter_space=copy.deepcopy(raw_parameter_space),
        parameter_space=parameter_space,
        search_config=search_config,
        topologies=topologies,
    )


def build_parameter_candidate(
    base_config: Dict[str, Any],
    parameters: Mapping[str, Any],
    generated_case_name: str,
    topologies: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Apply one selected combination using para_set's existing path rules."""
    config_copy = copy.deepcopy(base_config)
    topologies = (
        normalize_assess_topologies(base_config)
        if topologies is None
        else topologies
    )
    changed_paths = []
    changed_paths_by_param = {}
    topology_name = parameters.get(ASSESS_TOPOLOGY_PARAMETER)
    if topology_name is not None:
        changed_for_topology = apply_assess_topology(
            config_copy, topology_name, topologies
        )
        changed_paths.extend(changed_for_topology)
        changed_paths_by_param[ASSESS_TOPOLOGY_PARAMETER] = changed_for_topology
    for parameter_name, parameter_value in parameters.items():
        if parameter_name == ASSESS_TOPOLOGY_PARAMETER:
            continue
        changed_for_param = _update_config_by_path(
            config_copy,
            parameter_name,
            parameter_value,
            apply_derived=False,
        )
        changed_paths.extend(changed_for_param)
        changed_paths_by_param[parameter_name] = changed_for_param

    if "benchmark.perf.batch_size" in changed_paths:
        has_explicit_num_prompts = "benchmark.perf.num_prompts" in changed_paths
        if not has_explicit_num_prompts:
            batch_values = [
                parameters[name]
                for name, paths in changed_paths_by_param.items()
                if "benchmark.perf.batch_size" in paths
            ]
            if len(batch_values) != 1:
                raise ValueError(
                    "benchmark.perf.batch_size must be specified at most once"
                )
            _apply_batch_size_defaults(config_copy, batch_values[0])

    if topology_name is not None:
        from xtask.core.service_group import normalize_pd_deployment

        normalize_pd_deployment(copy.deepcopy(config_copy))

    config_copy["case_name"] = generated_case_name.lower()
    return config_copy


def write_parameter_candidate(
    search_case: SearchCase,
    candidate: SearchCandidate,
    output_root: Path,
) -> tuple[str, str]:
    generated_case_name = f"{search_case.case_name}_sg{candidate.sg_index}".lower()
    case_dir = output_root / generated_case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    task_file = case_dir / f"{generated_case_name}.yaml"
    candidate_config = build_parameter_candidate(
        search_case.base_config,
        candidate.parameters,
        generated_case_name,
        search_case.topologies,
    )
    with open(task_file, "w", encoding="utf-8") as stream:
        yaml.safe_dump(
            candidate_config,
            stream,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    metadata = {
        "current_combination": copy.deepcopy(candidate.parameters),
        "assess_test_param": copy.deepcopy(search_case.raw_parameter_space),
        "assess_search": {
            "strategy": "tpe",
            "trial_number": candidate.sg_index,
            "objective": copy.deepcopy(search_case.search_config["objective"]),
        },
    }
    resolved_topology = resolved_topology_metadata(
        candidate.parameters, search_case.topologies, candidate_config
    )
    if resolved_topology is not None:
        metadata["resolved_topology"] = resolved_topology
    with open(case_dir / "assess_test_param.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(
            metadata,
            stream,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    return str(task_file), str(case_dir)


class TpeSearchController:
    """Thin ask/tell wrapper around Optuna's TPE sampler."""

    def __init__(self, search_case: SearchCase, state_dir: Path):
        self.search_case = search_case
        self.config = search_case.search_config
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.optuna = _import_optuna()
        self.optuna.logging.set_verbosity(self.optuna.logging.WARNING)
        sampler = self.optuna.samplers.TPESampler(
            seed=self.config["seed"],
            n_startup_trials=self.config["startup_trials"],
        )
        storage_path = (self.state_dir / "search_state.db").resolve()
        metric_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(self.config["objective"]["metric"]),
        ).strip("_.-") or "objective"
        self.study = self.optuna.create_study(
            # Keep studies for different objective metrics separate.  This is
            # important when an old search_state.db contains avg-throughput
            # trials and the default objective has since changed.
            study_name=f"{search_case.case_name}_tpe_{metric_name}",
            storage=f"sqlite:///{storage_path}",
            sampler=sampler,
            direction=self.config["objective"]["direction"],
            load_if_exists=True,
        )
        self._prune_stale_running_trials()
        self._seen = {
            trial.user_attrs["combination_key"]
            for trial in self.study.trials
            if trial.user_attrs.get("executed")
            and trial.user_attrs.get("combination_key")
        }
        executed_indices = [
            int(trial.user_attrs["sg_index"])
            for trial in self.study.trials
            if trial.user_attrs.get("executed")
            and trial.user_attrs.get("sg_index") is not None
        ]
        self._next_sg_index = max(executed_indices, default=0) + 1
        self.total_combinations = math.prod(
            len(values) for values in search_case.parameter_space.values()
        )

    @property
    def executed_trials(self) -> int:
        return sum(
            bool(trial.user_attrs.get("executed")) for trial in self.study.trials
        )

    def ask(self) -> Optional[SearchCandidate]:
        trial_limit = min(self.config["max_trials"], self.total_combinations)
        if self.executed_trials >= trial_limit or len(self._seen) >= self.total_combinations:
            return None

        max_attempts = max(100, self.total_combinations * 2)
        for _ in range(max_attempts):
            trial = self.study.ask()
            parameters = {}
            for parameter_name, choices in self.search_case.parameter_space.items():
                choice_index = trial.suggest_categorical(
                    f"choice::{parameter_name}", list(range(len(choices)))
                )
                parameters[parameter_name] = copy.deepcopy(choices[choice_index])
            combination_key = _canonical_combination(parameters)
            if combination_key in self._seen:
                self.study.tell(
                    trial, state=self.optuna.trial.TrialState.PRUNED
                )
                continue

            candidate = SearchCandidate(
                trial=trial,
                sg_index=self._next_sg_index,
                parameters=parameters,
                combination_key=combination_key,
            )
            self._next_sg_index += 1
            trial.set_user_attr("combination_key", combination_key)
            trial.set_user_attr("sg_index", candidate.sg_index)
            return candidate

        logging.warning(
            "TPE could not find a new unique combination after %d attempts",
            max_attempts,
        )
        return None

    def tell_result(self, candidate: SearchCandidate, result: Dict[str, Any]) -> None:
        observation = build_search_observation(result, self.config)
        trial = candidate.trial
        trial.set_user_attr("executed", True)
        trial.set_user_attr("status", observation["status"])
        trial.set_user_attr("metrics", observation["metrics"])
        trial.set_user_attr("feasible", observation["feasible"])
        if observation["objective_value"] is not None:
            trial.set_user_attr("objective_value", observation["objective_value"])
        if observation.get("error"):
            trial.set_user_attr("error", observation["error"])
        self._seen.add(candidate.combination_key)

        if observation["score"] is None:
            self.study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(trial, observation["score"])
        self.write_reports()

    def tell_failure(self, candidate: SearchCandidate, error: str) -> None:
        trial = candidate.trial
        trial.set_user_attr("executed", True)
        trial.set_user_attr("status", "failed")
        trial.set_user_attr("feasible", False)
        trial.set_user_attr("error", error)
        self._seen.add(candidate.combination_key)
        self.study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
        self.write_reports()

    def write_reports(self) -> None:
        trials = list(self._report_trials())
        summary = {
            "strategy": "tpe",
            "case_name": self.search_case.case_name,
            "max_trials": self.config["max_trials"],
            "startup_trials": self.config["startup_trials"],
            "seed": self.config["seed"],
            "objective": copy.deepcopy(self.config["objective"]),
            "constraints": copy.deepcopy(self.config["constraints"]),
            "total_combinations": self.total_combinations,
            "executed_trials": self.executed_trials,
            "best_trial": _best_report_trial(
                trials, self.config["objective"]["direction"]
            ),
            "trials": trials,
        }
        with open(self.state_dir / "search_summary.yaml", "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                summary,
                stream,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        fieldnames = [
            "sg_index",
            "status",
            "feasible",
            "objective_value",
            "parameters",
            "metrics",
            "error",
        ]
        with open(
            self.state_dir / "search_trials.csv", "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for trial in trials:
                row = dict(trial)
                row["parameters"] = json.dumps(
                    row["parameters"], ensure_ascii=False, sort_keys=True
                )
                row["metrics"] = json.dumps(
                    row["metrics"], ensure_ascii=False, sort_keys=True
                )
                writer.writerow(row)

    def _report_trials(self) -> Iterable[Dict[str, Any]]:
        for trial in self.study.trials:
            if not trial.user_attrs.get("executed"):
                continue
            parameters = _decode_trial_parameters(
                trial.params, self.search_case.parameter_space
            )
            yield {
                "sg_index": int(trial.user_attrs.get("sg_index", 0)),
                "status": trial.user_attrs.get("status", "unknown"),
                "feasible": bool(trial.user_attrs.get("feasible", False)),
                "objective_value": trial.user_attrs.get("objective_value"),
                "parameters": parameters,
                "metrics": trial.user_attrs.get("metrics", {}),
                "error": trial.user_attrs.get("error", ""),
            }

    def _prune_stale_running_trials(self) -> None:
        state = self.optuna.trial.TrialState
        for trial in self.study.get_trials(deepcopy=False):
            if trial.state == state.RUNNING:
                self.study.tell(trial.number, state=state.PRUNED)


def run_tpe_parameter_search(args, openme_client, run_trial, output_root=None):
    """Execute TPE studies while preserving the scheduler's full lifecycle."""
    cases = load_search_cases(args)
    if any(case.search_config["strategy"] != "tpe" for case in cases):
        raise ValueError("run_tpe_parameter_search requires strategy: tpe")
    root = Path(output_root or init_task_path(args))
    root.mkdir(parents=True, exist_ok=True)
    all_results = []

    for search_case in cases:
        state_dir = root / "search" / search_case.case_name.lower()
        controller = TpeSearchController(search_case, state_dir)
        controller.write_reports()
        while True:
            candidate = controller.ask()
            if candidate is None:
                break
            task_file, case_output = write_parameter_candidate(
                search_case, candidate, root
            )
            logging.info(
                "TPE trial sg%d/%d for %s: %s",
                candidate.sg_index,
                min(
                    search_case.search_config["max_trials"],
                    controller.total_combinations,
                ),
                search_case.case_name,
                candidate.parameters,
            )
            try:
                result = run_trial(task_file, case_output, args)
            except Exception as exc:
                logging.error(
                    "TPE trial sg%d failed: %s",
                    candidate.sg_index,
                    exc,
                    exc_info=True,
                )
                controller.tell_failure(candidate, str(exc))
                continue

            all_results.append(result)
            openme_client.AddCaseResult(result, task_path=task_file)
            controller.tell_result(candidate, result)
        controller.write_reports()
    return all_results


def build_search_observation(
    result: Dict[str, Any], search_config: Dict[str, Any]
) -> Dict[str, Any]:
    from xtask.benchmarks.asr.config import is_asr_task
    from xtask.benchmarks.asr.reporting import search_metrics

    if is_asr_task(result.get("task_config") or {}):
        metrics = search_metrics(result)
        if result.get("success") is not True:
            return {"status": "failed", "metrics": metrics, "feasible": False,
                    "objective_value": None, "score": None,
                    "error": result.get("message") or "ASR trial failed completeness/quality/archive gates"}
    else:
        perf_result = result.get("benchmark_perf_result")
        summary = getattr(perf_result, "summary", None)
        metrics = _numeric_summary(summary) if summary is not None else {}
    if not metrics:
        return {
            "status": "failed",
            "metrics": {},
            "feasible": False,
            "objective_value": None,
            "score": None,
            "error": result.get("message", "missing benchmark performance summary"),
        }

    objective = search_config["objective"]
    objective_value = metrics.get(objective["metric"])
    if objective_value is None:
        return {
            "status": "failed",
            "metrics": metrics,
            "feasible": False,
            "objective_value": None,
            "score": None,
            "error": f"missing objective metric: {objective['metric']}",
        }

    feasible = True
    for metric, bounds in search_config.get("constraints", {}).items():
        value = metrics.get(metric)
        if value is None:
            feasible = False
            break
        if "min" in bounds and value < bounds["min"]:
            feasible = False
        if "max" in bounds and value > bounds["max"]:
            feasible = False

    score = objective_value
    if not feasible:
        penalty = 1.0e12
        score = (
            objective_value - penalty
            if objective["direction"] == "maximize"
            else objective_value + penalty
        )
    return {
        "status": "complete" if feasible else "constraint_failed",
        "metrics": metrics,
        "feasible": feasible,
        "objective_value": objective_value,
        "score": score,
        "error": "",
    }


def _get_case_names(args) -> list[str]:
    case_group = getattr(args, "case_group", None)
    if case_group:
        case_names = get_case_list_from_group(case_group)
    else:
        case_name = getattr(args, "case_name", None)
        case_names = [case_name] if case_name else []
    if not case_names:
        raise ValueError("para_set search requires a case_group or case_name")
    return case_names


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _import_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "TPE search requires Optuna. Install project search dependencies "
            "or run: pip install 'optuna>=4.0,<5.0'."
        ) from exc
    return optuna


def _canonical_combination(parameters: Mapping[str, Any]) -> str:
    return json.dumps(
        parameters,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )


def _decode_trial_parameters(
    trial_parameters: Mapping[str, Any], parameter_space: Mapping[str, list]
) -> Dict[str, Any]:
    decoded = {}
    for name, choices in parameter_space.items():
        index = trial_parameters.get(f"choice::{name}")
        if index is not None:
            decoded[name] = copy.deepcopy(choices[int(index)])
    return decoded


def _numeric_summary(summary: Any) -> Dict[str, float]:
    if isinstance(summary, dict):
        values = summary
    elif is_dataclass(summary):
        values = asdict(summary)
    else:
        values = vars(summary) if hasattr(summary, "__dict__") else {}
    return {
        key: float(value)
        for key, value in values.items()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    }


def _best_report_trial(trials: list[Dict[str, Any]], direction: str):
    eligible = [
        trial
        for trial in trials
        if trial["feasible"] and trial["objective_value"] is not None
    ]
    if not eligible:
        return None
    return copy.deepcopy(
        (max if direction == "maximize" else min)(
            eligible, key=lambda trial: trial["objective_value"]
        )
    )
