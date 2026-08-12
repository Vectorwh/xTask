"""Aggregate result reports for exhaustive and auto-assessment runs."""

from __future__ import annotations

import copy
import csv
import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from xtask.tasks.parameter_search import normalize_assess_search_config


REPORT_MODES = {"para_set", "auto_assess"}
_EXHAUSTIVE_CASE_RE = re.compile(r"^(?P<base>.+)_sg(?P<index>\d+)$")
_AUTO_CASE_RE = re.compile(r"^(?P<base>.+)_bs-(?P<batch>.+)$")


class AssessmentReporter:
    """Incrementally write one report set per base case."""

    def __init__(
        self,
        args,
        tasks: Mapping[str, str],
        *,
        strategy: str,
        output_root: Optional[Path] = None,
    ):
        self.args = args
        self.mode = getattr(args, "assess_test_type", "none")
        self.strategy = strategy
        self.output_root = Path(output_root or _resolve_output_root(args))
        self._states: Dict[str, Dict[str, Any]] = {}
        self._planned_trials = _count_planned_trials(tasks, self.mode)

    def record_result(
        self,
        task_file: str,
        output_path: str,
        result: Dict[str, Any],
        *,
        search_state: Optional[Mapping[str, Any]] = None,
        action: Optional[str] = None,
    ) -> None:
        identity = _resolve_trial_identity(task_file, result, self.mode)
        config = result.get("task_config") or _load_yaml_mapping(task_file)
        parameters = _load_trial_parameters(output_path, config, self.mode)
        metrics = _extract_metrics(result)
        state = self._get_state(identity["base_case_name"], config)

        if self.mode == "auto_assess":
            trial = self._build_auto_trial(
                identity,
                parameters,
                metrics,
                result,
                output_path,
                search_state,
                action,
            )
        else:
            trial = self._build_exhaustive_trial(
                identity,
                parameters,
                metrics,
                result,
                output_path,
                state,
            )

        state["trials"].append(trial)
        if self.mode == "auto_assess" and search_state:
            reason = search_state.get("stop_reason")
            if reason:
                state["stop_reason"] = str(reason)
        self._write_state(state)

    def record_failure(
        self,
        task_file: str,
        output_path: str,
        error: str,
        *,
        search_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        identity = _resolve_trial_identity(task_file, {}, self.mode)
        config = _load_yaml_mapping(task_file)
        state = self._get_state(identity["base_case_name"], config)
        parameters = _load_trial_parameters(output_path, config, self.mode)
        trial_number = (
            identity.get("sg_index")
            if self.mode == "para_set"
            else len(state["trials"]) + 1
        )
        state["trials"].append(
            {
                "trial_number": trial_number or len(state["trials"]) + 1,
                "case_name": identity["case_name"],
                "status": "failed",
                "case_success": False,
                "feasible": False,
                "objective_value": None,
                "parameters": parameters,
                "metrics": {},
                "search_bounds": _search_bounds(search_state),
                "output_path": str(output_path),
                "error": str(error),
            }
        )
        self._write_state(state)

    def finalize(self, stop_reason: Optional[str] = None) -> None:
        for state in self._states.values():
            state["status"] = "completed"
            if self.mode == "auto_assess":
                state["stop_reason"] = (
                    state.get("stop_reason") or stop_reason or "queue_exhausted"
                )
            self._write_state(state)

    def _get_state(
        self, base_case_name: str, config: Mapping[str, Any]
    ) -> Dict[str, Any]:
        existing = self._states.get(base_case_name)
        if existing is not None:
            return existing

        if self.mode == "para_set":
            search_config = normalize_assess_search_config(
                config.get("assess_search")
            )
            objective = copy.deepcopy(search_config.get("objective"))
            constraints = copy.deepcopy(search_config.get("constraints") or {})
            state = {
                "strategy": self.strategy,
                "case_name": base_case_name,
                "status": "running",
                "objective": objective,
                "constraints": constraints,
                "parameter_space": copy.deepcopy(
                    config.get("assess_test_param") or {}
                ),
                "planned_trials": self._planned_trials.get(base_case_name),
                "trials": [],
            }
        else:
            target_ttft = float(getattr(self.args, "target_ttft", 600000))
            target_tpot = float(getattr(self.args, "target_tpot", 100))
            state = {
                "strategy": self.strategy,
                "case_name": base_case_name,
                "status": "running",
                "objective": {
                    "metric": "benchmark.perf.batch_size",
                    "direction": "maximize",
                },
                "constraints": {
                    "ttft_p90": {"max": target_ttft},
                    "tpot_p90": {"max": target_tpot},
                },
                "targets": {
                    "ttft_p90": target_ttft,
                    "tpot_p90": target_tpot,
                },
                "stop_reason": None,
                "trials": [],
            }

        self._states[base_case_name] = state
        return state

    def _build_exhaustive_trial(
        self,
        identity: Mapping[str, Any],
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        result: Mapping[str, Any],
        output_path: str,
        state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        status, feasible, objective_value, error = _evaluate_metrics(
            metrics,
            state.get("objective"),
            state.get("constraints") or {},
            result.get("message"),
        )
        return {
            "trial_number": identity.get("sg_index") or len(state["trials"]) + 1,
            "case_name": identity["case_name"],
            "status": status,
            "case_success": bool(result.get("success", False)),
            "feasible": feasible,
            "objective_value": objective_value,
            "parameters": parameters,
            "metrics": metrics,
            "search_bounds": None,
            "output_path": str(result.get("output_file_path") or output_path),
            "error": error,
        }

    def _build_auto_trial(
        self,
        identity: Mapping[str, Any],
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        result: Mapping[str, Any],
        output_path: str,
        search_state: Optional[Mapping[str, Any]],
        action: Optional[str],
    ) -> Dict[str, Any]:
        batch_size = parameters.get("benchmark.perf.batch_size")
        if result.get("scheduler_error"):
            status = "failed"
            feasible = False
            error = str(result.get("message") or "scheduler error")
        else:
            constraints = {
                "ttft_p90": {"max": float(getattr(self.args, "target_ttft", 600000))},
                "tpot_p90": {"max": float(getattr(self.args, "target_tpot", 100))},
            }
            status, feasible, _unused, error = _evaluate_metrics(
                metrics, None, constraints, result.get("message")
            )
        return {
            "trial_number": len(self._states[identity["base_case_name"]]["trials"]) + 1,
            "case_name": identity["case_name"],
            "status": status,
            "case_success": bool(result.get("success", False)),
            "feasible": feasible,
            "objective_value": batch_size,
            "parameters": parameters,
            "metrics": metrics,
            "search_bounds": _search_bounds(search_state),
            "action": action,
            "output_path": str(result.get("output_file_path") or output_path),
            "error": error,
        }

    def _write_state(self, state: Mapping[str, Any]) -> None:
        trials = copy.deepcopy(state["trials"])
        objective = state.get("objective")
        best_trial = _select_best_trial(
            trials,
            objective.get("direction") if objective else None,
        )
        summary = {
            key: copy.deepcopy(value)
            for key, value in state.items()
            if key != "trials"
        }
        summary.update(
            {
                "executed_trials": len(trials),
                "completed_trials": sum(
                    trial["status"] != "failed" for trial in trials
                ),
                "failed_trials": sum(
                    trial["status"] == "failed" for trial in trials
                ),
                "constraint_failed_trials": sum(
                    trial["status"] == "constraint_failed" for trial in trials
                ),
                "feasible_trials": sum(
                    bool(trial.get("feasible")) for trial in trials
                ),
                "best_trial": best_trial,
                "trials": trials,
            }
        )
        if self.mode == "para_set":
            summary["total_combinations"] = state.get("planned_trials")
        if self.mode == "auto_assess":
            summary["selected_batch_size"] = (
                best_trial.get("objective_value") if best_trial else None
            )

        report_dir = self.output_root / "search" / _safe_name(state["case_name"])
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "search_summary.yaml", "w", encoding="utf-8") as stream:
            yaml.safe_dump(summary, stream, allow_unicode=True, sort_keys=False)
        _write_trials_csv(report_dir / "search_trials.csv", trials)


def create_assessment_reporter(
    args,
    tasks: Mapping[str, str],
    *,
    para_strategy: Optional[str] = None,
    output_root: Optional[Path] = None,
) -> Optional[AssessmentReporter]:
    mode = getattr(args, "assess_test_type", "none")
    if mode not in REPORT_MODES:
        return None
    if mode == "para_set" and para_strategy != "exhaustive":
        return None
    strategy = "exhaustive" if mode == "para_set" else "auto_assess"
    return AssessmentReporter(
        args,
        tasks,
        strategy=strategy,
        output_root=output_root,
    )


def _resolve_output_root(args) -> Path:
    for name in ("template", "case_name", "case_group"):
        value = getattr(args, name, None)
        if value:
            return Path.cwd() / "outputs" / str(value).lower()
    model_name = getattr(args, "model_name", None)
    if model_name:
        return Path.cwd() / "outputs" / str(model_name).lower()
    return Path.cwd() / "outputs" / "assessment"


def _count_planned_trials(
    tasks: Mapping[str, str], mode: str
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for task_file in tasks:
        identity = _resolve_trial_identity(task_file, {}, mode)
        base = identity["base_case_name"]
        counts[base] = counts.get(base, 0) + 1
    return counts


def _resolve_trial_identity(
    task_file: str, result: Mapping[str, Any], mode: str
) -> Dict[str, Any]:
    config = result.get("task_config") or _load_yaml_mapping(task_file)
    case_name = str(config.get("case_name") or Path(task_file).stem).lower()
    if mode == "para_set":
        match = _EXHAUSTIVE_CASE_RE.match(case_name)
        return {
            "case_name": case_name,
            "base_case_name": match.group("base") if match else case_name,
            "sg_index": int(match.group("index")) if match else None,
        }
    match = _AUTO_CASE_RE.match(case_name)
    return {
        "case_name": case_name,
        "base_case_name": match.group("base") if match else case_name,
        "batch_size": match.group("batch") if match else None,
    }


def _load_trial_parameters(
    output_path: str, config: Mapping[str, Any], mode: str
) -> Dict[str, Any]:
    if mode == "para_set":
        metadata = _load_yaml_mapping(
            str(Path(output_path) / "assess_test_param.yaml")
        )
        combination = metadata.get("current_combination")
        return copy.deepcopy(combination) if isinstance(combination, dict) else {}

    batch_size = (
        (config.get("benchmark") or {}).get("perf") or {}
    ).get("batch_size")
    return {"benchmark.perf.batch_size": batch_size}


def _load_yaml_mapping(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        return value if isinstance(value, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _extract_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    perf_result = result.get("benchmark_perf_result")
    summary = getattr(perf_result, "summary", None)
    if summary is None:
        return {}
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


def _evaluate_metrics(
    metrics: Mapping[str, float],
    objective: Optional[Mapping[str, str]],
    constraints: Mapping[str, Mapping[str, float]],
    result_message: Any,
) -> tuple[str, bool, Optional[float], str]:
    if not metrics:
        return (
            "failed",
            False,
            None,
            str(result_message or "missing benchmark performance summary"),
        )

    objective_value = None
    if objective:
        objective_value = metrics.get(objective["metric"])
        if objective_value is None:
            return (
                "failed",
                False,
                None,
                f"missing objective metric: {objective['metric']}",
            )

    violations = []
    for metric, bounds in constraints.items():
        value = metrics.get(metric)
        if value is None:
            violations.append(f"missing {metric}")
            continue
        if "min" in bounds and value < bounds["min"]:
            violations.append(f"{metric}={value} < {bounds['min']}")
        if "max" in bounds and value > bounds["max"]:
            violations.append(f"{metric}={value} > {bounds['max']}")
    if violations:
        return "constraint_failed", False, objective_value, "; ".join(violations)
    return "complete", True, objective_value, ""


def _select_best_trial(
    trials: list[Dict[str, Any]], direction: Optional[str]
) -> Optional[Dict[str, Any]]:
    if direction not in {"maximize", "minimize"}:
        return None
    eligible = [
        trial
        for trial in trials
        if trial.get("feasible") and trial.get("objective_value") is not None
    ]
    if not eligible:
        return None
    selector = max if direction == "maximize" else min
    return copy.deepcopy(selector(eligible, key=lambda item: item["objective_value"]))


def _search_bounds(search_state: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not search_state:
        return None
    values = {
        name: copy.deepcopy(search_state.get(name))
        for name in ("lower", "curr", "upper")
        if name in search_state
    }
    return values or None


def _write_trials_csv(path: Path, trials: list[Dict[str, Any]]) -> None:
    fieldnames = [
        "trial_number",
        "case_name",
        "status",
        "case_success",
        "feasible",
        "objective_value",
        "batch_size",
        "output_tps_total",
        "ttft_p90",
        "tpot_p90",
        "parameters",
        "metrics",
        "search_bounds",
        "action",
        "output_path",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            metrics = trial.get("metrics") or {}
            parameters = trial.get("parameters") or {}
            writer.writerow(
                {
                    "trial_number": trial.get("trial_number"),
                    "case_name": trial.get("case_name"),
                    "status": trial.get("status"),
                    "case_success": trial.get("case_success"),
                    "feasible": trial.get("feasible"),
                    "objective_value": trial.get("objective_value"),
                    "batch_size": parameters.get("benchmark.perf.batch_size"),
                    "output_tps_total": metrics.get(
                        "output_tps_total"
                    ),
                    "ttft_p90": metrics.get("ttft_p90"),
                    "tpot_p90": metrics.get("tpot_p90"),
                    "parameters": _json_value(parameters),
                    "metrics": _json_value(metrics),
                    "search_bounds": _json_value(trial.get("search_bounds")),
                    "action": trial.get("action"),
                    "output_path": trial.get("output_path"),
                    "error": trial.get("error", ""),
                }
            )


def _json_value(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr)


def _safe_name(value: Any) -> str:
    name = str(value).replace(os.sep, "_").replace(" ", "_")
    return name or "assessment"
