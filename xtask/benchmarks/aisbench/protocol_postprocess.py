"""A/B comparison and CI gate post-processing for protocol tests.

Compares a baseline run (no constrained decoding) against a constrained run
(json_schema / guided_regex / guided_grammar) on the same question set to
detect semantic degradation: "constrained decoding makes the model dumber".

CI gate rules (default):
    1. Protocol_Pass_Rate == 100%   (zero tolerance)
    2. Syntax_Valid_Rate == 100%    (zero tolerance)
    3. Acc_Baseline - Acc_Constrained <= degradation_threshold (default 1.5%)

All functions are pure and unit-testable without ais_bench.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DEGRADATION_THRESHOLD = 1.5  # percentage points
DEFAULT_PROTOCOL_TOLERANCE = 0.0     # zero tolerance: must be 100%
DEFAULT_SYNTAX_TOLERANCE = 0.0       # zero tolerance: must be 100%


def compare_ab_results(
    baseline_result: Dict[str, Any],
    constrained_result: Dict[str, Any],
    degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
) -> Dict[str, Any]:
    """Compare baseline vs constrained protocol test results.

    Args:
        baseline_result: score_predictions() output from the baseline run
            (same questions, no response_format / guided_* constraints).
        constrained_result: score_predictions() output from the constrained
            run (same questions with constraints applied).
        degradation_threshold: max allowed accuracy drop in percentage
            points (e.g. 1.5 means Acc_Baseline - Acc_Constrained <= 1.5%).

    Returns:
        {
            "baseline_accuracy": float,
            "constrained_accuracy": float,
            "accuracy_drop": float,
            "degradation_within_threshold": bool,
            "constrained_protocol_rate": float,
            "constrained_syntax_rate": float,
        }
    """
    baseline_acc = baseline_result.get("Semantic_Accuracy", 0.0)
    constrained_acc = constrained_result.get("Semantic_Accuracy", 0.0)
    accuracy_drop = round(baseline_acc - constrained_acc, 2)

    return {
        "baseline_accuracy": baseline_acc,
        "constrained_accuracy": constrained_acc,
        "accuracy_drop": accuracy_drop,
        "degradation_within_threshold": accuracy_drop <= degradation_threshold,
        "constrained_protocol_rate": constrained_result.get("Protocol_Pass_Rate", 0.0),
        "constrained_syntax_rate": constrained_result.get("Syntax_Valid_Rate", 0.0),
    }


def evaluate_ci_gate(
    comparison: Dict[str, Any],
    protocol_tolerance: float = DEFAULT_PROTOCOL_TOLERANCE,
    syntax_tolerance: float = DEFAULT_SYNTAX_TOLERANCE,
    degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate CI gate rules against an A/B comparison.

    Args:
        comparison: output of compare_ab_results().
        protocol_tolerance: allowed deviation from 100% for Protocol_Pass_Rate
            (default 0.0 = must be exactly 100%).
        syntax_tolerance: allowed deviation from 100% for Syntax_Valid_Rate.
        degradation_threshold: fallback threshold if not in comparison.

    Returns:
        {
            "passed": bool,
            "checks": [
                {"name": str, "passed": bool, "detail": str},
                ...
            ],
            "summary": str,
        }
    """
    checks = []

    # Rule 1: Protocol pass rate (zero tolerance by default)
    protocol_rate = comparison.get("constrained_protocol_rate", 0.0)
    protocol_ok = protocol_rate >= (100.0 - protocol_tolerance)
    checks.append({
        "name": "Protocol_Pass_Rate",
        "passed": protocol_ok,
        "detail": f"{protocol_rate}% (required >= {100.0 - protocol_tolerance}%)",
    })

    # Rule 2: Syntax valid rate (zero tolerance by default)
    syntax_rate = comparison.get("constrained_syntax_rate", 0.0)
    syntax_ok = syntax_rate >= (100.0 - syntax_tolerance)
    checks.append({
        "name": "Syntax_Valid_Rate",
        "passed": syntax_ok,
        "detail": f"{syntax_rate}% (required >= {100.0 - syntax_tolerance}%)",
    })

    # Rule 3: Semantic degradation within threshold
    degradation_ok = comparison.get(
        "degradation_within_threshold",
        comparison.get("accuracy_drop", 100.0) <= degradation_threshold,
    )
    accuracy_drop = comparison.get("accuracy_drop", "N/A")
    checks.append({
        "name": "Semantic_Degradation",
        "passed": degradation_ok,
        "detail": f"accuracy_drop={accuracy_drop}% (threshold={degradation_threshold}%)",
    })

    all_passed = all(c["passed"] for c in checks)
    failed_names = [c["name"] for c in checks if not c["passed"]]
    if all_passed:
        summary = "CI gate PASSED: all protocol & semantic checks passed"
    else:
        summary = f"CI gate FAILED: {', '.join(failed_names)}"

    return {"passed": all_passed, "checks": checks, "summary": summary}


def load_protocol_result(result_path: str) -> Optional[Dict[str, Any]]:
    """Load a protocol test result JSON file.

    The result file is expected to contain the output of
    score_predictions() (Protocol_Pass_Rate, Syntax_Valid_Rate,
    Semantic_Accuracy, total, details).
    """
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load protocol result from %s: %s", result_path, exc)
        return None


def generate_ab_report(
    baseline_result: Dict[str, Any],
    constrained_result: Dict[str, Any],
    degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
) -> Dict[str, Any]:
    """Generate a complete A/B comparison report with CI gate evaluation.

    This is the top-level entry point for post-processing after both the
    baseline and constrained protocol tests have completed.
    """
    comparison = compare_ab_results(
        baseline_result, constrained_result, degradation_threshold
    )
    gate = evaluate_ci_gate(comparison, degradation_threshold=degradation_threshold)

    report = {
        "comparison": comparison,
        "ci_gate": gate,
        "baseline": {
            "Protocol_Pass_Rate": baseline_result.get("Protocol_Pass_Rate", 0.0),
            "Syntax_Valid_Rate": baseline_result.get("Syntax_Valid_Rate", 0.0),
            "Semantic_Accuracy": baseline_result.get("Semantic_Accuracy", 0.0),
            "total": baseline_result.get("total", 0),
        },
        "constrained": {
            "Protocol_Pass_Rate": constrained_result.get("Protocol_Pass_Rate", 0.0),
            "Syntax_Valid_Rate": constrained_result.get("Syntax_Valid_Rate", 0.0),
            "Semantic_Accuracy": constrained_result.get("Semantic_Accuracy", 0.0),
            "total": constrained_result.get("total", 0),
        },
        "degradation_threshold": degradation_threshold,
    }
    logger.info("A/B report: %s", gate["summary"])
    return report


class ProtocolABPostProcessor:
    """Post-processor for protocol A/B comparison, integrated into xTask's
    result processing pipeline.

    Usage:
        processor = ProtocolABPostProcessor(degradation_threshold=1.5)
        report = processor.process(baseline_path, constrained_path)
        if not report["ci_gate"]["passed"]:
            raise RuntimeError(report["ci_gate"]["summary"])
    """

    def __init__(self, degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD):
        self.degradation_threshold = degradation_threshold

    def process(
        self,
        baseline_result_path: str,
        constrained_result_path: str,
    ) -> Dict[str, Any]:
        """Load both result files and generate the A/B comparison report."""
        baseline = load_protocol_result(baseline_result_path)
        constrained = load_protocol_result(constrained_result_path)

        if baseline is None:
            raise FileNotFoundError(
                f"Baseline protocol result not found or invalid: {baseline_result_path}"
            )
        if constrained is None:
            raise FileNotFoundError(
                f"Constrained protocol result not found or invalid: {constrained_result_path}"
            )

        return generate_ab_report(baseline, constrained, self.degradation_threshold)