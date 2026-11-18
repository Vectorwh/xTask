"""Three-layer gate evaluator for protocol consistency & constrained decoding.

Layer 1 - Protocol compliance:
    finish_reason match, reasoning tag leakage check, reasoning_content
    presence when required.

Layer 2 - Syntax / Schema validity:
    JSON Schema validation (jsonschema), regex fullmatch, or tool_calls
    structural verification, depending on the ground_truth type.

Layer 3 - Semantic fidelity:
    Exact answer match to catch "format correct but answer wrong" cases.
    Only evaluated when layer 2 passes.

The pure scoring functions are unit-testable without ais_bench installed;
:class:`ProtocolAndConstrainedEvaluator` wraps them for AISBench integration.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False
    logger.warning(
        "jsonschema not installed; JSON Schema validation falls back to "
        "basic json.loads check. Install with: pip install jsonschema"
    )


# ---------------------------------------------------------------------------
# Content extraction (reasoning may be mixed into content)
# ---------------------------------------------------------------------------

def _extract_answer_content(content: str) -> str:
    """Best-effort extraction of the answer portion from content that may
    include reasoning text before the actual answer.

    When the serving framework does not separate ``reasoning_content`` from
    ``content``, the model's thinking text and the final answer both land in
    ``content``.  Syntax validation (JSON Schema, regex fullmatch) must
    operate on the answer portion, not the reasoning preamble.

    Strategies (in order):
        1. Markdown code block: return the last ```...``` block content.
        2. Content itself is valid JSON: return as-is.
        3. Trailing JSON: content ends with a parseable JSON object/array
           (common pattern: ``reasoning...\\n\\n\\n{...}``).
        4. Otherwise: return content unchanged.
    """
    if not content or not content.strip():
        return content

    # Strategy 1: markdown code block (```json ... ``` or ``` ... ```)
    fence_matches = re.findall(
        r"```(?:[a-zA-Z0-9_+-]*)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL
    )
    if fence_matches:
        return fence_matches[-1].strip()

    stripped = content.strip()

    # Strategy 2: content itself is valid JSON
    try:
        json.loads(stripped)
        return stripped
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 3: trailing JSON object/array
    if stripped.endswith("}") or stripped.endswith("]"):
        for i in range(len(stripped) - 1, -1, -1):
            if stripped[i] not in "{[":
                continue
            candidate = stripped[i:]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    return content


# ---------------------------------------------------------------------------
# Layer 1: Protocol compliance
# ---------------------------------------------------------------------------

def check_protocol(
    message: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Check protocol compliance for a single prediction.

    Args:
        message: full message dict with content, reasoning_content,
            tool_calls, finish_reason, role.
        ground_truth: reference dict from JSONL ground_truth field.

    Returns:
        {"pass": bool, "errors": [str, ...]}
    """
    from xtask.benchmarks.aisbench.protocol_api import check_reasoning_leak

    errors: List[str] = []
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    finish_reason = message.get("finish_reason") or ""

    expected_finish = ground_truth.get("expected_finish_reason", "stop")
    if finish_reason != expected_finish:
        errors.append(
            f"finish_reason mismatch: got {finish_reason!r}, expected {expected_finish!r}"
        )

    if check_reasoning_leak(content):
        errors.append("reasoning tag leaked into content")

    if ground_truth.get("require_reasoning") and not reasoning.strip():
        errors.append("reasoning_content required but empty")

    if ground_truth.get("forbid_reasoning") and reasoning.strip():
        errors.append("reasoning_content present but forbidden for this case")

    return {"pass": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# Layer 2: Syntax / Schema validity
# ---------------------------------------------------------------------------

def _validate_json_schema(content: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate content as JSON against a JSON Schema."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        return {"pass": False, "errors": [f"invalid JSON: {exc}"], "parsed": None}

    if _HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=parsed, schema=schema)
        except jsonschema.ValidationError as exc:
            return {"pass": False, "errors": [f"schema violation: {exc.message}"], "parsed": parsed}
        except jsonschema.SchemaError as exc:
            return {"pass": False, "errors": [f"invalid schema: {exc}"], "parsed": parsed}
    else:
        # Fallback: only check that JSON parsed successfully.
        logger.debug("jsonschema unavailable; skipping deep schema validation")

    return {"pass": True, "errors": [], "parsed": parsed}


def _validate_regex(content: str, pattern: str) -> Dict[str, Any]:
    """Validate content against a regex pattern.

    First tries ``re.fullmatch`` (strict: content must be exactly the
    constrained string).  If that fails, falls back to ``re.search``
    (lenient: finds the constrained string anywhere in content, useful
    when reasoning text surrounds the actual answer).
    """
    try:
        matched = bool(re.fullmatch(pattern, content))
    except re.error as exc:
        return {"pass": False, "errors": [f"invalid regex pattern: {exc}"], "parsed": None}
    if matched:
        return {"pass": True, "errors": [], "parsed": content}
    # Lenient fallback: search for the pattern anywhere in the content.
    try:
        search_match = re.search(pattern, content)
    except re.error:
        search_match = None
    if search_match:
        return {"pass": True, "errors": [], "parsed": search_match.group(0)}
    preview = content[:80] + "..." if len(content) > 80 else content
    return {"pass": False, "errors": [f"regex fullmatch failed, content: {preview}"], "parsed": None}


def _validate_tool_calls(
    actual_calls: List[Dict[str, Any]],
    expected_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify tool_calls structure against expected calls.

    Expected calls can specify:
        - function name (must match)
        - function arguments subset (expected args must be present in actual)
        - count of calls (if expected has N entries, actual must have N)
    """
    errors: List[str] = []

    if not isinstance(actual_calls, list) or len(actual_calls) != len(expected_calls):
        return {
            "pass": False,
            "errors": [
                f"tool_calls count mismatch: got {len(actual_calls) if isinstance(actual_calls, list) else 'non-list'}, "
                f"expected {len(expected_calls)}"
            ],
            "parsed": actual_calls,
        }

    for i, (actual, expected) in enumerate(zip(actual_calls, expected_calls)):
        actual_fn = (actual.get("function") or {})
        expected_fn = (expected.get("function") or {})

        if expected_fn.get("name") and actual_fn.get("name") != expected_fn["name"]:
            errors.append(
                f"tool_calls[{i}] function name mismatch: "
                f"got {actual_fn.get('name')!r}, expected {expected_fn['name']!r}"
            )

        expected_args = expected_fn.get("arguments") or {}
        if expected_args:
            try:
                actual_args = json.loads(actual_fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                errors.append(f"tool_calls[{i}] arguments is not valid JSON")
                actual_args = {}
            for key, val in expected_args.items():
                if key not in actual_args:
                    errors.append(f"tool_calls[{i}] missing argument: {key}")
                elif actual_args[key] != val:
                    errors.append(
                        f"tool_calls[{i}] argument {key} mismatch: "
                        f"got {actual_args[key]!r}, expected {val!r}"
                    )

    return {"pass": len(errors) == 0, "errors": errors, "parsed": actual_calls}


def check_syntax(
    message: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Check syntax/schema validity for a single prediction.

    Dispatches to _validate_json_schema / _validate_regex / _validate_tool_calls
    based on which key has a **non-null** value in ground_truth.

    .. note::
        HuggingFace ``Dataset.from_list()`` unions all dict keys across rows
        and fills missing values with ``None``.  Therefore ``ground_truth``
        may contain keys like ``schema``/``regex`` with ``None`` values for
        cases that don't use them.  We must check for non-null values, not
        mere key presence, to dispatch correctly.

    Returns:
        {"pass": bool, "errors": [str, ...], "parsed": Any}
    """
    raw_content = message.get("content") or ""
    # Extract the answer portion when reasoning is mixed into content.
    content = _extract_answer_content(raw_content)

    if ground_truth.get("schema"):
        return _validate_json_schema(content, ground_truth["schema"])
    if ground_truth.get("regex"):
        return _validate_regex(content, ground_truth["regex"])
    if ground_truth.get("tool_calls"):
        return _validate_tool_calls(message.get("tool_calls") or [], ground_truth["tool_calls"])
    if ground_truth.get("grammar"):
        # EBNF grammar validation requires the serving engine's parser;
        # here we only verify the content is non-empty (best-effort).
        if not content.strip():
            return {"pass": False, "errors": ["empty content for grammar-constrained case"], "parsed": None}
        return {"pass": True, "errors": [], "parsed": content}

    # No syntax constraint specified - pass by default.
    return {"pass": True, "errors": [], "parsed": content}


# ---------------------------------------------------------------------------
# Layer 3: Semantic fidelity
# ---------------------------------------------------------------------------

def check_semantic(
    parsed: Any,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Check semantic fidelity: the parsed answer must match the expected.

    Supports:
        - ground_truth["answer"]: exact match against parsed["answer"] (dict)
          or the full parsed value (non-dict).
        - ground_truth["answer_regex"]: regex search against stringified answer.

    .. note::
        Uses ``.get() is not None`` / truthy checks instead of ``in`` because
        HuggingFace ``Dataset.from_list()`` unions all dict keys across rows,
        filling missing values with ``None``.
    """
    errors: List[str] = []

    if ground_truth.get("answer") is not None:
        expected = ground_truth["answer"]
        if isinstance(parsed, dict):
            actual = parsed.get("answer")
        else:
            actual = parsed
        if actual != expected:
            errors.append(f"answer mismatch: got {actual!r}, expected {expected!r}")

    if ground_truth.get("answer_regex"):
        pattern = ground_truth["answer_regex"]
        answer_str = str(parsed.get("answer", "") if isinstance(parsed, dict) else parsed)
        if not re.search(pattern, answer_str):
            errors.append(f"answer_regex mismatch: {answer_str!r} does not match {pattern!r}")

    return {"pass": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# Aggregate scoring (pure, unit-testable)
# ---------------------------------------------------------------------------

def score_predictions(
    predictions: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score a batch of predictions against references through three gates.

    Args:
        predictions: list of full message dicts (from model output).
        references: list of ground_truth dicts (from JSONL dataset).

    Returns:
        {
            "Protocol_Pass_Rate": float,   # percentage
            "Syntax_Valid_Rate": float,    # percentage
            "Semantic_Accuracy": float,    # percentage
            "total": int,
            "details": [per-case detail dict, ...],
        }
    """
    total = len(predictions)
    if total == 0:
        return {
            "Protocol_Pass_Rate": 0.0,
            "Syntax_Valid_Rate": 0.0,
            "Semantic_Accuracy": 0.0,
            "total": 0,
            "details": [],
        }

    protocol_pass = 0
    syntax_pass = 0
    semantic_pass = 0
    details: List[Dict[str, Any]] = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        case_id = ref.get("id", f"case_{i}")

        # Layer 1: protocol
        proto_result = check_protocol(pred, ref)
        if proto_result["pass"]:
            protocol_pass += 1

        # Layer 2: syntax (independent of layer 1 result)
        syntax_result = check_syntax(pred, ref)
        if syntax_result["pass"]:
            syntax_pass += 1

        # Layer 3: semantic (only when syntax passes and parsed data exists)
        sem_result = {"pass": False, "errors": ["skipped: syntax check failed"]}
        if syntax_result["pass"] and syntax_result.get("parsed") is not None:
            sem_result = check_semantic(syntax_result["parsed"], ref)
            if sem_result["pass"]:
                semantic_pass += 1

        details.append({
            "id": case_id,
            "protocol": proto_result,
            "syntax": syntax_result,
            "semantic": sem_result,
        })

    return {
        "Protocol_Pass_Rate": round(protocol_pass / total * 100, 2),
        "Syntax_Valid_Rate": round(syntax_pass / total * 100, 2),
        "Semantic_Accuracy": round(semantic_pass / total * 100, 2),
        "total": total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# AISBench evaluator wrapper
# ---------------------------------------------------------------------------

class ProtocolAndConstrainedEvaluator:
    """AISBench-compatible evaluator implementing the three-layer gate.

    Registered in dataset config's eval_cfg as::
        evaluator=dict(type=ProtocolAndConstrainedEvaluator)

    AISBench's openicl_eval.py ``_score()`` builds a ``preds`` dict from the
    prediction file, then **filters it by the ``score()`` method's signature
    parameters** before calling::

        icl_evaluator.evaluate(k, n, copy.deepcopy(test_set), **preds)

    where ``k``/``n`` are integers (num_return_sequences, NOT dataset abbr),
    and the filtered ``preds`` contains only keys matching ``score()`` params:
    ``predictions`` (list of prediction strings) and ``references`` (list of
    ``test_set[output_column]`` values).

    Predictions are expected to be full message dicts serialized as JSON
    strings (content + reasoning_content + tool_calls + finish_reason) as
    produced by ProtocolAwareAPIChat._parse_response.
    """

    def evaluate(self, k, n, test_set, predictions=None, references=None,
                 **kwargs) -> Dict[str, Any]:
        """AISBench evaluation entry point.

        Args:
            k: number of return sequences per sample (int, not dataset abbr).
            n: number of samples for pass@k (int).
            test_set: HuggingFace Dataset of test samples (kept for
                compatibility; references are extracted from it only as
                fallback).
            predictions: list of prediction strings (JSON-serialized message
                dicts), passed as kwarg matching ``score()`` param name.
            references: list of ground_truth values from
                ``test_set[output_column]``, passed as kwarg matching
                ``score()`` param name.
            **kwargs: catches any extra filtered keys (ignored).

        Returns:
            Dict of metric name -> value.
        """
        # AISBench filters preds by score() signature, so predictions and
        # references arrive as keyword arguments.  Fall back to kwargs and
        # then to test_set extraction for robustness.
        if predictions is None:
            predictions = kwargs.get('predictions', [])
        if not isinstance(predictions, list):
            predictions = list(predictions) if predictions is not None else []

        if references is None:
            references = kwargs.get('references')
        if references is None:
            # Fallback: extract references from test_set.
            references = []
            for item in test_set:
                if isinstance(item, dict):
                    gold = item.get("gold")
                    if gold is None:
                        gold = item.get("ground_truth")
                    if gold is None:
                        gold = item.get("answer")
                    references.append(gold if gold is not None else {})
                else:
                    references.append({"answer": item})

        return self.score(predictions, references)

    def score(self, predictions, references) -> Dict[str, Any]:
        """Score predictions against references.

        Accepts both list-of-dict (direct call) and AISBench's internal
        prediction/reference formats (graceful degradation: if a prediction
        is a plain string, wrap it into a minimal message dict).
        """
        normalized_preds: List[Dict[str, Any]] = []
        for pred in predictions:
            if isinstance(pred, dict) and "content" in pred:
                normalized_preds.append(pred)
            elif isinstance(pred, str):
                # ProtocolAwareAPIChat serializes the full message as JSON.
                # Try to parse it; fall back to treating it as plain content.
                try:
                    parsed = json.loads(pred)
                    if isinstance(parsed, dict) and "content" in parsed:
                        normalized_preds.append(parsed)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                normalized_preds.append({
                    "content": pred,
                    "reasoning_content": "",
                    "tool_calls": [],
                    "finish_reason": "stop",
                    "role": "assistant",
                })
            else:
                normalized_preds.append({
                    "content": str(pred),
                    "reasoning_content": "",
                    "tool_calls": [],
                    "finish_reason": "",
                    "role": "assistant",
                })

        normalized_refs: List[Dict[str, Any]] = []
        for ref in references:
            if isinstance(ref, dict):
                normalized_refs.append(ref)
            else:
                normalized_refs.append({"answer": ref})

        result = score_predictions(normalized_preds, normalized_refs)
        logger.info(
            "ProtocolAndConstrainedEvaluator: protocol=%.2f%% syntax=%.2f%% semantic=%.2f%% (n=%d)",
            result["Protocol_Pass_Rate"],
            result["Syntax_Valid_Rate"],
            result["Semantic_Accuracy"],
            result["total"],
        )
        return result