"""Unit tests for AISBench protocol consistency & constrained decoding feature.

Covers:
- protocol_api.py: payload injection, message extraction, SSE assembly,
  reasoning leak detection, stream consistency verification.
- protocol_constrained_evaluator.py: three-layer gate (protocol / syntax /
  semantic), aggregate scoring, AISBench evaluator wrapper.
- protocol_postprocess.py: A/B comparison, CI gate evaluation.
- config_parser.py: global switch is_protocol_test_enabled.
- generate_aisbench_script.py: protocol section generation (switch on/off).
"""

import asyncio
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

# Ensure generate_aisbench_script is loaded as a package attribute so that
# unittest.mock.patch can resolve its path.
import xtask.benchmarks.aisbench.generate_aisbench_script  # noqa: F401

from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
    _HAS_JSONSCHEMA,
)


class TestInjectProtocolFields(unittest.TestCase):
    """Tests for protocol_api.inject_protocol_fields."""

    def test_inject_tools(self):
        from xtask.benchmarks.aisbench.protocol_api import inject_protocol_fields

        payload = {"model": "test", "messages": []}
        extra = {"tools": [{"type": "function", "function": {"name": "f"}}]}
        result = inject_protocol_fields(payload, extra)
        self.assertIn("tools", result)
        self.assertEqual(result["tools"], extra["tools"])

    def test_inject_response_format(self):
        from xtask.benchmarks.aisbench.protocol_api import inject_protocol_fields

        payload = {"model": "test", "messages": []}
        rf = {"type": "json_schema", "json_schema": {"name": "test", "schema": {}}}
        result = inject_protocol_fields(payload, {"response_format": rf})
        self.assertEqual(result["response_format"], rf)

    def test_inject_guided_regex(self):
        from xtask.benchmarks.aisbench.protocol_api import inject_protocol_fields

        payload = {"model": "test", "messages": []}
        result = inject_protocol_fields(payload, {"guided_regex": "[0-9]+"})
        self.assertEqual(result["guided_regex"], "[0-9]+")

    def test_ignores_unknown_fields(self):
        from xtask.benchmarks.aisbench.protocol_api import inject_protocol_fields

        payload = {"model": "test", "messages": []}
        result = inject_protocol_fields(payload, {"unknown_field": "x", "tools": None})
        self.assertNotIn("unknown_field", result)
        self.assertNotIn("tools", result)

    def test_none_extra_params_noop(self):
        from xtask.benchmarks.aisbench.protocol_api import inject_protocol_fields

        payload = {"model": "test"}
        result = inject_protocol_fields(payload, None)
        self.assertEqual(result, payload)


class TestExtractFullMessage(unittest.TestCase):
    """Tests for protocol_api.extract_full_message."""

    def test_full_message(self):
        from xtask.benchmarks.aisbench.protocol_api import extract_full_message

        resp = {
            "choices": [{
                "message": {
                    "content": "answer",
                    "reasoning_content": "thinking",
                    "tool_calls": [{"id": "1", "function": {"name": "f"}}],
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }]
        }
        msg = extract_full_message(resp)
        self.assertEqual(msg["content"], "answer")
        self.assertEqual(msg["reasoning_content"], "thinking")
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["finish_reason"], "stop")

    def test_missing_fields_default(self):
        from xtask.benchmarks.aisbench.protocol_api import extract_full_message

        msg = extract_full_message({})
        self.assertEqual(msg["content"], "")
        self.assertEqual(msg["reasoning_content"], "")
        self.assertEqual(msg["tool_calls"], [])
        self.assertEqual(msg["finish_reason"], "")


class TestAssembleSSEDeltas(unittest.TestCase):
    """Tests for protocol_api.assemble_sse_deltas."""

    def test_content_assembly(self):
        from xtask.benchmarks.aisbench.protocol_api import assemble_sse_deltas

        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        result = assemble_sse_deltas(chunks)
        self.assertEqual(result["content"], "Hello world")
        self.assertEqual(result["finish_reason"], "stop")

    def test_reasoning_assembly(self):
        from xtask.benchmarks.aisbench.protocol_api import assemble_sse_deltas

        chunks = [
            {"choices": [{"delta": {"reasoning_content": "step1"}, "finish_reason": None}]},
            {"choices": [{"delta": {"reasoning_content": "step2"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
        ]
        result = assemble_sse_deltas(chunks)
        self.assertEqual(result["reasoning_content"], "step1step2")
        self.assertEqual(result["content"], "answer")

    def test_tool_calls_assembly(self):
        from xtask.benchmarks.aisbench.protocol_api import assemble_sse_deltas

        chunks = [
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}
            }]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": "{\"ci"}
            }]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": "ty\": \"BJ\"}"}
            }]}, "finish_reason": "tool_calls"}]},
        ]
        result = assemble_sse_deltas(chunks)
        self.assertEqual(len(result["tool_calls"]), 1)
        tc = result["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(tc["function"]["arguments"], '{"city": "BJ"}')
        self.assertEqual(result["finish_reason"], "tool_calls")

    def test_empty_chunks(self):
        from xtask.benchmarks.aisbench.protocol_api import assemble_sse_deltas

        result = assemble_sse_deltas([])
        self.assertEqual(result["content"], "")
        self.assertEqual(result["tool_calls"], [])


class TestReasoningLeakAndStreamVerify(unittest.TestCase):
    """Tests for check_reasoning_leak and verify_stream_consistency."""

    def test_leak_detected(self):
        from xtask.benchmarks.aisbench.protocol_api import check_reasoning_leak

        self.assertTrue(check_reasoning_leak("answer </think> leaked"))
        self.assertTrue(check_reasoning_leak("has </think> inside"))

    def test_no_leak(self):
        from xtask.benchmarks.aisbench.protocol_api import check_reasoning_leak

        self.assertFalse(check_reasoning_leak("clean answer"))
        self.assertFalse(check_reasoning_leak(""))

    def test_stream_consistency_pass(self):
        from xtask.benchmarks.aisbench.protocol_api import verify_stream_consistency

        stream = {"content": "a", "reasoning_content": "r", "tool_calls": [], "finish_reason": "stop"}
        non_stream = {"content": "a", "reasoning_content": "r", "tool_calls": [], "finish_reason": "stop"}
        result = verify_stream_consistency(stream, non_stream)
        self.assertTrue(result["consistent"])
        self.assertEqual(result["errors"], [])

    def test_stream_consistency_content_mismatch(self):
        from xtask.benchmarks.aisbench.protocol_api import verify_stream_consistency

        stream = {"content": "a", "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"}
        non_stream = {"content": "b", "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"}
        result = verify_stream_consistency(stream, non_stream)
        self.assertFalse(result["consistent"])
        self.assertTrue(any("content mismatch" in e for e in result["errors"]))

    def test_stream_leak_detected(self):
        from xtask.benchmarks.aisbench.protocol_api import (
            REASONING_LEAK_TAGS,
            verify_stream_consistency,
        )

        # Construct content that actually contains a reasoning leak tag.
        leak_tag = REASONING_LEAK_TAGS[0]
        leaky_content = f"answer {leak_tag} rest"
        stream = {"content": leaky_content, "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"}
        non_stream = {"content": leaky_content, "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"}
        result = verify_stream_consistency(stream, non_stream)
        self.assertFalse(result["consistent"])
        self.assertTrue(any("leaked" in e for e in result["errors"]))


class TestCheckProtocol(unittest.TestCase):
    """Tests for evaluator check_protocol (layer 1)."""

    def test_pass(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_protocol

        msg = {"content": "answer", "reasoning_content": "thinking", "finish_reason": "stop"}
        gt = {"expected_finish_reason": "stop", "require_reasoning": True}
        result = check_protocol(msg, gt)
        self.assertTrue(result["pass"])

    def test_finish_reason_mismatch(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_protocol

        msg = {"content": "answer", "reasoning_content": "", "finish_reason": "length"}
        gt = {"expected_finish_reason": "stop"}
        result = check_protocol(msg, gt)
        self.assertFalse(result["pass"])

    def test_missing_reasoning(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_protocol

        msg = {"content": "answer", "reasoning_content": "", "finish_reason": "stop"}
        gt = {"require_reasoning": True}
        result = check_protocol(msg, gt)
        self.assertFalse(result["pass"])

    def test_forbidden_reasoning(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_protocol

        msg = {"content": "answer", "reasoning_content": "thoughts", "finish_reason": "stop"}
        gt = {"forbid_reasoning": True}
        result = check_protocol(msg, gt)
        self.assertFalse(result["pass"])


class TestCheckSyntax(unittest.TestCase):
    """Tests for evaluator check_syntax (layer 2)."""

    def test_json_schema_pass(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"content": '{"steps": "s", "answer": 42}'}
        gt = {"schema": {"type": "object", "properties": {"steps": {"type": "string"}, "answer": {"type": "number"}}, "required": ["steps", "answer"]}}
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])
        self.assertEqual(result["parsed"]["answer"], 42)

    def test_json_schema_fail_invalid_json(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"content": "not json"}
        gt = {"schema": {"type": "object"}}
        result = check_syntax(msg, gt)
        self.assertFalse(result["pass"])

    @unittest.skipUnless(_HAS_JSONSCHEMA, "jsonschema not installed on host")
    def test_json_schema_fail_type_violation(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"content": '{"answer": "not_a_number"}'}
        gt = {"schema": {"type": "object", "properties": {"answer": {"type": "number"}}}}
        result = check_syntax(msg, gt)
        self.assertFalse(result["pass"])

    def test_regex_pass(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"content": "13812345678"}
        gt = {"regex": "1[3-9][0-9]{9}"}
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])

    def test_regex_fail(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"content": "12345678901"}
        gt = {"regex": "1[3-9][0-9]{9}"}
        result = check_syntax(msg, gt)
        self.assertFalse(result["pass"])

    def test_tool_calls_pass(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "BJ"}'}}]}
        gt = {"tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "BJ"}}}]}
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])

    def test_tool_calls_name_mismatch(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"tool_calls": [{"function": {"name": "wrong_fn", "arguments": '{}'}}]}
        gt = {"tool_calls": [{"function": {"name": "get_weather", "arguments": {}}}]}
        result = check_syntax(msg, gt)
        self.assertFalse(result["pass"])

    def test_tool_calls_count_mismatch(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"tool_calls": [{"function": {"name": "f1", "arguments": '{}'}}]}
        gt = {"tool_calls": [{"function": {"name": "f1"}}, {"function": {"name": "f2"}}]}
        result = check_syntax(msg, gt)
        self.assertFalse(result["pass"])


class TestCheckSemantic(unittest.TestCase):
    """Tests for evaluator check_semantic (layer 3)."""

    def test_answer_match(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_semantic

        result = check_semantic({"answer": 6000}, {"answer": 6000})
        self.assertTrue(result["pass"])

    def test_answer_mismatch(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_semantic

        result = check_semantic({"answer": 5000}, {"answer": 6000})
        self.assertFalse(result["pass"])

    def test_answer_regex(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_semantic

        result = check_semantic({"answer": "391"}, {"answer_regex": "391"})
        self.assertTrue(result["pass"])


class TestExtractAnswerContent(unittest.TestCase):
    """Tests for _extract_answer_content (reasoning/content separation)."""

    def test_plain_content_unchanged(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _extract_answer_content

        self.assertEqual(_extract_answer_content("13812345678"), "13812345678")

    def test_markdown_json_code_block(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _extract_answer_content

        content = '一些推理文本\n\n```json\n{"answer": 42}\n```'
        result = _extract_answer_content(content)
        self.assertEqual(json.loads(result), {"answer": 42})

    def test_bare_code_block(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _extract_answer_content

        content = '推理\n```\n1 + 2\n```'
        result = _extract_answer_content(content)
        self.assertEqual(result, "1 + 2")

    def test_trailing_json_after_reasoning(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _extract_answer_content

        content = '推理文本\n\n\n{"answer": 126}'
        result = _extract_answer_content(content)
        self.assertEqual(json.loads(result), {"answer": 126})

    def test_content_is_json(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _extract_answer_content

        result = _extract_answer_content('{"answer": 42}')
        self.assertEqual(json.loads(result), {"answer": 42})

    def test_no_extraction_possible(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _extract_answer_content

        content = "just plain reasoning text with no answer"
        self.assertEqual(_extract_answer_content(content), content)

    def test_empty_content(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _extract_answer_content

        self.assertEqual(_extract_answer_content(""), "")


class TestNullKeyGroundTruth(unittest.TestCase):
    """Tests for HuggingFace Dataset union behavior.

    Dataset.from_list() unions all dict keys across rows and fills missing
    values with None.  ground_truth dicts therefore contain ALL keys
    (schema, regex, grammar, answer, ...) with None for unused ones.
    check_syntax / check_semantic must dispatch on non-null values.
    """

    _ALL_KEYS = [
        "expected_finish_reason", "schema", "answer", "regex",
        "grammar", "answer_regex", "require_reasoning",
    ]

    def _union_gt(self, **active):
        gt = {k: None for k in self._ALL_KEYS}
        gt.update(active)
        return gt

    def test_syntax_regex_with_null_schema(self):
        """Regex case must dispatch to _validate_regex even when schema=None."""
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"content": "13812345678"}
        gt = self._union_gt(regex="1[3-9][0-9]{9}")
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])
        self.assertEqual(result["parsed"], "13812345678")

    def test_syntax_schema_with_null_regex(self):
        """Schema case must dispatch to _validate_json_schema even when regex=None."""
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"content": '{"answer": 42}'}
        gt = self._union_gt(
            schema={"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}
        )
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])

    def test_syntax_grammar_with_all_nulls(self):
        """Grammar case must dispatch to grammar check."""
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        msg = {"content": "1 + 2"}
        gt = self._union_gt(grammar=True)
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])

    def test_semantic_answer_null_skipped(self):
        """answer=None must NOT trigger answer comparison."""
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_semantic

        result = check_semantic({"answer": 42}, self._union_gt())
        self.assertTrue(result["pass"])

    def test_semantic_answer_present(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_semantic

        result = check_semantic({"answer": 42}, self._union_gt(answer=42))
        self.assertTrue(result["pass"])

    def test_semantic_answer_regex_null_skipped(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_semantic

        result = check_semantic("some text", self._union_gt())
        self.assertTrue(result["pass"])

    def test_score_predictions_with_union_references(self):
        """End-to-end: unioned references (all keys, None for unused) must
        produce correct metrics — not 0% across the board."""
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import score_predictions

        preds = [
            {"content": "13812345678", "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"},
            {"content": '{"answer": 42}', "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"},
        ]
        refs = [
            self._union_gt(regex="1[3-9][0-9]{9}"),
            self._union_gt(
                schema={"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]},
                answer=42,
            ),
        ]
        result = score_predictions(preds, refs)
        self.assertEqual(result["Syntax_Valid_Rate"], 100.0)
        self.assertEqual(result["Semantic_Accuracy"], 100.0)  # regex case: no answer check → pass; schema case: answer matches


class TestStripNulls(unittest.TestCase):
    """Tests for _strip_nulls (HuggingFace Dataset nested union cleanup)."""

    def test_strips_top_level_nulls(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _strip_nulls

        gt = {"schema": None, "regex": "1[3-9]", "answer": None}
        result = _strip_nulls(gt)
        self.assertEqual(result, {"regex": "1[3-9]"})

    def test_strips_nested_schema_properties(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _strip_nulls

        # Simulates id 2's schema after HuggingFace Dataset union:
        # properties gains null entries from other cases' schemas.
        polluted = {
            "type": "object",
            "properties": {
                "steps": None, "answer": None, "persons": None,
                "name": {"type": "string"}, "age": {"type": "integer"},
                "address": {"type": "object", "properties": {
                    "street": {"type": "string"}, "city": {"type": "string"},
                    "zipcode": {"type": "string"},
                }},
                "category": None, "reason": None,
            },
            "required": None,
        }
        result = _strip_nulls(polluted)
        self.assertNotIn("steps", result["properties"])
        self.assertNotIn("required", result)
        self.assertIn("name", result["properties"])
        self.assertIn("address", result["properties"])

    def test_preserves_non_null_values(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _strip_nulls

        gt = {"answer": 0, "regex": "", "grammar": True, "require_reasoning": True}
        result = _strip_nulls(gt)
        # 0, "", True are NOT None — must be preserved.
        self.assertEqual(result, gt)

    def test_handles_lists(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _strip_nulls

        data = {"required": ["steps", "answer"], "items": [None, "a", {"x": None, "y": 1}]}
        result = _strip_nulls(data)
        self.assertEqual(result["required"], ["steps", "answer"])
        self.assertEqual(result["items"], ["a", {"y": 1}])

    def test_score_with_polluted_schema(self):
        """End-to-end: a schema polluted by HF Dataset union (null properties)
        must not cause SchemaError — _strip_nulls cleans it before validation."""
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
            ProtocolAndConstrainedEvaluator,
        )

        # id 2 scenario: model outputs correct JSON matching intended schema,
        # but reference schema is polluted with null property entries.
        full_msg = {
            "content": '{"name": "张三", "age": 30, "address": {"street": "中山路123号", "city": "北京市", "zipcode": "100000"}}',
            "reasoning_content": "", "tool_calls": [], "finish_reason": "stop",
        }
        polluted_ref = {
            "expected_finish_reason": "stop",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": None, "answer": None, "persons": None, "locations": None,
                    "name": {"type": "string"}, "age": {"type": "integer"},
                    "address": {"type": "object", "properties": {
                        "street": {"type": "string"}, "city": {"type": "string"},
                        "zipcode": {"type": "string"},
                    }},
                    "category": None, "reason": None,
                },
                "required": None,
            },
            "answer": None, "regex": None, "grammar": None,
            "answer_regex": None, "require_reasoning": None,
        }
        evaluator = ProtocolAndConstrainedEvaluator()
        result = evaluator.score([json.dumps(full_msg)], [polluted_ref])
        self.assertEqual(result["Syntax_Valid_Rate"], 100.0,
                         f"Polluted schema should be cleaned: {result['details'][0]['syntax']['errors']}")
        self.assertEqual(result["Protocol_Pass_Rate"], 100.0)


class TestRegexSearchFallback(unittest.TestCase):
    """Tests for _validate_regex search fallback."""

    def test_fullmatch_strict(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _validate_regex

        result = _validate_regex("13812345678", "1[3-9][0-9]{9}")
        self.assertTrue(result["pass"])
        self.assertEqual(result["parsed"], "13812345678")

    def test_search_fallback_with_surrounding_text(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _validate_regex

        content = "好的，这是一个号码：13812345678，请查收。"
        result = _validate_regex(content, "1[3-9][0-9]{9}")
        self.assertTrue(result["pass"])
        self.assertEqual(result["parsed"], "13812345678")

    def test_no_match_fails(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import _validate_regex

        result = _validate_regex("no phone here", "1[3-9][0-9]{9}")
        self.assertFalse(result["pass"])


class TestCheckSyntaxContentExtraction(unittest.TestCase):
    """Tests for check_syntax with reasoning mixed into content."""

    def test_json_in_code_block(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        content = '让我想想...\n\n```json\n{"answer": 42}\n```'
        msg = {"content": content}
        gt = {"schema": {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}}
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])
        self.assertEqual(result["parsed"]["answer"], 42)

    def test_trailing_json_after_reasoning(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        content = '推理过程...\n\n\n{"answer": 126}'
        msg = {"content": content}
        gt = {"schema": {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}}
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])
        self.assertEqual(result["parsed"]["answer"], 126)

    def test_regex_with_reasoning_prefix(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import check_syntax

        content = "好的，让我生成一个号码：13812345678"
        msg = {"content": content}
        gt = {"regex": "1[3-9][0-9]{9}"}
        result = check_syntax(msg, gt)
        self.assertTrue(result["pass"])


class TestScorePredictions(unittest.TestCase):
    """Tests for aggregate score_predictions."""

    def test_mixed_results(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import score_predictions

        preds = [
            {"content": '{"steps": "s", "answer": 6000}', "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"},
            {"content": "invalid json", "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"},
        ]
        refs = [
            {"id": "c1", "schema": {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}, "answer": 6000},
            {"id": "c2", "schema": {"type": "object"}},
        ]
        result = score_predictions(preds, refs)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["Protocol_Pass_Rate"], 100.0)
        self.assertEqual(result["Syntax_Valid_Rate"], 50.0)
        self.assertEqual(result["Semantic_Accuracy"], 50.0)

    def test_empty(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import score_predictions

        result = score_predictions([], [])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["Protocol_Pass_Rate"], 0.0)


class TestEvaluatorWrapper(unittest.TestCase):
    """Tests for ProtocolAndConstrainedEvaluator.score with JSON string preds."""

    def test_json_string_prediction(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
            ProtocolAndConstrainedEvaluator,
        )

        full_msg = {"content": '{"answer": 42}', "reasoning_content": "", "tool_calls": [], "finish_reason": "stop"}
        evaluator = ProtocolAndConstrainedEvaluator()
        result = evaluator.score(
            [json.dumps(full_msg)],
            [{"schema": {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}, "answer": 42}],
        )
        self.assertEqual(result["Semantic_Accuracy"], 100.0)

    def test_plain_string_prediction(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
            ProtocolAndConstrainedEvaluator,
        )

        evaluator = ProtocolAndConstrainedEvaluator()
        result = evaluator.score(["plain text answer"], [{"answer_regex": "plain"}])
        self.assertEqual(result["Protocol_Pass_Rate"], 100.0)

    def test_json_string_reference(self):
        """score() must deserialize JSON string references from
        ProtocolConstrainedDataset (which serializes ground_truth to avoid
        pyarrow ArrowInvalid on heterogeneous struct types)."""
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
            ProtocolAndConstrainedEvaluator,
        )

        full_msg = {"content": '{"answer": 42}', "reasoning_content": "",
                    "tool_calls": [], "finish_reason": "stop"}
        gt_str = json.dumps({
            "schema": {"type": "object",
                       "properties": {"answer": {"type": "number"}},
                       "required": ["answer"]},
            "answer": 42,
        })
        evaluator = ProtocolAndConstrainedEvaluator()
        result = evaluator.score([json.dumps(full_msg)], [gt_str])
        self.assertEqual(result["Semantic_Accuracy"], 100.0)
        self.assertEqual(result["Syntax_Valid_Rate"], 100.0)


class TestEvaluatorEvaluateMethod(unittest.TestCase):
    """Tests for ProtocolAndConstrainedEvaluator.evaluate (AISBench entry point).

    AISBench's openicl_eval.py ``_score()`` filters the preds dict by the
    ``score()`` method's signature parameters, then calls:
        evaluator.evaluate(k, n, test_set, predictions=pred_strs, references=refs)
    where k/n are integers (num_return_sequences), predictions is a list of
    prediction strings, and references is test_set[output_column].
    """

    def _full_msg(self, content, finish_reason="stop"):
        return json.dumps({
            "content": content,
            "reasoning_content": "",
            "tool_calls": [],
            "finish_reason": finish_reason,
        })

    def test_evaluate_with_predictions_and_references_kwargs(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
            ProtocolAndConstrainedEvaluator,
        )

        pred_str = self._full_msg('{"answer": 42}')
        references = [{"schema": {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}, "answer": 42}]
        evaluator = ProtocolAndConstrainedEvaluator()
        result = evaluator.evaluate(1, 1, test_set=[], predictions=[pred_str], references=references)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["Protocol_Pass_Rate"], 100.0)
        self.assertEqual(result["Semantic_Accuracy"], 100.0)

    def test_evaluate_fallback_references_from_test_set(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
            ProtocolAndConstrainedEvaluator,
        )

        pred_str = self._full_msg("plain answer")
        test_set = [{"ground_truth": {"answer_regex": "plain"}}]
        evaluator = ProtocolAndConstrainedEvaluator()
        result = evaluator.evaluate(1, 1, test_set=test_set, predictions=[pred_str])
        self.assertEqual(result["Protocol_Pass_Rate"], 100.0)

    def test_evaluate_multiple_samples(self):
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
            ProtocolAndConstrainedEvaluator,
        )

        preds = [
            self._full_msg('{"answer": 6000}'),
            self._full_msg("invalid json"),
        ]
        references = [
            {"id": "c1", "schema": {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}, "answer": 6000},
            {"id": "c2", "schema": {"type": "object"}},
        ]
        evaluator = ProtocolAndConstrainedEvaluator()
        result = evaluator.evaluate(1, 2, test_set=[], predictions=preds, references=references)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["Syntax_Valid_Rate"], 50.0)

    def test_evaluate_simulates_aisbench_score_signature_filter(self):
        """Simulate the full AISBench flow: filter preds by score() signature."""
        from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
            ProtocolAndConstrainedEvaluator,
        )
        from inspect import signature

        evaluator = ProtocolAndConstrainedEvaluator()

        # Simulate what openicl_eval.py does: build preds dict then filter
        # by score() signature parameters.
        pred_strs = [self._full_msg('{"answer": 7}')]
        refs = [{"schema": {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}, "answer": 7}]
        preds = {
            "predictions": pred_strs,
            "references": refs,
            "test_set": [{"ground_truth": refs[0]}],
            "origin_prompt": [None],
        }
        filtered = {k: preds[k] for k in signature(evaluator.score).parameters}
        result = evaluator.evaluate(1, 1, test_set=preds["test_set"], **filtered)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["Semantic_Accuracy"], 100.0)


class TestProtocolPostprocess(unittest.TestCase):
    """Tests for A/B comparison and CI gate."""

    def test_compare_no_degradation(self):
        from xtask.benchmarks.aisbench.protocol_postprocess import compare_ab_results

        baseline = {"Semantic_Accuracy": 95.0}
        constrained = {"Semantic_Accuracy": 94.0, "Protocol_Pass_Rate": 100.0, "Syntax_Valid_Rate": 100.0}
        result = compare_ab_results(baseline, constrained, degradation_threshold=1.5)
        self.assertEqual(result["accuracy_drop"], 1.0)
        self.assertTrue(result["degradation_within_threshold"])

    def test_compare_degradation_exceeded(self):
        from xtask.benchmarks.aisbench.protocol_postprocess import compare_ab_results

        baseline = {"Semantic_Accuracy": 95.0}
        constrained = {"Semantic_Accuracy": 90.0, "Protocol_Pass_Rate": 100.0, "Syntax_Valid_Rate": 100.0}
        result = compare_ab_results(baseline, constrained, degradation_threshold=1.5)
        self.assertEqual(result["accuracy_drop"], 5.0)
        self.assertFalse(result["degradation_within_threshold"])

    def test_ci_gate_all_pass(self):
        from xtask.benchmarks.aisbench.protocol_postprocess import evaluate_ci_gate

        comparison = {
            "constrained_protocol_rate": 100.0,
            "constrained_syntax_rate": 100.0,
            "accuracy_drop": 1.0,
            "degradation_within_threshold": True,
        }
        gate = evaluate_ci_gate(comparison)
        self.assertTrue(gate["passed"])

    def test_ci_gate_protocol_fail(self):
        from xtask.benchmarks.aisbench.protocol_postprocess import evaluate_ci_gate

        comparison = {
            "constrained_protocol_rate": 95.0,
            "constrained_syntax_rate": 100.0,
            "accuracy_drop": 0.5,
            "degradation_within_threshold": True,
        }
        gate = evaluate_ci_gate(comparison)
        self.assertFalse(gate["passed"])
        self.assertTrue(any(c["name"] == "Protocol_Pass_Rate" and not c["passed"] for c in gate["checks"]))

    def test_ci_gate_degradation_fail(self):
        from xtask.benchmarks.aisbench.protocol_postprocess import evaluate_ci_gate

        comparison = {
            "constrained_protocol_rate": 100.0,
            "constrained_syntax_rate": 100.0,
            "accuracy_drop": 3.0,
            "degradation_within_threshold": False,
        }
        gate = evaluate_ci_gate(comparison)
        self.assertFalse(gate["passed"])

    def test_generate_ab_report(self):
        from xtask.benchmarks.aisbench.protocol_postprocess import generate_ab_report

        baseline = {"Semantic_Accuracy": 95.0, "Protocol_Pass_Rate": 100.0, "Syntax_Valid_Rate": 100.0, "total": 100}
        constrained = {"Semantic_Accuracy": 94.0, "Protocol_Pass_Rate": 100.0, "Syntax_Valid_Rate": 100.0, "total": 100}
        report = generate_ab_report(baseline, constrained, degradation_threshold=1.5)
        self.assertTrue(report["ci_gate"]["passed"])
        self.assertEqual(report["comparison"]["accuracy_drop"], 1.0)

    def test_post_processor_with_files(self):
        from xtask.benchmarks.aisbench.protocol_postprocess import ProtocolABPostProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = os.path.join(tmpdir, "baseline.json")
            constrained_path = os.path.join(tmpdir, "constrained.json")
            with open(baseline_path, "w") as f:
                json.dump({"Semantic_Accuracy": 95.0, "Protocol_Pass_Rate": 100.0, "Syntax_Valid_Rate": 100.0, "total": 10}, f)
            with open(constrained_path, "w") as f:
                json.dump({"Semantic_Accuracy": 94.0, "Protocol_Pass_Rate": 100.0, "Syntax_Valid_Rate": 100.0, "total": 10}, f)

            processor = ProtocolABPostProcessor(degradation_threshold=1.5)
            report = processor.process(baseline_path, constrained_path)
            self.assertTrue(report["ci_gate"]["passed"])


class TestGlobalSwitch(unittest.TestCase):
    """Tests for is_protocol_test_enabled global switch."""

    @patch("xtask.utils.config_parser.get_xtask_config_data")
    def test_switch_on(self, mock_config):
        from xtask.utils.config_parser import is_protocol_test_enabled

        mock_config.return_value = {"ENV_CONFIG": {"enable_protocol_test": True}}
        self.assertTrue(is_protocol_test_enabled())

    @patch("xtask.utils.config_parser.get_xtask_config_data")
    def test_switch_off(self, mock_config):
        from xtask.utils.config_parser import is_protocol_test_enabled

        mock_config.return_value = {"ENV_CONFIG": {"enable_protocol_test": False}}
        self.assertFalse(is_protocol_test_enabled())

    @patch("xtask.utils.config_parser.get_xtask_config_data")
    def test_switch_default_off(self, mock_config):
        from xtask.utils.config_parser import is_protocol_test_enabled

        mock_config.return_value = {"ENV_CONFIG": {}}
        self.assertFalse(is_protocol_test_enabled())


class TestGenerateScriptProtocolSection(unittest.TestCase):
    """Tests for protocol section in generate_aisbench_script."""

    def _make_task_config(self, metrics="protocol"):
        return {
            "case_name": "test_protocol",
            "host_addr": "127.0.0.1",
            "port": 8000,
            "served_model_name": "test-model",
            "model_path": "/data/model",
            "dataset_name": "custom",
            "dataset_path": "/data/test.jsonl",
            "metrics": metrics,
            "benchmark": {
                "benchmark_type": "aisbench",
                "protocol": {
                    "api_type": "vllm_api_general_chat",
                    "dataset_conf": "protocol_constrained_gen",
                    "dataset_path": "/data/constrained.jsonl",
                    "stream_verify": False,
                    "batch_size": 4,
                    "output_len": 2048,
                },
            },
        }

    @patch("xtask.benchmarks.aisbench.generate_aisbench_script.is_protocol_test_enabled", return_value=False)
    def test_switch_off_skips_protocol(self, _mock_switch):
        from xtask.benchmarks.aisbench.generate_aisbench_script import generate_aisbench_script

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aisbench_script(self._make_task_config(), tmpdir, "test.sh")
            with open(os.path.join(tmpdir, "test.sh"), "r") as f:
                content = f.read()
            self.assertIn("全局开关", content)
            self.assertIn("跳过协议一致性测试", content)
            self.assertNotIn("ais_bench --config-dir", content)

    @patch("xtask.benchmarks.aisbench.generate_aisbench_script.is_protocol_test_enabled", return_value=True)
    @patch("xtask.benchmarks.aisbench.generate_aisbench_script.create_protocol_model_config")
    @patch("xtask.benchmarks.aisbench.generate_aisbench_script.create_dataset_config_override")
    def test_switch_on_generates_protocol_cmd(
        self, _mock_dataset, _mock_model, _mock_switch
    ):
        from xtask.benchmarks.aisbench.generate_aisbench_script import generate_aisbench_script

        _mock_model.return_value = "xtask_protocol_vllm_api_general_chat"
        _mock_dataset.return_value = "xtask_protocol_constrained_gen"

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aisbench_script(self._make_task_config(), tmpdir, "test.sh")
            with open(os.path.join(tmpdir, "test.sh"), "r") as f:
                content = f.read()
            self.assertIn("xtask_protocol_vllm_api_general_chat", content)
            self.assertIn("xtask_protocol_constrained_gen", content)
            self.assertIn("protocol_results", content)

    @patch("xtask.benchmarks.aisbench.generate_aisbench_script.is_protocol_test_enabled", return_value=True)
    @patch("xtask.benchmarks.aisbench.generate_aisbench_script.create_protocol_model_config")
    @patch("xtask.benchmarks.aisbench.generate_aisbench_script.create_dataset_config_override")
    def test_protocol_overrides_passed_and_no_sed(
        self, _mock_dataset, mock_model, _mock_switch
    ):
        """Connection params are baked via overrides kwarg; no sed for protocol model."""
        from xtask.benchmarks.aisbench.generate_aisbench_script import generate_aisbench_script

        mock_model.return_value = "xtask_protocol_vllm_api_general_chat"
        _mock_dataset.return_value = "xtask_protocol_constrained_gen"

        config = self._make_task_config()
        config["benchmark"]["protocol"]["output_len"] = 2048
        config["benchmark"]["protocol"]["batch_size"] = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aisbench_script(config, tmpdir, "test.sh")
            with open(os.path.join(tmpdir, "test.sh"), "r") as f:
                content = f.read()

            # create_protocol_model_config must receive overrides with all params
            _, kwargs = mock_model.call_args
            self.assertIn("overrides", kwargs)
            ov = kwargs["overrides"]
            self.assertEqual(ov["host_ip"], "127.0.0.1")
            self.assertEqual(ov["host_port"], 8000)
            self.assertEqual(ov["model"], "test-model")
            self.assertEqual(ov["path"], "/data/model")
            self.assertEqual(ov["max_out_len"], 2048)
            self.assertEqual(ov["batch_size"], 4)
            self.assertEqual(ov["request_rate"], 0)
            # temperature must NOT be in overrides — it belongs in
            # generation_kwargs, not as a top-level model dict key.
            self.assertNotIn("temperature", ov)

            # No sed commands targeting the protocol model config file
            self.assertNotIn("sed -i", content.split("# 6.2")[1].split("# 6.4")[0]
                             if "# 6.2" in content and "# 6.4" in content else "no-sed")

    def test_no_protocol_metrics_skips(self):
        from xtask.benchmarks.aisbench.generate_aisbench_script import generate_aisbench_script

        # Use metrics="acc" without an acc dataset_conf so the acc section is
        # skipped (no AISBench config files needed on the host); verify the
        # protocol section is not generated either.
        config = self._make_task_config(metrics="acc")
        config["benchmark"]["acc"] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aisbench_script(config, tmpdir, "test.sh")
            with open(os.path.join(tmpdir, "test.sh"), "r") as f:
                content = f.read()
            self.assertNotIn("协议一致性", content)
            self.assertNotIn("xtask_protocol_", content)


class TestCreateProtocolModelConfigOverrides(unittest.TestCase):
    """Tests that connection params are baked into the AST, not left to sed."""

    _SOURCE_CONFIG = '''\
from ais_bench.benchmark.models.vllm_api_general_chat import VLLMCustomAPIChat

models = [
    dict(
        abbr='vllm-api-general-chat',
        attr='service',
        model='placeholder-model',
        path='/placeholder/path',
        host_ip='localhost',
        host_port=8080,
        max_out_len=512,
        batch_size=1,
        request_rate=-1,
        temperature=0.6,
        type=VLLMCustomAPIChat,
        generation_kwargs=dict(do_sample=True),
    ),
]
'''

    @staticmethod
    def _extract_model_dict(file_path):
        """Extract keyword values from the first dict() in models = [...] via AST."""
        tree = ast.parse(Path(file_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not (isinstance(target, ast.Name) and target.id == "models"):
                    continue
                if not isinstance(node.value, ast.List) or not node.value.elts:
                    continue
                call = node.value.elts[0]
                if not isinstance(call, ast.Call):
                    continue

                def extract_value(value_node):
                    # Nested dict(...) call → recurse into its keywords.
                    if (
                        isinstance(value_node, ast.Call)
                        and isinstance(value_node.func, ast.Name)
                        and value_node.func.id == "dict"
                    ):
                        nested = {}
                        for nkw in value_node.keywords:
                            if nkw.arg:
                                nested[nkw.arg] = extract_value(nkw.value)
                        return nested
                    try:
                        return ast.literal_eval(value_node)
                    except (ValueError, SyntaxError):
                        return ast.unparse(value_node)

                result = {}
                for kw in call.keywords:
                    if kw.arg is None:
                        continue
                    result[kw.arg] = extract_value(kw.value)
                return result
        return {}

    def _make_source(self, tmpdir):
        root = Path(tmpdir) / "ais"
        config = (
            root
            / "ais_bench"
            / "benchmark"
            / "configs"
            / "models"
            / "vllm_api"
            / "vllm_api_general_chat.py"
        )
        config.parent.mkdir(parents=True)
        config.write_text(self._SOURCE_CONFIG, encoding="utf-8")
        return root

    def test_overrides_baked_into_ast(self):
        """All connection params appear in the generated config with correct values."""
        from xtask.benchmarks.aisbench import generate_aisbench_script as generator

        with tempfile.TemporaryDirectory() as tmpdir:
            ais_root = self._make_source(tmpdir)
            overrides = {
                "host_ip": "0.0.0.0",
                "host_port": 8800,
                "model": "Qwen3-8B",
                "path": "/export/home/models/Qwen3-8B",
                "max_out_len": 4096,
                "batch_size": 8,
                "request_rate": 0,
            }
            with mock.patch.object(
                generator,
                "_aisbench_dataset_config_roots",
                return_value=[ais_root],
            ):
                name = generator.create_protocol_model_config(
                    "vllm_api_general_chat",
                    tmpdir,
                    stream_verify=True,
                    overrides=overrides,
                    generation_kwargs={"top_k": 50},
                )

            self.assertEqual(name, "xtask_protocol_vllm_api_general_chat")
            generated = Path(tmpdir) / "aisbench_configs" / "models" / f"{name}.py"
            self.assertTrue(generated.exists())

            model_dict = self._extract_model_dict(generated)
            self.assertEqual(model_dict["host_ip"], "0.0.0.0")
            self.assertEqual(model_dict["host_port"], 8800)
            self.assertEqual(model_dict["model"], "Qwen3-8B")
            self.assertEqual(model_dict["path"], "/export/home/models/Qwen3-8B")
            self.assertEqual(model_dict["max_out_len"], 4096)
            self.assertEqual(model_dict["batch_size"], 8)
            self.assertEqual(model_dict["request_rate"], 0)
            # temperature from the source config (0.6) must be REMOVED —
            # it's a generation-only key that crashes VLLMCustomAPIChat.__init__.
            self.assertNotIn("temperature", model_dict)
            self.assertTrue(model_dict["stream_verify"])
            self.assertEqual(model_dict["generation_kwargs"]["top_k"], 50)
            self.assertEqual(model_dict["generation_kwargs"]["do_sample"], True)


class TestProtocolAwareAPIChatWrapper(unittest.TestCase):
    """Tests for the ProtocolAwareAPIChat wrapper class (lazy composition)."""

    def _import_wrapper(self):
        from xtask.benchmarks.aisbench.protocol_api import ProtocolAwareAPIChat
        return ProtocolAwareAPIChat


    def test_wrapper_exists_and_is_class(self):
        cls = self._import_wrapper()
        self.assertTrue(isinstance(cls, type))

    def test_new_delegates_to_factory(self):
        """__new__ should call create_protocol_aware_chat with stream_verify=False."""
        cls = self._import_wrapper()

        class _FakeComposed:
            def __init__(self, *a, **kw):
                pass

        with patch(
            "xtask.benchmarks.aisbench.protocol_api.create_protocol_aware_chat"
        ) as mock_factory:
            mock_factory.return_value = _FakeComposed
            instance = cls.__new__(cls)
            mock_factory.assert_called_once_with(stream_verify=False)
            self.assertIsInstance(instance, _FakeComposed)

    def test_new_stream_verify_true(self):
        """Passing stream_verify=True should forward to the factory."""
        cls = self._import_wrapper()

        class _FakeComposed:
            def __init__(self, *a, **kw):
                pass

        with patch(
            "xtask.benchmarks.aisbench.protocol_api.create_protocol_aware_chat"
        ) as mock_factory:
            mock_factory.return_value = _FakeComposed
            cls.__new__(cls, stream_verify=True)
            mock_factory.assert_called_once_with(stream_verify=True)

    def test_instance_proxies_attributes(self):
        """The returned instance should be of the composed class."""
        cls = self._import_wrapper()

        class _FakeImpl:
            answer = 42

            def greet(self):
                return "hello"

        with patch(
            "xtask.benchmarks.aisbench.protocol_api.create_protocol_aware_chat"
        ) as mock_factory:
            mock_factory.return_value = _FakeImpl
            instance = cls.__new__(cls)
            self.assertEqual(instance.answer, 42)
            self.assertEqual(instance.greet(), "hello")

    def test_new_pops_generation_only_params(self):
        """temperature/top_p etc. must be popped before base __init__ and
        merged into generation_kwargs so they reach each request payload
        via VLLMCustomAPIChat.get_request_body()."""
        cls = self._import_wrapper()
        received_kwargs = {}

        class _FakeComposed:
            def __init__(self, *a, **kw):
                received_kwargs.update(kw)

        with patch(
            "xtask.benchmarks.aisbench.protocol_api.create_protocol_aware_chat"
        ) as mock_factory:
            mock_factory.return_value = _FakeComposed
            instance = cls.__new__(
                cls, host_ip="1.2.3.4", temperature=0.7, top_p=0.9
            )
            # Base constructor must NOT receive generation-only params.
            self.assertNotIn("temperature", received_kwargs)
            self.assertNotIn("top_p", received_kwargs)
            self.assertEqual(received_kwargs["host_ip"], "1.2.3.4")
            # They are merged into generation_kwargs for payload injection.
            self.assertEqual(
                received_kwargs["generation_kwargs"]["temperature"], 0.7
            )
            self.assertEqual(
                received_kwargs["generation_kwargs"]["top_p"], 0.9
            )

    def test_new_preserves_existing_generation_kwargs(self):
        """Pre-existing generation_kwargs must be preserved when merging
        generation-only params into them."""
        cls = self._import_wrapper()
        received_kwargs = {}

        class _FakeComposed:
            def __init__(self, *a, **kw):
                received_kwargs.update(kw)

        with patch(
            "xtask.benchmarks.aisbench.protocol_api.create_protocol_aware_chat"
        ) as mock_factory:
            mock_factory.return_value = _FakeComposed
            cls.__new__(
                cls,
                generation_kwargs={"do_sample": True},
                temperature=0.6,
            )
            gk = received_kwargs["generation_kwargs"]
            self.assertEqual(gk["do_sample"], True)
            self.assertEqual(gk["temperature"], 0.6)


class TestProtocolResultPostProcess(unittest.TestCase):
    """Tests for aisbench_postprocess.protocol_result_post_process."""

    def _make_result_dir(self, tmpdir, date_dir_name="20261117_210000"):
        """Create a protocol_results/<date>/ structure with a summary JSON."""
        result_dir = os.path.join(tmpdir, "protocol_results", date_dir_name)
        os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def test_finds_flat_protocol_scores(self):
        from xtask.benchmarks.aisbench.aisbench_postprocess import (
            protocol_result_post_process,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = self._make_result_dir(tmpdir)
            summary = {
                "Protocol_Pass_Rate": 100.0,
                "Syntax_Valid_Rate": 100.0,
                "Semantic_Accuracy": 87.5,
            }
            with open(os.path.join(result_dir, "summary.json"), "w") as f:
                json.dump(summary, f)

            result, output_dir = protocol_result_post_process(
                {}, tmpdir, []
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["Protocol_Pass_Rate"], 100.0)
            self.assertEqual(result["Semantic_Accuracy"], 87.5)
            self.assertIn("20261117_210000", output_dir)

    def test_finds_nested_protocol_scores(self):
        from xtask.benchmarks.aisbench.aisbench_postprocess import (
            protocol_result_post_process,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = self._make_result_dir(tmpdir)
            # ais_bench may nest: {dataset: {model: scores}}
            nested = {
                "protocol_constrained": {
                    "xtask_protocol_vllm_api_general_chat": {
                        "Protocol_Pass_Rate": 75.0,
                        "Syntax_Valid_Rate": 90.0,
                        "Semantic_Accuracy": 50.0,
                    }
                }
            }
            with open(os.path.join(result_dir, "result.json"), "w") as f:
                json.dump(nested, f)

            result, _ = protocol_result_post_process({}, tmpdir, [])
            self.assertIsNotNone(result)
            self.assertEqual(result["Protocol_Pass_Rate"], 75.0)

    def test_missing_protocol_results_dir(self):
        from xtask.benchmarks.aisbench.aisbench_postprocess import (
            protocol_result_post_process,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result, output_dir = protocol_result_post_process({}, tmpdir, [])
            self.assertIsNone(result)
            self.assertEqual(output_dir, "")

    def test_no_scores_in_result_dir(self):
        from xtask.benchmarks.aisbench.aisbench_postprocess import (
            protocol_result_post_process,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_result_dir(tmpdir)
            with open(
                os.path.join(tmpdir, "protocol_results", "20261117_210000", "other.json"),
                "w",
            ) as f:
                json.dump({"foo": "bar"}, f)

            result, output_dir = protocol_result_post_process({}, tmpdir, [])
            self.assertIsNone(result)
            self.assertEqual(output_dir, "")


class TestVerifyResultProtocolBranch(unittest.TestCase):
    """Tests for result_processor.verify_result with metrics=protocol."""

    def _make_task_config(self, baseline=None, threshold=0.04,
                          protocol_tolerance=None, syntax_tolerance=None):
        protocol_cfg = {
            "case_type": "protocol",
            "api_type": "vllm_api_general_chat",
        }
        if baseline is not None:
            protocol_cfg["baseline"] = baseline
            protocol_cfg["threshold"] = threshold
        if protocol_tolerance is not None:
            protocol_cfg["protocol_tolerance"] = protocol_tolerance
        if syntax_tolerance is not None:
            protocol_cfg["syntax_tolerance"] = syntax_tolerance
        return {
            "metrics": ["protocol"],
            "benchmark": {"benchmark_type": "aisbench", "protocol": protocol_cfg},
        }

    def _make_result(self, protocol_pass=100.0, syntax=100.0, semantic=100.0):
        return {
            "benchmark_success": True,
            "benchmark_protocol_result": {
                "Protocol_Pass_Rate": protocol_pass,
                "Syntax_Valid_Rate": syntax,
                "Semantic_Accuracy": semantic,
            },
        }

    def test_all_pass_returns_true(self):
        from xtask.core.result_processor import verify_result

        result = self._make_result()
        task_config = self._make_task_config()
        self.assertTrue(verify_result(result, task_config))

    def test_missing_protocol_result_returns_false(self):
        from xtask.core.result_processor import verify_result

        result = {"benchmark_success": True}
        task_config = self._make_task_config()
        self.assertFalse(verify_result(result, task_config))

    def test_low_protocol_rate_returns_false(self):
        from xtask.core.result_processor import verify_result

        result = self._make_result(protocol_pass=50.0)
        task_config = self._make_task_config()
        self.assertFalse(verify_result(result, task_config))

    def test_low_syntax_rate_returns_false(self):
        from xtask.core.result_processor import verify_result

        result = self._make_result(syntax=80.0)
        task_config = self._make_task_config()
        self.assertFalse(verify_result(result, task_config))

    def test_protocol_tolerance_allows_below_100(self):
        from xtask.core.result_processor import verify_result

        result = self._make_result(protocol_pass=95.0)
        task_config = self._make_task_config(protocol_tolerance=5.0)
        self.assertTrue(verify_result(result, task_config))

    def test_baseline_check_passes(self):
        from xtask.core.result_processor import verify_result

        result = self._make_result(semantic=96.0)
        task_config = self._make_task_config(baseline=96.0, threshold=0.04)
        self.assertTrue(verify_result(result, task_config))

    def test_baseline_check_fails(self):
        from xtask.core.result_processor import verify_result

        result = self._make_result(semantic=50.0)
        task_config = self._make_task_config(baseline=96.0, threshold=0.04)
        self.assertFalse(verify_result(result, task_config))


class TestProtocolConstrainedDatasetLoad(unittest.TestCase):
    """Tests that ProtocolConstrainedDataset.load() accepts meta_path kwarg.

    AISBench's CustomDataset.__init__ calls ``self.load(**kwargs)`` passing
    all dataset-config dict keys (path, meta_path, reader_cfg, ...).  The
    override must accept these without TypeError.
    """

    @classmethod
    def setUpClass(cls):
        """Inject a fake ais_bench.benchmark.datasets module so the import
        succeeds on the host without ais_bench installed."""
        import sys
        import types

        fake_pkg = types.ModuleType("ais_bench")
        fake_bench = types.ModuleType("ais_bench.benchmark")
        fake_datasets = types.ModuleType("ais_bench.benchmark.datasets")

        class _FakeCustomDataset:
            def __init__(self, **kwargs):
                self.path = kwargs.get("path", "")

        fake_datasets.CustomDataset = _FakeCustomDataset
        fake_pkg.benchmark = fake_bench
        fake_bench.datasets = fake_datasets

        cls._saved_modules = {}
        for name, mod in [
            ("ais_bench", fake_pkg),
            ("ais_bench.benchmark", fake_bench),
            ("ais_bench.benchmark.datasets", fake_datasets),
        ]:
            cls._saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = mod

        # Force re-import of protocol_dataset with the fake base class.
        cls._saved_protocol_dataset = sys.modules.pop(
            "xtask.benchmarks.aisbench.protocol_dataset", None
        )
        from xtask.benchmarks.aisbench.protocol_dataset import (
            ProtocolConstrainedDataset,
        )

        cls.ProtocolConstrainedDataset = ProtocolConstrainedDataset

    @classmethod
    def tearDownClass(cls):
        import sys

        for name, mod in cls._saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
        if cls._saved_protocol_dataset is not None:
            sys.modules["xtask.benchmarks.aisbench.protocol_dataset"] = (
                cls._saved_protocol_dataset
            )
        else:
            sys.modules.pop("xtask.benchmarks.aisbench.protocol_dataset", None)

    def _write_jsonl(self, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_load_accepts_path_kwarg(self):
        """load(path=...) must read from the path kwarg — this is how
        AISBench CustomDataset.__init__ passes the data file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(jsonl, [
                {"question": "q1", "extra_params": {"tools": []},
                 "ground_truth": {"answer": "a1"}},
            ])
            ds = self.ProtocolConstrainedDataset(path="unused")
            data = ds.load(path=jsonl, meta_path="", abbr="test", type=None)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["question"], "q1")
            # extra_params is serialized to a JSON string (see load() docstring);
            # verify it round-trips back to the original dict.
            self.assertIsInstance(data[0]["extra_params"], str)
            self.assertEqual(json.loads(data[0]["extra_params"]), {"tools": []})

    def test_load_returns_hf_dataset(self):
        """load() must return a HuggingFace datasets.Dataset, not a list —
        AISBench check_dataset() rejects plain lists."""
        from datasets import Dataset as HFDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(jsonl, [
                {"question": "q1", "extra_params": {"response_format": "json"},
                 "ground_truth": {"expected_finish_reason": "stop"}},
            ])
            ds = self.ProtocolConstrainedDataset(path="unused")
            data = ds.load(path=jsonl)
            self.assertIsInstance(data, HFDataset)
            self.assertNotIsInstance(data, list)

    def test_load_accepts_meta_path_kwarg(self):
        """load(meta_path=...) must read from meta_path when path is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(jsonl, [
                {"question": "q1", "extra_params": {"tools": []},
                 "ground_truth": {"answer": "a1"}},
            ])
            ds = self.ProtocolConstrainedDataset(path="unused")
            data = ds.load(meta_path=jsonl)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["question"], "q1")
            self.assertIsInstance(data[0]["extra_params"], str)
            self.assertEqual(json.loads(data[0]["extra_params"]), {"tools": []})

    def test_load_accepts_extra_kwargs(self):
        """load() must tolerate arbitrary kwargs from the config dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(jsonl, [
                {"question": "q1", "extra_params": {}, "ground_truth": {}},
            ])
            ds = self.ProtocolConstrainedDataset(path=jsonl)
            data = ds.load(
                path=jsonl, meta_path="", reader_cfg={}, infer_cfg={},
                eval_cfg={}, abbr="test", type=None,
            )
            self.assertEqual(len(data), 1)

    def test_load_falls_back_to_self_path(self):
        """When path and meta_path are empty/None, load() falls back to self.path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(jsonl, [
                {"question": "q1", "extra_params": {}, "ground_truth": {}},
            ])
            ds = self.ProtocolConstrainedDataset(path=jsonl)
            data = ds.load(path="", meta_path="")
            self.assertEqual(len(data), 1)

    def test_load_raises_without_path(self):
        ds = self.ProtocolConstrainedDataset(path="")
        with self.assertRaises(ValueError):
            ds.load(path=None, meta_path=None)

    def test_load_serializes_complex_fields(self):
        """extra_params and ground_truth are serialized to JSON strings.

        This avoids pyarrow ArrowInvalid when the same key has different
        value types across rows (e.g. tool_choice: "none" vs {"type": ...}).
        Consumers deserialize via json.loads (evaluator score(), API adapter
        _capture_extra_params()).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(jsonl, [
                {"question": "q1", "extra_params": {"response_format": "json"},
                 "ground_truth": {"expected_finish_reason": "stop"}},
                {"question": "q2"},  # missing extra_params/ground_truth
            ])
            ds = self.ProtocolConstrainedDataset(path="")
            data = ds.load(path=jsonl)
            # Row 1: both fields are JSON strings that round-trip to dicts.
            self.assertIsInstance(data[0]["extra_params"], str)
            self.assertIsInstance(data[0]["ground_truth"], str)
            self.assertEqual(json.loads(data[0]["extra_params"]),
                             {"response_format": "json"})
            self.assertEqual(json.loads(data[0]["ground_truth"]),
                             {"expected_finish_reason": "stop"})
            # Row 2: missing fields default to "{}" strings.
            self.assertEqual(data[1]["extra_params"], "{}")
            self.assertEqual(data[1]["ground_truth"], "{}")

    def test_load_handles_heterogeneous_types(self):
        """Mixed value types for the same key must not crash pyarrow.

        tool_choice is "none" (string) in one row and a dict in another —
        both valid per OpenAI API spec.  Without serialization this raises
        ``ArrowInvalid: cannot mix struct and non-struct, non-null values``.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(jsonl, [
                {"question": "q1",
                 "extra_params": {"tools": [], "tool_choice": "none"},
                 "ground_truth": {"expected_finish_reason": "stop"}},
                {"question": "q2",
                 "extra_params": {"tools": [], "tool_choice": {
                     "type": "function", "function": {"name": "f"}}},
                 "ground_truth": {"expected_finish_reason": "tool_calls"}},
            ])
            ds = self.ProtocolConstrainedDataset(path="")
            data = ds.load(path=jsonl)  # must not raise ArrowInvalid
            self.assertEqual(len(data), 2)
            ep1 = json.loads(data[0]["extra_params"])
            ep2 = json.loads(data[1]["extra_params"])
            self.assertEqual(ep1["tool_choice"], "none")
            self.assertIsInstance(ep2["tool_choice"], dict)


class TestParseExtraParams(unittest.TestCase):
    """Tests for parse_extra_params pure helper."""

    def test_dict_passthrough(self):
        from xtask.benchmarks.aisbench.protocol_api import parse_extra_params

        result = parse_extra_params({"tools": []})
        self.assertEqual(result, {"tools": []})

    def test_json_string_deserialized(self):
        from xtask.benchmarks.aisbench.protocol_api import parse_extra_params

        result = parse_extra_params('{"tool_choice": "none"}')
        self.assertEqual(result, {"tool_choice": "none"})

    def test_invalid_json_returns_none(self):
        from xtask.benchmarks.aisbench.protocol_api import parse_extra_params

        self.assertIsNone(parse_extra_params("not json"))

    def test_none_returns_none(self):
        from xtask.benchmarks.aisbench.protocol_api import parse_extra_params

        self.assertIsNone(parse_extra_params(None))

    def test_non_dict_returns_none(self):
        from xtask.benchmarks.aisbench.protocol_api import parse_extra_params

        self.assertIsNone(parse_extra_params(42))
        self.assertIsNone(parse_extra_params([1, 2]))
        # Valid JSON but not a dict
        self.assertIsNone(parse_extra_params('"just a string"'))
        self.assertIsNone(parse_extra_params('[1, 2, 3]'))





class TestProtocolAwareAdapterIntegration(unittest.TestCase):
    """Integration tests for the protocol-aware adapter with fake ais_bench.

    Injects fake ais_bench modules matching the PyPI 3.1.x async
    architecture (generate -> get_request_body -> text_infer /
    stream_infer -> parse_*_response) so the full async flow can be
    tested on the host without ais_bench installed.
    """

    @classmethod
    def setUpClass(cls):
        import sys
        import types

        class FakeOutput:
            """Minimal stand-in for ais_bench Output."""

            def __init__(self):
                self.content = ""
                self.reasoning_content = ""
                self.success = True

            def get_prediction(self):
                if self.reasoning_content:
                    return self.reasoning_content + "\n\n" + self.content
                return self.content

        class FakeAiohttpResponse:
            def __init__(self, status=200, json_data=None):
                self.status = status
                self._json_data = json_data or {}

            async def json(self):
                return self._json_data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeAiohttpSession:
            def __init__(self, response=None):
                self._response = response
                self.last_request = None

            def post(self, url, json=None, headers=None):
                self.last_request = {"url": url, "json": json,
                                     "headers": headers}
                return FakeAiohttpResponse(json_data=self._response)

        class FakeVLLMCustomAPIChat:
            """Minimal async base matching PyPI BaseAPIModel call chain."""

            def __init__(self, path="", model="test-model",
                         generation_kwargs=None, stream=False, **kwargs):
                self.path = path
                self.model = model
                self.generation_kwargs = generation_kwargs or {}
                self.stream = stream
                self.url = "http://localhost:8080/v1/chat/completions"
                self.headers = {"Authorization": "Bearer EMPTY"}
                self.session = None
                self._mock_response = None
                self._mock_stream_chunks = []
                self.last_request_body = None

            async def get_request_body(self, input, max_out_len, output,
                                       **args):
                messages = input if isinstance(input, list) else [
                    {"role": "user", "content": str(input)}]
                body = {"model": self.model, "messages": messages,
                        "max_tokens": max_out_len, "stream": self.stream}
                body.update(self.generation_kwargs)
                return body

            async def text_infer(self, request_body, output):
                self.last_request_body = request_body
                await self.parse_text_response(self._mock_response, output)

            async def parse_text_response(self, json_content, output):
                choices = (json_content or {}).get("choices") or [{}]
                choice = choices[0] if isinstance(choices[0], dict) else {}
                msg = choice.get("message") or {}
                output.content = msg.get("content") or ""
                output.reasoning_content = msg.get("reasoning_content") or ""

            async def stream_infer(self, request_body, output):
                self.last_request_body = request_body
                for chunk in self._mock_stream_chunks:
                    await self.parse_stream_response(chunk, output)

            async def parse_stream_response(self, json_content, output):
                choices = (json_content or {}).get("choices") or [{}]
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    output.content += delta["content"]
                if delta.get("reasoning_content"):
                    output.reasoning_content += delta["reasoning_content"]

            async def generate(self, input, max_out_len, output,
                               session=None, **args):
                body = await self.get_request_body(
                    input, max_out_len, output, **args)
                if self.stream:
                    await self.stream_infer(body, output)
                else:
                    await self.text_infer(body, output)
                return output

        fake_pkg = types.ModuleType("ais_bench")
        fake_bench = types.ModuleType("ais_bench.benchmark")
        fake_models = types.ModuleType("ais_bench.benchmark.models")
        fake_models.VLLMCustomAPIChat = FakeVLLMCustomAPIChat
        fake_pkg.benchmark = fake_bench
        fake_bench.models = fake_models

        cls._saved_modules = {}
        for name, mod in [
            ("ais_bench", fake_pkg),
            ("ais_bench.benchmark", fake_bench),
            ("ais_bench.benchmark.models", fake_models),
        ]:
            cls._saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = mod

        cls._saved_protocol_api = sys.modules.pop(
            "xtask.benchmarks.aisbench.protocol_api", None)
        from xtask.benchmarks.aisbench.protocol_api import (
            create_protocol_aware_chat,
        )

        cls.create_protocol_aware_chat = staticmethod(create_protocol_aware_chat)
        cls.FakeOutput = FakeOutput
        cls.FakeAiohttpSession = FakeAiohttpSession

    @classmethod
    def tearDownClass(cls):
        import sys

        for name, mod in cls._saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
        if cls._saved_protocol_api is not None:
            sys.modules["xtask.benchmarks.aisbench.protocol_api"] = (
                cls._saved_protocol_api
            )
        else:
            sys.modules.pop("xtask.benchmarks.aisbench.protocol_api", None)

    def test_non_stream_generate_preserves_full_message(self):
        """generate() must store the full message as JSON in output.content."""
        adapter_cls = self.create_protocol_aware_chat(stream_verify=False)
        adapter = adapter_cls(path="/fake", model="test-model")

        full_response = {
            "choices": [{
                "message": {
                    "content": "answer",
                    "reasoning_content": "thinking",
                    "tool_calls": [{"id": "1", "function": {"name": "f"}}],
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }]
        }
        adapter._mock_response = full_response

        output = self.FakeOutput()
        asyncio.run(adapter.generate("Hello", 512, output))

        parsed = json.loads(output.content)
        self.assertEqual(parsed["content"], "answer")
        self.assertEqual(parsed["reasoning_content"], "thinking")
        self.assertEqual(parsed["finish_reason"], "stop")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        # reasoning_content cleared so get_prediction returns full JSON
        self.assertEqual(output.reasoning_content, "")
        self.assertEqual(output.get_prediction(), output.content)

    def test_non_stream_generate_injects_protocol_fields(self):
        """extra_params from generate args must be injected into the request."""
        adapter_cls = self.create_protocol_aware_chat(stream_verify=False)
        adapter = adapter_cls(path="/fake", model="test-model")

        adapter._mock_response = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }

        output = self.FakeOutput()
        extra = ('{"tools": [{"type": "function", "function": {"name": "f"}}],'
                 ' "tool_choice": "auto"}')
        asyncio.run(adapter.generate("Hello", 256, output, extra_params=extra))

        body = adapter.last_request_body
        self.assertIn("tools", body)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["max_tokens"], 256)
        self.assertEqual(body["model"], "test-model")

    def test_non_stream_generate_with_generation_kwargs(self):
        """Construction-time generation_kwargs must reach the request body."""
        adapter_cls = self.create_protocol_aware_chat(stream_verify=False)
        adapter = adapter_cls(
            path="/fake", model="test-model",
            generation_kwargs={"temperature": 0.7},
        )

        adapter._mock_response = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }

        output = self.FakeOutput()
        asyncio.run(adapter.generate("Hi", 128, output))

        body = adapter.last_request_body
        self.assertEqual(body["temperature"], 0.7)

    def test_non_stream_generate_string_input(self):
        """String input (not a list) must also work."""
        adapter_cls = self.create_protocol_aware_chat(stream_verify=False)
        adapter = adapter_cls(path="/fake", model="test-model")

        adapter._mock_response = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }

        output = self.FakeOutput()
        asyncio.run(adapter.generate("Hello world", 64, output))
        parsed = json.loads(output.content)
        self.assertEqual(parsed["content"], "ok")

    def test_stream_generate_assembles_deltas(self):
        """Stream generate must assemble SSE deltas into a full message."""
        adapter_cls = self.create_protocol_aware_chat(stream_verify=False)
        adapter = adapter_cls(path="/fake", model="test-model", stream=True)

        adapter._mock_stream_chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]

        output = self.FakeOutput()
        asyncio.run(adapter.generate("Hi", 64, output))

        parsed = json.loads(output.content)
        self.assertEqual(parsed["content"], "Hello world")
        self.assertEqual(parsed["finish_reason"], "stop")

    def test_stream_generate_with_reasoning_and_tool_calls(self):
        """Stream generate must assemble reasoning + tool_calls deltas."""
        adapter_cls = self.create_protocol_aware_chat(stream_verify=False)
        adapter = adapter_cls(path="/fake", model="test-model", stream=True)

        adapter._mock_stream_chunks = [
            {"choices": [{"delta": {"reasoning_content": "step1"}, "finish_reason": None}]},
            {"choices": [{"delta": {"reasoning_content": "step2"}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "get_weather", "arguments": ""}
            }]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": '{"city": "BJ"}'}
            }]}, "finish_reason": "tool_calls"}]},
        ]

        output = self.FakeOutput()
        asyncio.run(adapter.generate("Weather?", 128, output))

        parsed = json.loads(output.content)
        self.assertEqual(parsed["reasoning_content"], "step1step2")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(
            parsed["tool_calls"][0]["function"]["arguments"], '{"city": "BJ"}'
        )
        self.assertEqual(parsed["finish_reason"], "tool_calls")

    def test_stream_verify_consistent(self):
        """Stream verify passes when stream and non-stream match."""
        adapter_cls = self.create_protocol_aware_chat(stream_verify=True)
        adapter = adapter_cls(path="/fake", model="test-model", stream=True)

        adapter._mock_stream_chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        adapter.session = self.FakeAiohttpSession(response={
            "choices": [{
                "message": {"content": "Hello", "role": "assistant"},
                "finish_reason": "stop",
            }]
        })

        output = self.FakeOutput()
        asyncio.run(adapter.generate("Hi", 64, output))

        parsed = json.loads(output.content)
        self.assertEqual(parsed["content"], "Hello")
        self.assertEqual(parsed["finish_reason"], "stop")
        # Verification request was made with stream=False
        self.assertFalse(adapter.session.last_request["json"]["stream"])

    def test_stream_verify_inconsistent(self):
        """Stream verify logs a warning but still returns the stream result."""
        adapter_cls = self.create_protocol_aware_chat(stream_verify=True)
        adapter = adapter_cls(path="/fake", model="test-model", stream=True)

        adapter._mock_stream_chunks = [
            {"choices": [{"delta": {"content": "Stream says A"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        adapter.session = self.FakeAiohttpSession(response={
            "choices": [{
                "message": {"content": "Non-stream says B", "role": "assistant"},
                "finish_reason": "stop",
            }]
        })

        output = self.FakeOutput()
        with self.assertLogs(
            "xtask.benchmarks.aisbench.protocol_api", level="WARNING"
        ) as cm:
            asyncio.run(adapter.generate("Hi", 64, output))

        # Stream result is still returned despite mismatch
        parsed = json.loads(output.content)
        self.assertEqual(parsed["content"], "Stream says A")
        self.assertTrue(any("mismatch" in msg.lower() for msg in cm.output))


class TestWarmupDefenseInferencer(unittest.TestCase):
    """ProtocolGenInferencer.warmup() strips extra_params from the
    data_list parameter so warmup requests don't carry protocol fields.

    Real AISBench 3.1.x flow (verified from source):
        Task.warm_up(data_list) -> new inferencer.warmup(data_list, ...)
        -> BaseApiInferencer.warmup() iterates data_list, calls
           do_request(copy.deepcopy(data)) -> model.generate(**data)
        -> get_request_body(**args) -> inject_protocol_fields()

    Since get_data_list() already injected extra_params into data_list,
    warmup() must strip them before delegating to super().warmup().
    """

    @classmethod
    def setUpClass(cls):
        import sys
        import types

        # Fake BaseApiInferencer.warmup — records what data it receives
        class FakeGenInferencer:
            def __init__(self):
                self.warmup_received_data = None
                self.warmup_call_args = None

            def get_data_list(self, retriever):
                return []

            async def warmup(self, data_list, warmup_times=1,
                             concurrency=1):
                # Record what data_list the parent warmup received
                self.warmup_received_data = [
                    dict(item) if isinstance(item, dict) else item
                    for item in data_list
                ]
                self.warmup_call_args = (warmup_times, concurrency)
                return {"success": 1, "failed": 0, "failed_reasons": {}}

        fake_pkg = types.ModuleType("ais_bench")
        fake_bench = types.ModuleType("ais_bench.benchmark")
        fake_openicl = types.ModuleType("ais_bench.benchmark.openicl")
        fake_inferencer_mod = types.ModuleType(
            "ais_bench.benchmark.openicl.icl_inferencer")
        fake_inferencer_mod.GenInferencer = FakeGenInferencer
        fake_pkg.benchmark = fake_bench
        fake_bench.openicl = fake_openicl
        fake_openicl.icl_inferencer = fake_inferencer_mod

        cls._fake_modules = {
            "ais_bench": fake_pkg,
            "ais_bench.benchmark": fake_bench,
            "ais_bench.benchmark.openicl": fake_openicl,
            "ais_bench.benchmark.openicl.icl_inferencer": fake_inferencer_mod,
        }
        cls.FakeGenInferencer = FakeGenInferencer

        cls._saved_modules = {}
        for name, mod in cls._fake_modules.items():
            cls._saved_modules[name] = sys.modules.get(name)
            sys.modules[name] = mod

        cls._saved_proto_inc = sys.modules.pop(
            "xtask.benchmarks.aisbench.protocol_inferencer", None)
        from xtask.benchmarks.aisbench.protocol_inferencer import (
            ProtocolGenInferencer,
        )
        cls.ProtocolGenInferencer = ProtocolGenInferencer

    @classmethod
    def tearDownClass(cls):
        import sys

        for name, mod in cls._saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
        if cls._saved_proto_inc is not None:
            sys.modules[
                "xtask.benchmarks.aisbench.protocol_inferencer"
            ] = cls._saved_proto_inc

    def _make_data_list(self):
        return [
            {"prompt": "q1", "index": 0, "data_abbr": "ds",
             "max_out_len": 128,
             "extra_params": {"response_format": {"type": "json_schema"}}},
            {"prompt": "q2", "index": 1, "data_abbr": "ds",
             "max_out_len": 128,
             "extra_params": {"guided_regex": "[0-9]+"}},
            {"prompt": "q3", "index": 2, "data_abbr": "ds",
             "max_out_len": 128},
        ]

    def test_warmup_strips_extra_params(self):
        """warmup() must strip extra_params from data passed to parent."""
        inferencer = self.ProtocolGenInferencer()
        data_list = self._make_data_list()
        asyncio.run(inferencer.warmup(data_list, warmup_times=2,
                                      concurrency=1))

        # Parent warmup received data WITHOUT extra_params
        received = inferencer.warmup_received_data
        self.assertIsNotNone(received)
        for item in received:
            self.assertNotIn("extra_params", item)
        # Other fields preserved
        self.assertEqual(received[0]["prompt"], "q1")
        self.assertEqual(received[0]["index"], 0)

    def test_warmup_does_not_modify_original(self):
        """Original data_list must be untouched for real inference."""
        inferencer = self.ProtocolGenInferencer()
        data_list = self._make_data_list()
        asyncio.run(inferencer.warmup(data_list, warmup_times=1))

        # Original still has extra_params
        self.assertIn("extra_params", data_list[0])
        self.assertEqual(
            data_list[0]["extra_params"],
            {"response_format": {"type": "json_schema"}},
        )
        self.assertIn("extra_params", data_list[1])
        # Item without extra_params is unaffected
        self.assertNotIn("extra_params", data_list[2])

    def test_warmup_forwards_args(self):
        """warmup_times and concurrency must be forwarded to parent."""
        inferencer = self.ProtocolGenInferencer()
        asyncio.run(inferencer.warmup(self._make_data_list(),
                                      warmup_times=5, concurrency=3))
        self.assertEqual(inferencer.warmup_call_args, (5, 3))

    def test_warmup_returns_parent_result(self):
        """warmup() must return the parent's result dict."""
        inferencer = self.ProtocolGenInferencer()
        result = asyncio.run(inferencer.warmup(self._make_data_list()))
        self.assertEqual(result["success"], 1)
        self.assertEqual(result["failed"], 0)

    def test_warmup_items_without_extra_params_passthrough(self):
        """Items without extra_params pass through unchanged."""
        inferencer = self.ProtocolGenInferencer()
        data_list = [{"prompt": "plain", "index": 0}]
        asyncio.run(inferencer.warmup(data_list))
        received = inferencer.warmup_received_data
        self.assertEqual(received[0], {"prompt": "plain", "index": 0})

    def test_warmup_non_dict_items_passthrough(self):
        """Non-dict items in data_list pass through as-is."""
        inferencer = self.ProtocolGenInferencer()
        data_list = ["not-a-dict", 42, None]
        asyncio.run(inferencer.warmup(data_list))
        received = inferencer.warmup_received_data
        self.assertEqual(received, ["not-a-dict", 42, None])

    def test_warmup_empty_data_list(self):
        """Empty data_list is handled without error."""
        inferencer = self.ProtocolGenInferencer()
        asyncio.run(inferencer.warmup([]))
        self.assertEqual(inferencer.warmup_received_data, [])


if __name__ == "__main__":
    unittest.main()