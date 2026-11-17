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

import json
import os
import tempfile
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()