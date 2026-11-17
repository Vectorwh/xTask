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
            self.assertEqual(ov["temperature"], 0)

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
                "temperature": 0,
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
            self.assertEqual(model_dict["temperature"], 0)
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


if __name__ == "__main__":
    unittest.main()