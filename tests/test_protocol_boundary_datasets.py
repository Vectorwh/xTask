"""Tests for protocol boundary dataset generation and conversion scripts."""

import json
import os
import re
import sys

import pytest

# Make tools/ importable
_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from generate_protocol_boundary_datasets import (  # noqa: E402
    generate_adversarial_cases,
    generate_grammar_cases,
    generate_regex_cases,
    write_jsonl,
)
from convert_open_protocol_datasets import (  # noqa: E402
    _extract_gsm8k_answer,
    convert_bfcl,
    convert_gsm8k,
    convert_jsonschemabench,
    load_source_rows,
)

REQUIRED_KEYS = {"id", "question", "max_out_len", "extra_params", "ground_truth"}
DISPATCH_KEYS = (
    "regex", "grammar", "schema", "tool_calls",
    "answer", "answer_regex", "expected_finish_reason",
)


# ---------------------------------------------------------------------------
# Regex boundary cases
# ---------------------------------------------------------------------------

class TestRegexCases:
    def test_count(self):
        assert len(generate_regex_cases()) == 20

    def test_required_keys(self):
        for case in generate_regex_cases():
            assert REQUIRED_KEYS <= set(case.keys()), f"{case['id']} missing keys"

    def test_regex_compilable(self):
        for case in generate_regex_cases():
            re.compile(case["ground_truth"]["regex"])

    def test_ids_unique(self):
        cases = generate_regex_cases()
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids))

    def test_max_out_len_floor(self):
        for case in generate_regex_cases():
            assert case["max_out_len"] >= 2048

    def test_dispatch_keys(self):
        for case in generate_regex_cases():
            gt = case["ground_truth"]
            assert gt.get("regex"), f"{case['id']} missing regex"
            assert "expected_finish_reason" in gt


# ---------------------------------------------------------------------------
# Grammar boundary cases
# ---------------------------------------------------------------------------

class TestGrammarCases:
    def test_count(self):
        assert len(generate_grammar_cases()) == 20

    def test_required_keys(self):
        for case in generate_grammar_cases():
            assert REQUIRED_KEYS <= set(case.keys())

    def test_grammar_nonempty(self):
        for case in generate_grammar_cases():
            assert case["ground_truth"]["grammar"] is True
            assert case["extra_params"]["guided_grammar"]

    def test_answer_regex_compilable(self):
        for case in generate_grammar_cases():
            re.compile(case["ground_truth"]["answer_regex"])

    def test_ids_unique(self):
        cases = generate_grammar_cases()
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Adversarial boundary cases
# ---------------------------------------------------------------------------

class TestAdversarialCases:
    def test_count(self):
        assert len(generate_adversarial_cases()) == 30

    def test_required_keys(self):
        for case in generate_adversarial_cases():
            assert REQUIRED_KEYS <= set(case.keys())

    def test_ids_unique(self):
        cases = generate_adversarial_cases()
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids))

    def test_dispatch_compatible(self):
        for case in generate_adversarial_cases():
            gt = case["ground_truth"]
            assert any(gt.get(k) for k in DISPATCH_KEYS), \
                f"{case['id']} has no dispatchable check"

    def test_tag_collision_count(self):
        cases = generate_adversarial_cases()
        assert len([c for c in cases if c["id"].startswith("boundary_adv_tag_")]) == 5

    def test_escaping_count(self):
        cases = generate_adversarial_cases()
        assert len([c for c in cases if c["id"].startswith("boundary_adv_escape_")]) == 8

    def test_tool_choice_count(self):
        cases = generate_adversarial_cases()
        assert len([c for c in cases if "tool_choice" in c["extra_params"]]) == 9

    def test_minimal_count(self):
        cases = generate_adversarial_cases()
        assert len([c for c in cases if c["id"].startswith("boundary_adv_minimal_")]) == 4

    def test_deep_nesting_count_and_depth(self):
        cases = generate_adversarial_cases()
        deep = [c for c in cases if c["id"].startswith("boundary_adv_deep_")]
        assert len(deep) == 4
        for case in deep:
            schema = case["ground_truth"]["schema"]
            depth = 0
            cur = schema
            while isinstance(cur, dict) and cur.get("properties"):
                depth += 1
                key = list(cur["properties"].keys())[0]
                cur = cur["properties"][key]
                if isinstance(cur, dict) and cur.get("type") == "array":
                    cur = cur.get("items", {})
            assert depth >= 5, f"{case['id']} nesting depth {depth} < 5"


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------

class TestWriteJsonl:
    def test_write_and_read_back(self, tmp_path):
        cases = generate_regex_cases()[:3]
        filepath = str(tmp_path / "test.jsonl")
        write_jsonl(cases, filepath)
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert REQUIRED_KEYS <= set(obj.keys())


# ---------------------------------------------------------------------------
# BFCL conversion
# ---------------------------------------------------------------------------

class TestConvertBfcl:
    _WEATHER_FUNC = {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }

    def test_basic_conversion(self):
        rows = [{
            "id": "test_1",
            "question": "What's the weather?",
            "function": [self._WEATHER_FUNC],
            "answer": [{"name": "get_weather", "arguments": {"city": "NYC"}}],
        }]
        cases = convert_bfcl(rows)
        assert len(cases) == 1
        case = cases[0]
        assert case["id"] == "bfcl_test_1"
        assert case["extra_params"]["tools"][0]["type"] == "function"
        assert case["ground_truth"]["expected_finish_reason"] == "tool_calls"
        assert case["ground_truth"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_skip_no_functions(self):
        assert convert_bfcl([{"id": "x", "question": "hi", "function": []}]) == []

    def test_nested_tool_format(self):
        rows = [{
            "id": "t2",
            "question": "Calculate",
            "function": [{"type": "function", "function": {
                "name": "calc", "parameters": {"type": "object"}}}],
            "answer": [],
        }]
        cases = convert_bfcl(rows)
        assert len(cases) == 1
        assert cases[0]["extra_params"]["tools"][0]["function"]["name"] == "calc"
        # No expected_calls when answer is empty
        assert "tool_calls" not in cases[0]["ground_truth"]


# ---------------------------------------------------------------------------
# JSONSchemaBench conversion
# ---------------------------------------------------------------------------

class TestConvertJsonschemabench:
    _SCHEMA = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def test_basic_conversion(self):
        rows = [{"id": "p1", "prompt": "Generate a person", "schema": self._SCHEMA}]
        cases = convert_jsonschemabench(rows)
        assert len(cases) == 1
        case = cases[0]
        assert case["id"] == "jsonschemabench_p1"
        assert case["ground_truth"]["schema"] == self._SCHEMA
        assert case["extra_params"]["response_format"]["type"] == "json_schema"

    def test_skip_no_schema(self):
        assert convert_jsonschemabench([{"id": "x", "prompt": "hi"}]) == []

    def test_skip_non_dict_schema(self):
        assert convert_jsonschemabench([{"id": "x", "prompt": "hi", "schema": "str"}]) == []


# ---------------------------------------------------------------------------
# GSM8K conversion
# ---------------------------------------------------------------------------

class TestConvertGsm8k:
    def test_extract_answer(self):
        assert _extract_gsm8k_answer("reasoning\n#### 42") == "42"
        assert _extract_gsm8k_answer("#### 1,234.5") == "1234.5"
        assert _extract_gsm8k_answer("#### -7") == "-7"
        assert _extract_gsm8k_answer("no marker") is None

    def test_basic_conversion(self):
        rows = [{"question": "What is 2+2?", "answer": "2+2=4\n#### 4"}]
        cases = convert_gsm8k(rows)
        assert len(cases) == 1
        case = cases[0]
        assert case["ground_truth"]["require_reasoning"] is True
        assert case["ground_truth"]["answer"] == "4"
        assert case["ground_truth"]["answer_regex"] == "4"

    def test_skip_no_marker(self):
        assert convert_gsm8k([{"question": "hi", "answer": "no marker"}]) == []


# ---------------------------------------------------------------------------
# load_source_rows
# ---------------------------------------------------------------------------

class TestLoadSourceRows:
    def test_jsonl(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        assert len(load_source_rows(str(p))) == 2

    def test_json_array(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text('[{"a": 1}, {"b": 2}]', encoding="utf-8")
        assert len(load_source_rows(str(p))) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "e.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_source_rows(str(p)) == []