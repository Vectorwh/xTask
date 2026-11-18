#!/usr/bin/env python3
"""Convert open-source benchmark datasets to protocol_constrained JSONL format.

Supported sources:
    - bfcl:           Berkeley Function Calling Leaderboard (tool call evaluation)
    - jsonschemabench: JSONSchemaBench (JSON Schema constrained generation)
    - gsm8k:          GSM8K math problems (reasoning content separation)

Each converter is a pure function: takes raw source data (list of dicts),
returns a list of protocol_constrained-format dicts. This makes them
unit-testable without file I/O.

Usage:
    python tools/convert_open_protocol_datasets.py --source bfcl \\
        --input /path/to/bfcl_data.json --output /path/to/output.jsonl
    python tools/convert_open_protocol_datasets.py --source jsonschemabench \\
        --input /path/to/jsb.jsonl --output /path/to/output.jsonl
    python tools/convert_open_protocol_datasets.py --source gsm8k \\
        --input /path/to/gsm8k.jsonl --output /path/to/output.jsonl
"""

import argparse
import json
import os
import re


# ---------------------------------------------------------------------------
# BFCL → protocol_constrained (tool call)
# ---------------------------------------------------------------------------

def convert_bfcl(rows, max_out_len=2048):
    """Convert BFCL v3 rows to tool-call protocol cases.

    BFCL row format:
        {"id": "...", "question": "...", "function": [...], "answer": [...]}

    Output ground_truth:
        {"expected_finish_reason": "tool_calls", "tool_calls": [...]}
    """
    cases = []
    for row in rows:
        functions = row.get("function") or row.get("functions") or []
        if not functions:
            continue
        tools = []
        for func in functions:
            if isinstance(func, dict) and "name" in func:
                tools.append({"type": "function", "function": func})
            elif isinstance(func, dict) and "function" in func:
                tools.append(func)
        if not tools:
            continue

        answer = row.get("answer") or []
        expected_calls = []
        for ans in answer:
            if isinstance(ans, dict) and "name" in ans:
                expected_calls.append({
                    "function": {
                        "name": ans["name"],
                        "arguments": ans.get("arguments", ans.get("parameters", {})),
                    }
                })

        gt = {"expected_finish_reason": "tool_calls"}
        if expected_calls:
            gt["tool_calls"] = expected_calls

        cases.append({
            "id": f"bfcl_{row.get('id', len(cases))}",
            "question": row["question"],
            "max_out_len": max_out_len,
            "extra_params": {"tools": tools},
            "ground_truth": gt,
        })
    return cases


# ---------------------------------------------------------------------------
# JSONSchemaBench → protocol_constrained (json_schema)
# ---------------------------------------------------------------------------

def convert_jsonschemabench(rows, max_out_len=4096):
    """Convert JSONSchemaBench rows to JSON Schema protocol cases.

    JSONSchemaBench row format (varies by subset):
        {"id": "...", "prompt": "...", "schema": {...}}
        or
        {"prompt_id": "...", "prompt": "...", "schema": {...}}
    """
    cases = []
    for i, row in enumerate(rows):
        schema = row.get("schema") or row.get("json_schema")
        if not schema or not isinstance(schema, dict):
            continue
        prompt = row.get("prompt") or row.get("question") or row.get("instruction", "")
        if not prompt:
            continue
        case_id = row.get("id") or row.get("prompt_id") or str(i)

        cases.append({
            "id": f"jsonschemabench_{case_id}",
            "question": prompt,
            "max_out_len": max_out_len,
            "extra_params": {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"jsb_{case_id}",
                        "strict": True,
                        "schema": schema,
                    },
                },
            },
            "ground_truth": {"expected_finish_reason": "stop", "schema": schema},
        })
    return cases


# ---------------------------------------------------------------------------
# GSM8K → protocol_constrained (reasoning separation)
# ---------------------------------------------------------------------------

def _extract_gsm8k_answer(answer_text):
    """Extract the final numeric answer from GSM8K answer text.

    GSM8K answers end with '#### <number>'.
    """
    match = re.search(r"####\s*([\d,.\-]+)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


def convert_gsm8k(rows, max_out_len=4096):
    """Convert GSM8K rows to reasoning-separation protocol cases.

    GSM8K row format:
        {"question": "...", "answer": "...\\n#### 42"}

    These cases test that reasoning_content is properly separated from content.
    Ground truth includes require_reasoning=True and answer/answer_regex.
    """
    cases = []
    for i, row in enumerate(rows):
        question = row.get("question", "")
        answer_text = row.get("answer", "")
        if not question or not answer_text:
            continue
        final_answer = _extract_gsm8k_answer(answer_text)
        if final_answer is None:
            continue

        cases.append({
            "id": f"gsm8k_{row.get('id', i)}",
            "question": question,
            "max_out_len": max_out_len,
            "extra_params": {},
            "ground_truth": {
                "expected_finish_reason": "stop",
                "require_reasoning": True,
                "answer": final_answer,
                "answer_regex": re.escape(final_answer),
            },
        })
    return cases


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_source_rows(input_path):
    """Load source data from a JSON array or JSONL file."""
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return rows
    if content[0] == "{":
        # JSONL: one JSON object per line
        for line in content.split("\n"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    elif content[0] == "[":
        data = json.loads(content)
        rows = data if isinstance(data, list) else [data]
    else:
        data = json.loads(content)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            for key in ("data", "rows", "questions", "test"):
                if key in data and isinstance(data[key], list):
                    rows = data[key]
                    break
            if not rows:
                rows = [data]
    return rows


def write_jsonl(cases, output_path):
    """Write cases to a JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} cases to {output_path}")


CONVERTERS = {
    "bfcl": convert_bfcl,
    "jsonschemabench": convert_jsonschemabench,
    "gsm8k": convert_gsm8k,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert open-source benchmark datasets to "
                    "protocol_constrained JSONL format")
    parser.add_argument("--source", required=True, choices=sorted(CONVERTERS),
                        help="Source dataset type")
    parser.add_argument("--input", required=True,
                        help="Path to source data file (JSON or JSONL)")
    parser.add_argument("--output", required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--max-out-len", type=int, default=None,
                        help="Override default max_out_len for all cases")
    args = parser.parse_args()

    converter = CONVERTERS[args.source]
    rows = load_source_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    kwargs = {}
    if args.max_out_len is not None:
        kwargs["max_out_len"] = args.max_out_len

    cases = converter(rows, **kwargs)
    skipped = len(rows) - len(cases)
    print(f"Converted {len(cases)} cases ({skipped} skipped)")

    write_jsonl(cases, args.output)


if __name__ == "__main__":
    main()