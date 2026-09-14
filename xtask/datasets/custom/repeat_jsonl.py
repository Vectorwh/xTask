#!/usr/bin/env python3
"""Repeat a jsonl file until it contains exactly num_prompts records."""

import argparse
from pathlib import Path


def repeat_jsonl_records(records, num_prompts):
    if not records:
        raise ValueError("jsonl dataset is empty")
    if not isinstance(num_prompts, int) or isinstance(num_prompts, bool) or num_prompts <= 0:
        raise ValueError(f"num_prompts must be a positive integer, got {num_prompts!r}")
    multiplier = (num_prompts + len(records) - 1) // len(records)
    return (records * multiplier)[:num_prompts]


def load_jsonl_records(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(stripped + "\n")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prompts", required=True, type=int)
    args = parser.parse_args()
    records = load_jsonl_records(args.source)
    repeated = repeat_jsonl_records(records, args.num_prompts)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(repeated), encoding="utf-8")


if __name__ == "__main__":
    main()
