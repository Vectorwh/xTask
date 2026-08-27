import json
import tempfile
import unittest
from pathlib import Path

from xtask.benchmarks.aisbench.run_benchmark import process_dataset


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(token_ids)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True
    ):
        prompt = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            prompt += "<assistant>"
        return list(prompt) if tokenize else prompt


def sharegpt_sample(sample_id, conversations):
    return {
        "id": sample_id,
        "conversations": [
            {"from": role, "value": value} for role, value in conversations
        ],
    }


class ShareGPTDatasetProcessingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_fixed_length_and_requested_count_are_exact(self):
        source_data = [
            sharegpt_sample(
                "first",
                [
                    ("human", "hello"),
                    ("gpt", "hi"),
                    ("human", "tell me about today"),
                    ("gpt", "it is a good day"),
                ],
            ),
            sharegpt_sample(
                "second",
                [("human", "what should I cook"), ("gpt", "try noodles")],
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.json"
            output_path = root / "fixed.json"
            source_path.write_text(json.dumps(source_data), encoding="utf-8")

            process_dataset.process_sharegpt_dataset(
                self.tokenizer,
                str(source_path),
                str(output_path),
                input_len=80,
                num_prompts=5,
            )
            generated = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(generated), 5)
        self.assertEqual(len({sample["id"] for sample in generated}), 5)
        for sample in generated:
            self.assertEqual(
                process_dataset._sharegpt_prompt_length(
                    self.tokenizer, sample["conversations"]
                ),
                80,
            )
            self.assertEqual(sample["conversations"][-2]["from"], "human")
            self.assertEqual(sample["conversations"][-1]["from"], "gpt")

    def test_overlong_sample_drops_oldest_complete_rounds(self):
        sample = sharegpt_sample(
            "long",
            [
                ("human", "old question " * 8),
                ("gpt", "old answer " * 8),
                ("human", "recent question"),
                ("gpt", "recent answer"),
            ],
        )

        fixed = process_dataset.fix_sharegpt_sample(
            self.tokenizer, sample, target_len=60
        )

        self.assertEqual(len(fixed["conversations"]), 2)
        self.assertEqual(fixed["conversations"][-1]["value"], "recent answer")
        self.assertEqual(
            process_dataset._sharegpt_prompt_length(
                self.tokenizer, fixed["conversations"]
            ),
            60,
        )

    def test_target_smaller_than_chat_template_overhead_fails(self):
        sample = sharegpt_sample(
            "small", [("human", "question"), ("gpt", "answer")]
        )

        with self.assertRaisesRegex(ValueError, "chat-template overhead"):
            process_dataset.fix_sharegpt_sample(
                self.tokenizer, sample, target_len=5
            )

    def test_prefix_cache_groups_are_shared_distinct_and_seeded(self):
        source_data = [
            sharegpt_sample(
                "aurora",
                [
                    ("human", "Aurora deployment history begins with these details. "),
                    ("gpt", "Aurora history response records the completed first turn. "),
                    ("human", "What should the Aurora team verify before rollout?"),
                    ("gpt", "Verify capacity, health checks, and rollback readiness."),
                    ("human", "How should Aurora monitor the rollout?"),
                    ("gpt", "Monitor latency, errors, and resource usage."),
                ],
            ),
            sharegpt_sample(
                "borealis",
                [
                    ("human", "Borealis research history starts with a different subject. "),
                    ("gpt", "Borealis history response preserves the first completed turn. "),
                    ("human", "What evidence supports the Borealis conclusion?"),
                    ("gpt", "Use the reviewed experiment results and source records."),
                    ("human", "How should Borealis present the results?"),
                    ("gpt", "Present assumptions, results, and known limitations."),
                ],
            ),
            sharegpt_sample(
                "cirrus",
                [
                    ("human", "Cirrus source conversation provides additional suffix material. "),
                    ("gpt", "Cirrus source response completes the first source turn. "),
                    ("human", "What is the Cirrus follow-up question?"),
                    ("gpt", "The Cirrus follow-up answer is recorded here."),
                ],
            ),
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.json"
            output_path = root / "prefix.json"
            source_path.write_text(json.dumps(source_data), encoding="utf-8")
            original_source = source_path.read_text(encoding="utf-8")

            process_dataset.process_sharegpt_prefix_cache_dataset(
                self.tokenizer,
                str(source_path),
                str(output_path),
                input_len=240,
                prefix_len=32,
                prefix_groups=2,
                num_prompts=6,
                seed_requests=True,
            )
            generated = json.loads(output_path.read_text(encoding="utf-8"))
            source_unchanged = source_path.read_text(encoding="utf-8") == original_source

        self.assertEqual(len(generated), 6)
        self.assertTrue(source_unchanged)
        self.assertEqual(
            [sample["xtask_prefix_group"] for sample in generated[:2]], [0, 1]
        )
        self.assertTrue(all(sample["xtask_prefix_seed"] for sample in generated[:2]))
        self.assertTrue(all(not sample["xtask_prefix_seed"] for sample in generated[2:]))
        self.assertEqual(len({sample["id"] for sample in generated}), 6)
        self.assertTrue(
            all(sample["xtask_prefix_source"] == "dataset" for sample in generated)
        )
        self.assertTrue(
            all(
                process_dataset._sharegpt_prompt_length(
                    self.tokenizer, sample["conversations"]
                )
                == 240
                for sample in generated
            )
        )
        self.assertTrue(
            all(
                sample["conversations"][-2]["from"] == "human"
                and sample["conversations"][-1]["from"] == "gpt"
                for sample in generated
            )
        )
        self.assertTrue(
            all(
                "[xTask prefix group" not in message["value"]
                for sample in generated
                for message in sample["conversations"]
            )
        )
        source_by_id = {sample["id"]: sample for sample in source_data}
        for sample in generated:
            history_source = source_by_id[sample["xtask_prefix_history_source_id"]]
            history_messages = sample["xtask_prefix_history_turns"] * 2
            self.assertEqual(
                sample["conversations"][:history_messages],
                history_source["conversations"][:history_messages],
            )

        grouped = {
            group_id: [
                process_dataset._sharegpt_prompt_token_ids(
                    self.tokenizer, sample["conversations"]
                )
                for sample in generated
                if sample["xtask_prefix_group"] == group_id
            ]
            for group_id in (0, 1)
        }
        self.assertGreaterEqual(
            process_dataset._common_prefix_length(grouped[0]), 32
        )
        self.assertGreaterEqual(
            process_dataset._common_prefix_length(grouped[1]), 32
        )
        self.assertLess(
            process_dataset._common_prefix_length([grouped[0][0], grouped[1][0]]),
            32,
        )

    def test_packed_prefix_uses_longest_distinct_anchors_and_multiple_sources(self):
        source_data = []
        content_lengths = [70, 64, 58, 52, 46, 40, 34, 28, 22, 16, 14, 12]
        for index, content_length in enumerate(content_lengths):
            source_data.append(
                sharegpt_sample(
                    f"source-{index}",
                    [
                        ("human", chr(65 + index) * content_length),
                        ("gpt", chr(97 + index) * content_length),
                    ],
                )
            )

        self.assertTrue(
            all(
                process_dataset._packed_history_prefix_probe(
                    self.tokenizer, sample["conversations"]
                )[0]
                < 250
                for sample in source_data
            )
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.json"
            output_path = root / "packed-prefix.json"
            source_path.write_text(json.dumps(source_data), encoding="utf-8")
            original_source = source_path.read_text(encoding="utf-8")

            process_dataset.process_sharegpt_prefix_cache_dataset(
                self.tokenizer,
                str(source_path),
                str(output_path),
                input_len=500,
                prefix_len=250,
                prefix_groups=2,
                num_prompts=6,
                seed_requests=True,
                prefix_assembly="packed",
            )
            generated = json.loads(output_path.read_text(encoding="utf-8"))
            source_unchanged = (
                source_path.read_text(encoding="utf-8") == original_source
            )

        self.assertTrue(source_unchanged)
        self.assertEqual(len(generated), 6)
        self.assertEqual(
            [entry["xtask_prefix_group"] for entry in generated[:2]], [0, 1]
        )
        self.assertEqual(
            [entry["xtask_prefix_history_source_ids"][0] for entry in generated[:2]],
            ["source-0", "source-1"],
        )
        self.assertTrue(
            all(entry["xtask_prefix_assembly"] == "packed" for entry in generated)
        )
        self.assertTrue(
            all(
                len(entry["xtask_prefix_history_source_ids"]) > 1
                for entry in generated
            )
        )
        self.assertTrue(
            all(
                process_dataset._sharegpt_prompt_length(
                    self.tokenizer, entry["conversations"]
                )
                == 500
                for entry in generated
            )
        )

        grouped = {
            group_id: [
                process_dataset._sharegpt_prompt_token_ids(
                    self.tokenizer, entry["conversations"]
                )
                for entry in generated
                if entry["xtask_prefix_group"] == group_id
            ]
            for group_id in (0, 1)
        }
        self.assertEqual(process_dataset._common_prefix_length(grouped[0]), 250)
        self.assertEqual(process_dataset._common_prefix_length(grouped[1]), 250)
        self.assertTrue(
            all(entry["xtask_prefix_tail_truncated"] for entry in generated)
        )
        self.assertLess(
            process_dataset._common_prefix_length([grouped[0][0], grouped[1][0]]),
            250,
        )
        self.assertTrue(
            set(generated[0]["xtask_prefix_history_source_ids"]).isdisjoint(
                generated[1]["xtask_prefix_history_source_ids"]
            )
        )
        self.assertTrue(
            all(
                len(entry["xtask_prefix_suffix_source_ids"]) > 1
                for entry in generated
            )
        )

    def test_prefix_cache_rejects_prefix_not_smaller_than_input(self):
        source = sharegpt_sample(
            "only",
            [
                ("human", "A sufficiently long source history for validation."),
                ("gpt", "A sufficiently long source response for validation."),
                ("human", "final question"),
                ("gpt", "final answer"),
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.json"
            source_path.write_text(json.dumps([source]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prefix_len must be smaller"):
                process_dataset.process_sharegpt_prefix_cache_dataset(
                    self.tokenizer,
                    str(source_path),
                    str(root / "prefix.json"),
                    input_len=64,
                    prefix_len=64,
                    prefix_groups=1,
                    num_prompts=1,
                )

    def test_prefix_cache_rejects_more_groups_than_requests(self):
        source = sharegpt_sample("only", [("human", "question"), ("gpt", "answer")])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.json"
            source_path.write_text(json.dumps([source]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prefix_groups cannot exceed"):
                process_dataset.process_sharegpt_prefix_cache_dataset(
                    self.tokenizer,
                    str(source_path),
                    str(root / "prefix.json"),
                    input_len=64,
                    prefix_len=16,
                    prefix_groups=3,
                    num_prompts=2,
                )


if __name__ == "__main__":
    unittest.main()
