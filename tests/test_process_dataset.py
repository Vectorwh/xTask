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


if __name__ == "__main__":
    unittest.main()
