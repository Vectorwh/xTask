"""AISBench dataset config for protocol consistency & constrained decoding tests.

Loads a JSONL file where each line has:
    - question: the user prompt
    - max_out_len: max output tokens for this case
    - extra_params: per-sample protocol/constrained-decoding params
      (tools, tool_choice, response_format, guided_json, guided_regex,
       guided_grammar, extra_body)
    - ground_truth: reference dict for three-layer evaluation
      (expected_finish_reason, require_reasoning, schema, regex, tool_calls,
       answer, answer_regex)

The dataset path is overridden at runtime by xTask's
create_dataset_config_override() to point at the user-provided JSONL.
"""

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import CustomDataset

from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
    ProtocolAndConstrainedEvaluator,
)


class ProtocolConstrainedDataset(CustomDataset):
    """CustomDataset variant that preserves dict-typed extra_params and
    ground_truth fields instead of stringifying them.

    AISBench's base CustomDataset may coerce complex fields; this subclass
    re-parses them from the raw line to guarantee the dict structure.
    """

    def load(self):
        """Load JSONL and ensure extra_params / ground_truth stay as dicts."""
        import json as _json
        data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError as exc:
                    raise ValueError(
                        f"protocol_constrained JSONL parse error at line {line_no}: {exc}"
                    ) from exc
                # Ensure dict types survive the pipeline.
                if "extra_params" not in row or not isinstance(row["extra_params"], dict):
                    row["extra_params"] = row.get("extra_params") or {}
                if "ground_truth" not in row or not isinstance(row["ground_truth"], dict):
                    row["ground_truth"] = row.get("ground_truth") or {}
                data.append(row)
        return data


class ProtocolConstrainedInferencer(GenInferencer):
    """GenInferencer variant that explicitly forwards extra_params to the
    model's generate() call.

    The standard GenInferencer passes remaining data fields as **kwargs;
    this subclass makes the extra_params forwarding explicit and also
    strips ground_truth from the kwargs sent to the model (it should only
    go to the evaluator).
    """

    async def infer(self, data, session):
        import copy as _copy
        data = _copy.deepcopy(data)
        # ground_truth must not be sent to the model; it's for the evaluator.
        data.pop("ground_truth", None)
        return await super().infer(data, session)


protocol_constrained_reader_cfg = dict(
    input_columns=['question', 'max_out_len', 'extra_params'],
    output_column='ground_truth',
)

protocol_constrained_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{question}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ProtocolConstrainedInferencer),
)

protocol_constrained_eval_cfg = dict(
    evaluator=dict(type=ProtocolAndConstrainedEvaluator),
    pred_role='BOT',
)

protocol_constrained_datasets = [
    dict(
        abbr='protocol_constrained',
        type=ProtocolConstrainedDataset,
        path='ais_bench/datasets/protocol_constrained/constrained_decoding.jsonl',
        meta_path='',
        reader_cfg=protocol_constrained_reader_cfg,
        infer_cfg=protocol_constrained_infer_cfg,
        eval_cfg=protocol_constrained_eval_cfg,
    )
]