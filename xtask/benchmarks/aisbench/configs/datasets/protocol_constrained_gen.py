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

NOTE: ProtocolConstrainedDataset / ProtocolConstrainedInferencer are defined
in xtask.benchmarks.aisbench.protocol_dataset (a regular module), NOT here.
mmengine's _parse_lazy_import forbids subclassing lazily-imported names at
module level (class Foo(Bar) triggers LazyAttr.__call__ -> RuntimeError).
Referencing them as type= values below is lazy-import-safe.
"""

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
    ProtocolAndConstrainedEvaluator,
)
from xtask.benchmarks.aisbench.protocol_dataset import (
    ProtocolConstrainedDataset,
    ProtocolConstrainedInferencer,
)

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