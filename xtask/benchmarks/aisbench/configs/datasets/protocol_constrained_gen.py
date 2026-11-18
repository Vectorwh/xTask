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

NOTE: ProtocolConstrainedDataset is defined in
xtask.benchmarks.aisbench.protocol_dataset (a regular module), NOT here.
mmengine's _parse_lazy_import forbids subclassing lazily-imported names at
module level (class Foo(Bar) triggers LazyAttr.__call__ -> RuntimeError).
Referencing it as a type= value below is lazy-import-safe.

ProtocolGenInferencer (a GenInferencer subclass) is used instead of the
stock GenInferencer because the stock get_data_list() only emits
prompt/data_abbr/index/max_out_len/gold -- the extra_params column would
never reach do_request(**data) and thus never reach
model.generate(**args).  ProtocolGenInferencer.get_data_list() injects
extra_params into each data_list entry so it flows through the standard
async pipeline to ProtocolAwareAPIChat.get_request_body(**args).

The reader_cfg separates input_columns (question, max_out_len,
extra_params) from output_column (ground_truth), so ground_truth is never
sent to the model.
"""

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

from xtask.benchmarks.aisbench.protocol_constrained_evaluator import (
    ProtocolAndConstrainedEvaluator,
)
from xtask.benchmarks.aisbench.protocol_dataset import (
    ProtocolConstrainedDataset,
)
from xtask.benchmarks.aisbench.protocol_inferencer import (
    ProtocolGenInferencer,
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
    inferencer=dict(type=ProtocolGenInferencer),
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