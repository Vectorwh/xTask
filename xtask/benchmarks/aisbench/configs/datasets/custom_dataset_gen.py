from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator


custom_dataset_reader_cfg = dict(
    input_columns=['question', 'max_out_len'],
    output_column=None,
)

custom_dataset_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{question}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

custom_dataset_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
)

custom_dataset_datasets = [
    dict(
        abbr='custom_dataset',
        type=CustomDataset,
        path='ais_bench/datasets/custom/qa.jsonl',
        meta_path='',
        reader_cfg=custom_dataset_reader_cfg,
        infer_cfg=custom_dataset_infer_cfg,
        eval_cfg=custom_dataset_eval_cfg,
    )
]
