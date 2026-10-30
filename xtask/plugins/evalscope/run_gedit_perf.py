"""Qwen-Image-Edit 性能评测 wrapper 入口。

为什么需要 wrapper：evalscope 1.14.0 的 ``ApiRegistry`` 只在插件模块被
import 时通过 ``@register_api`` 装饰器完成注册，不做任何外部扫描；
``run_perf_benchmark`` 在内部按 ``--api`` 查表取类。因此必须先
``import gedit_api`` 再调用 ``run_perf_benchmark``，本模块即为该入口。

用法（与 evalscope perf CLI 参数基本同名，仅去掉了不需要的项）：

    python run_gedit_perf.py \
        --url http://127.0.0.1:8800/v1/image/generation \
        --model Qwen-Image-Edit \
        --dataset-path gedit_perf.jsonl \
        --number 10 --parallel 1 --warmup-num 1 \
        --outputs-dir /path/to/outputs

固定项（不可通过参数覆盖）：
    --dataset line_by_line   数据集逐行读取（每行一个 JSON dict）
    --api gedit_image_gen    自定义 API 插件名

注意：``gedit_image_gen`` 不在 evalscope 的
``_OPENAI_API_ENDPOINT_MAP`` 中，框架不会给 ``--url`` 自动追加端点
后缀，因此 ``--url`` 必须写完整路径（含 ``/v1/image/generation``）。
"""

import argparse
import os
import sys

# --- sys.path 自举：确保无论从哪里启动都能 import xtask.plugins ---
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (os.path.dirname(os.path.dirname(_HERE)), os.getcwd()):
    if os.path.isdir(os.path.join(_candidate, 'xtask')) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
        break

# 先 import 插件触发 @register_api('gedit_image_gen') 注册，
# 再 import run_perf_benchmark（其内部按 args.api 查询注册表）。
import xtask.plugins.evalscope.gedit_api  # noqa: F401  (import 即注册)

from evalscope.perf.main import run_perf_benchmark  # noqa: E402  (须在插件注册之后)

GEDIT_API_NAME = 'gedit_image_gen'
GEDIT_DATASET = 'line_by_line'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Qwen-Image-Edit perf benchmark wrapper (evalscope plugin entry).'
    )
    parser.add_argument('--url', type=str, required=True,
                        help='Full endpoint URL, e.g. http://host:port/v1/image/generation')
    parser.add_argument('--encoder-url', type=str, default=None,
                        help='Text encoder base URL for the two-stage protocol '
                             '(e.g. http://127.0.0.1:18007); defaults to the '
                             'GEDIT_ENCODER_URL env var, then http://127.0.0.1:18007')
    parser.add_argument('--model', type=str, required=True,
                        help='Served model name sent in the request body')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='JSONL dataset; one JSON dict per line (images are file paths)')
    parser.add_argument('--number', type=int, nargs='+', default=None,
                        help='Number of requests (same semantics as evalscope perf -n)')
    parser.add_argument('--parallel', type=int, nargs='+', default=1,
                        help='Concurrency (list allowed, same semantics as evalscope perf)')
    parser.add_argument('--rate', type=float, nargs='+', default=-1,
                        help='Request rate limit, -1 means unlimited')
    parser.add_argument('--warmup-num', type=float, default=0,
                        help='Warmup request count/ratio, excluded from metrics')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (also the default request seed when the '
                             'dataset line has none)')
    parser.add_argument('--extra-args', type=str, default='{}',
                        help='Extra evalscope Arguments fields as a JSON dict string')
    parser.add_argument('--outputs-dir', type=str, default='outputs',
                        help='Output directory for benchmark results')
    parser.add_argument('--name', type=str, default=None,
                        help='Run name used for the output sub-directory')
    parser.add_argument('--total-timeout', type=int, default=6 * 60 * 60,
                        help='Total HTTP timeout per request in seconds')
    parser.add_argument('--connect-timeout', type=int, default=None,
                        help='Connection timeout in seconds')
    parser.add_argument('--read-timeout', type=int, default=None,
                        help='Read timeout in seconds')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    import json

    extra = {}
    if args.extra_args and args.extra_args.strip():
        extra = json.loads(args.extra_args)
        if not isinstance(extra, dict):
            raise SystemExit('--extra-args must be a JSON object')

    run_config = {
        'url': args.url,
        'model': args.model,
        'api': GEDIT_API_NAME,          # 固定，防止误传 openai 等导致端点改写
        'dataset': GEDIT_DATASET,       # 固定 line_by_line
        'dataset_path': args.dataset_path,
        'parallel': args.parallel,
        'rate': args.rate,
        'warmup_num': args.warmup_num,
        'seed': args.seed,
        'outputs_dir': args.outputs_dir,
        'name': args.name,
        'total_timeout': args.total_timeout,
        'connect_timeout': args.connect_timeout,
        'read_timeout': args.read_timeout,
        # 两段式插件在 build_request 里先调 encoder；evalscope 的 connect_test
        # 用无图 'hello' 探活会绕过 encoder 生成无效请求，关闭它（调度器
        # 已有独立 /health 探活）。
        'no_test_connection': True,
    }
    # encoder 地址：CLI > 环境变量 > 插件内缺省。CLI 传入时注入子进程环境，
    # 插件经 os.environ 读取。
    if args.encoder_url:
        os.environ['GEDIT_ENCODER_URL'] = args.encoder_url
    if args.number is not None:
        run_config['number'] = args.number

    # --extra-args 里混有两类键：
    # 1. evalscope Arguments 合法字段（如 tokenizer_path）——直接并入 run_config；
    # 2. 请求体级参数（size / num_inference_steps / guidance_scale /
    #    true_cfg_scale 等）——Arguments 是 extra='forbid' 的 pydantic 模型，
    #    传入会直接 ValidationError。这些参数以数据集行为准（行内缺省时
    #    build_request 用插件默认值），这里显式剔除并仅打日志说明。
    from evalscope.perf.arguments import Arguments as _Arguments

    _valid_keys = set(getattr(_Arguments, 'model_fields', {}).keys())
    _body_level_keys = [key for key in extra if key not in _valid_keys]
    if _body_level_keys:
        print(
            f'[run_gedit_perf] dropping body-level keys from --extra-args '
            f'(dataset lines take precedence): {sorted(_body_level_keys)}'
        )
    extra = {key: value for key, value in extra.items() if key in _valid_keys}
    run_config.update(extra)

    assert run_config['api'] == GEDIT_API_NAME, 'api must be gedit_image_gen'
    assert run_config['dataset'] == GEDIT_DATASET, 'dataset must be line_by_line'

    run_perf_benchmark(run_config)


if __name__ == '__main__':
    main()
