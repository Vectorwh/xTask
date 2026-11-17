# xLLM YAML 用例写作指导

写一个用例时，按下面四步拼起来即可：

1. 先写模型和数据集。
2. 再选一个 benchmark：AISBench、EvalScope、VBench、GEdit 图片编辑（`image_edit` 性能 / AISBench gedit 精度），或 ASR 原始评测（`asr` / `legacy_infer`）。
3. 再选一种测试：性能 `perf`，或精度 `acc`（VBench 视频质量与 AISBench SWE-bench 也写 `acc`，由 `benchmark.benchmark_type` 区分）；ASR 原始评测推荐 `metrics: acc,perf`，一次请求同时产出质量和性能结果。
4. 最后选部署：单机、普通多机、普通实例统一路由、PD 分离，或用 `external_service` 对接已有服务；需要搜索时再加搜索段。

## 1. 可参考示例

可直接参考的独立示例放在 `tests/e2e/xllm/examples/yaml_authoring/`，不会修改或替换旧用例。

| 目的 | 示例 |
| --- | --- |
| 单机 AISBench 性能 | `01_single_aisbench_perf.yaml` |
| 单机 EvalScope 性能 | `02_single_evalscope_perf.yaml` |
| 单机 AISBench 精度 | `03_single_aisbench_acc.yaml` |
| 单机 EvalScope 精度 | `04_single_evalscope_acc.yaml` |
| 普通多机混部性能/精度 | `05_multihost_aisbench_perf.yaml`、`06_multihost_aisbench_acc.yaml` |
| PD 分离性能/精度 | `07_pd_aisbench_perf.yaml`、`08_pd_aisbench_acc.yaml` |
| 单机并发摸高 | `09_single_auto_assess.yaml` |
| 单机 exhaustive 搜参 | `10_single_para_set_exhaustive.yaml` |
| 单机 TPE 搜参 | `11_single_para_set_tpe.yaml` |
| 多机并发摸高 | `12_multihost_auto_assess.yaml` |
| 多机普通搜参 | `13_multihost_para_set.yaml` |
| 多机 PD 拓扑搜参 | `14_pd_topology_para_set.yaml` |
| PD 并发摸高 | `15_pd_auto_assess.yaml` |
| PD 固定拓扑 exhaustive 参数搜索 | `16_pd_para_set_exhaustive.yaml` |
| 单机 AISBench ShareGPT 默认多轮模式性能测试 | `17_single_aisbench_sharegpt_perf.yaml` |
| 单机 AISBench ShareGPT `last` 模式固定输入长度性能测试 | `18_single_aisbench_sharegpt_last_fixed_perf.yaml` |
| 普通服务同节点多小实例混部 | `19_samehost_mixed_aisbench_perf.yaml` |
| PD 多逻辑实例、同逻辑实例多 node 共节点部署 | `20_pd_samehost_multi_instance_perf.yaml` |
| 多个普通独立实例通过 xllm-service 统一分发 | `21_service_routed_samehost_multi_instance_perf.yaml` |
| 单机 AISBench ShareGPT 多组 Prefix Cache 性能测试 | `22_single_aisbench_sharegpt_prefix_cache_perf.yaml` |
| 单机 AISBench ShareGPT `packed` 多组 Prefix Cache 长上下文性能测试 | `23_single_aisbench_sharegpt_prefix_cache_packed_perf.yaml` |
| 单机 AISBench GPQA Diamond 精度测试 | `24_single_aisbench_gpqa_acc.yaml` |
| 单机 Wan2.2 I2V + VBench 质量评测 | `25_single_vbench_wan_i2v.yaml` |
| 外部已有服务 + AISBench 性能 | `26_external_aisbench_perf.yaml` |
| 外部已有 Wan2.2 + VBench | `27_external_vbench_wan_i2v.yaml` |
| 单机起 xLLM + AISBench SWE-bench（x86 Agent/打分） | `28_single_aisbench_swebench.yaml` |
| 外部已有服务 + AISBench SWE-bench | `29_external_aisbench_swebench.yaml` |
| 单机 Wan2.2 I2V + VBench mini 子集（107 条采样） | `30_single_vbench_wan_i2v_mini.yaml` |
| DeepSeek V4 + AISBench SWE-bench（verified_mini 单条 filter） | `31_single_aisbench_swebench_dsv4.yaml` |
| DeepSeek V4 + AISBench SWE-bench（3 条 Django filter，含代理/重试开关） | `32_single_aisbench_swebench_dsv4_django3.yaml` |
| DeepSeek V4 + AISBench SWE-bench（verified_mini 全量） | `33_single_aisbench_swebench_dsv4_verified_mini.yaml` |
| DeepSeek V4 + AISBench SWE-bench（verified_mini 10 条快集） | `35_single_aisbench_swebench_dsv4_verified_mini10.yaml` |
| 单机 Qwen-Image-Edit gedit 性能（`image_edit` + 自服务 DiT/Encoder） | `36_single_image_edit_gedit_perf.yaml` |
| 单机 Qwen-Image-Edit gedit 精度（AISBench LLM-judge + 自服务三服务） | `37_single_aisbench_gedit_acc.yaml` |
| 单机 JoyAI-Image-Edit-Plus gedit 性能（`image_edit`，4 卡 DiT sp2*cfg2） | `38_single_image_edit_joyai_perf.yaml` |
| 单机 JoyAI-Image-Edit-Plus gedit 精度（AISBench LLM-judge + 自服务三服务） | `39_single_aisbench_joyai_acc.yaml` |
| 外部 JoyaiASR + 原始客户端，完整数据单轮精度/性能 | `40_external_joyai_asr_legacy_acc_perf.yaml` |
| 外部 JoyaiASR 两轮重复评测，验证日期归档和报表 | `41_external_joyai_asr_legacy_repeats.yaml` |
| 普通 Qwen3.6-35B-A3B AISBench perf 回归抽测 | `42_single_qwen36_35b_a3b_perf_regression.yaml` |
| 外部已有服务 + AISBench GPQA 精度，endpoint 用**域名**（验证走 AISBench `url` 字段） | `43_external_aisbench_gpqa_acc_domain.yaml` |
| PD 场景 `startup_environment` 启动环境注入示例（字段演示，非新场景） | `28_startup_environment.yaml` |

这些示例使用工程测试环境中的模型路径和 IP。换环境时，优先改 `model_path`、`host`、`master_host`、卡号和基线；旧 YAML 不需要跟着改。

ASR 自部署基础用例另放在 `tests/e2e/xllm/joyai/joyai_asr/`：
`joyai_asr_xllm_tp2_legacy.yaml` 由 xTask 部署 TP2 服务；对接已有服务可参考
40/41。三个 ASR 用例的区别与配置见第 3 节“JoyAI-ASR 原始评测”。

## 2. 顶层公共字段

```yaml
case_name: qwen3_8b_xllm_perf_i1024_o1024_bs1
engine_type: xllm
model_path: /export/home/models/Qwen3-8B
served_model_name: Qwen3-8B
metrics: perf                 # perf=性能，acc=精度；VBench、SWE-bench 新格式也写 acc，由 benchmark 块区分
dataset_name: gsm8k
dataset_path: ''
dataset_format: json
communication_backend: hccl
userid: 1048
serving_wait_time: 10
benchmark: {}
deployment: {}
```

先看这些字段的完整含义。表里的“必填”是指新写用例时应当明确填写，并不表示代码对每个字段都做了强制校验。

| 字段 | 新用例是否填写 | 通俗解释 | 当前行为和注意事项 |
| --- | --- | --- | --- |
| `case_name` | 是 | 用例的唯一名字 | 建议与 YAML 文件名相同，只使用字母、数字、下划线和短横线。搜参生成的子用例会在它后面追加 `_sg1`、`_sg2`。 |
| `engine_type` | 是 | 要启动哪个推理引擎 | 本文只讲 `xllm`。写成其他值会走其他引擎流程。 |
| `model_path` | 自部署时是 | 模型权重在宿主机上的目录 | 所有参与启动 xLLM 的服务器都要有这个目录。工程不会自动同步整个模型目录；容器内会通过路径映射访问它。外部 ASR `legacy_infer` 客户端不加载模型权重或 tokenizer，可不填。 |
| `served_model_name` | 是 | 请求接口里传给 `model` 的名字 | AISBench 和 EvalScope 都会使用它。它可以与目录名不同，但必须是当前 xLLM 服务接受的模型名。 |
| `metrics` | 是 | 选择性能还是精度大类 | `perf` 表示性能，`acc` 表示精度。VBench 视频质量用例与 AISBench SWE-bench 新格式同样写 `acc`，实际执行分支由 `benchmark.benchmark_type`（路由依据）和 `benchmark` 下的配置块共同决定（见第 3、8 节）。`swebench` 只作为旧写法兼容保留。普通用例建议一次只写一种；ASR `legacy_infer` 推荐 `acc,perf`，联合结果放在 `benchmark.asr`，见第 3 节。 |
| `dataset_name` | 是 | benchmark 使用的数据集或数据生成器名称 | AISBench 性能常用 `gsm8k`、`syn`、`sharegpt`；EvalScope 精度直接把它传给 `--datasets`，性能传给 `--dataset`。 |
| `dataset_path` | 按数据集需要 | 顶层数据集路径 | AISBench 配置了本地路径时会生成用例级数据集配置，优先使用 `benchmark.perf/acc.dataset_path`，否则使用此处路径；EvalScope 精度更常在 `dataset_args` 中指定。gedit 用例（36/37）不写 `dataset_path` 时使用内置默认数据集路径，见第 3 节 GEdit 小节。多机资源同步建议填写 `/xllm_test_060/...` 容器路径。 |
| `dataset_format` | 建议 | 数据文件格式说明 | 常写 `json`。它主要作为任务元数据保留，具体解析方式仍由 benchmark 和数据集配置决定。 |
| `communication_backend` | 多卡建议 | 多卡通信后端 | NPU 多卡通常写 `hccl`。 |
| `hccl_if_port` | 端口冲突时填写 | HCCL 通信使用的基础端口 | xLLM 启动模板会导出为 `HCCL_IF_BASE_PORT`，默认 43439。它是物理 runtime/node 级配置：同一 host 上的多个 runtime 必须不同，不同 host 可以复用。顶层值只是默认值，可在普通 `instances[]` 或 PD `nodes[]` 中覆盖。 |
| `progress_conn_port` | 端口冲突时填写 | xLLM 多进程主节点协调端口 | 启动模板用它组成 `MASTER_NODE_ADDR`，默认 9590。它是逻辑实例级配置：普通部署整组共享；统一路由和 PD 分别在各 `service_instances[]`、`pd_instances[]` 配置，同组所有 node 必须一致；多个逻辑组的主节点在同一 host 时必须使用不同值。 |
| `userid` | 建议 | 结果文件归属用户的历史配置 | 常沿用环境中的 UID，例如 `1048`。部分当前流程直接取实际运行用户 UID，因此不要把它当作容器权限问题的唯一控制项。 |
| `serving_wait_time` | 建议 | 服务启动命令发出后额外等待多少秒 | 它不是完整的服务启动超时。多机还会做实例 readiness 和统一 endpoint 健康检查。 |
| `benchmark_timeout` | 长评测按需 | 单次 benchmark 执行的容器内超时秒数 | 未写时沿用引擎配置默认（当前 7200 秒）。VBench 全量/mini 等长评测用例可显式调大，例如 30 号用例的 `benchmark_timeout: 86400`。 |
| `benchmark` | 是 | benchmark 类型及性能/精度/ASR/视频质量参数 | 详见第 3、4、8 节。 |
| `deployment` | 自部署时是 | xLLM 用哪些服务器、卡和端口启动 | 详见第 4 节。`external_service` 开启时拓扑被忽略，可不填，见第 9 节。 |
| `external_service` | 否 | 是否对接已有服务、不由 xTask 启动 xLLM | 默认关闭。`true` 或带 `endpoint` 的 mapping 进入外部模式；不写或 `false` 保持现有端到端路径。 |
| `assess_test_param` | 仅搜参 | 哪些字段需要搜索，以及每个字段有哪些候选值 | 只在 `--assess_test_type para_set` 下使用。 |
| `assess_search` | 搜参建议 | 选择 exhaustive 或 TPE，并配置目标与约束 | 不写时，`para_set` 默认按 exhaustive 全量组合执行。 |
| `assess_topologies` | 仅 PD 拓扑搜索 | 预先定义多个完整、合法的 PD 拓扑 | 不能用于普通 `deployment.instances`。 |

### `--host_addr` 到底会改什么

命令行的 `--host_addr` 不是“把 YAML 中所有 IP 全部替换掉”：

- 单实例部署时，它会覆盖顶层 `host_addr`、`deployment.master_host` 和这个唯一实例的 `host`，适合把单机用例临时换到另一台服务器。
- 普通多机或 PD 多节点部署时，它不会重写整套拓扑。各实例、各 node 的 `host` 仍以 YAML 为准。
- 多机用例应直接把正确 IP 写进 `deployment`，不要依赖 `--host_addr` 搬迁整套拓扑。

### 历史兼容字段

旧用例顶层可能还有 `block_size`、`additional_config`、`host_addr`、`port`、`num_devices`、`start_device_id`、`world_size` 和顶层 `serving_extensiable_param`。调度器仍保留部分兼容逻辑，但新用例应按场景使用 `deployment.instances`、`deployment.service_instances` 或 `deployment.pd_instances`：

- xLLM 启动参数写到实例的 `serving_extensiable_param`。
- 卡号写 `device_ids`。
- 服务端口写 `local_port`。
- 不要同时维护两套含义相同的顶层和实例字段。

## 3. Benchmark 部分

`benchmark.benchmark_type` 决定使用哪个测试工具，目前本文覆盖 `aisbench`、`evalscope`、
`vbench`、`image_edit`（gedit 图片编辑性能）和 `asr`（ASR 原始客户端/评分）。
`benchmark_type` 是执行路由依据；
`metrics` 只决定性能/精度大类：

- `metrics: perf` 时提供 `benchmark.perf`（普通 `aisbench`/`evalscope` 性能，以及
  `benchmark_type: image_edit` 的 gedit 性能都用它）。
- `metrics: acc` 时提供 `benchmark.acc`——普通精度和 SWE-bench 新格式都用它：
  `benchmark.acc` 内含 `mode`/`runner` 等 SWE 特征字段时按 AISBench SWE-bench 处理，
  否则按普通 AISBench 精度（含 gedit）处理。
- ASR 例外：`benchmark_type: asr`、`metrics: acc,perf`，全部评测配置放在
  `benchmark.asr`；不额外填写 `benchmark.acc` / `benchmark.perf`，不套用 LLM
  token/s、TTFT/TPOT 门禁。当前只支持 `backend: legacy_infer`，详见下方 ASR 小节。
- VBench 例外：`metrics: acc`（旧写法 `metrics: vbench` 兼容）但配置块是
  `benchmark.vbench`，见第 8 节；判定入口是 `xtask/benchmarks/vbench/task_gate.py`。
- 旧写法 `metrics: swebench` + `benchmark.swebench` 仍兼容保留；
  顶层 `dataset_name: swebench` 也是有效的旧格式兼容信号。

`benchmark.perf/acc.case_type` 建议分别写成 `performance`、`accuracy`，便于阅读和结果归类；但切换执行流程的仍是顶层 `metrics`。

### AISBench 性能

```yaml
benchmark:
  benchmark_type: aisbench
  perf:
    case_type: performance
    api_type: vllm_api_general_stream
    dataset_conf: gsm8k_gen_0_shot_cot_str_perf
    input_len: 1024
    output_len: 1024
    max_out_len: 1024
    batch_size: 1
    request_rate: 1
    num_prompts: 6
    num_warmups: 20
    baseline: 110.6873
    threshold: 0.10
    generation_kwargs: {temperature: 0, ignore_eos: true}
```

AISBench 性能字段如下：

| 字段 | 建议填写 | 通俗解释 | 当前实际行为 |
| --- | --- | --- | --- |
| `case_type` | 是 | 标记这是性能测试 | 建议固定写 `performance`；执行分支仍由顶层 `metrics: perf` 决定。 |
| `api_type` | 是 | AISBench 使用哪种 OpenAI/VLLM 接口配置 | 常用 `vllm_api_general_stream`；多轮对话使用 `vllm_api_stream_chat_multiturn`。xTask 会按名称查找 AISBench 内置配置并校验文件存在，不再维护固定白名单。 |
| `dataset_conf` | 是 | AISBench 数据集配置名 | 通用数据集直接使用此配置名；GSM8K/SYN 等需要专用预处理的数据集仍由预处理结果覆盖。AISBench 新增内置数据集后，通常只需在 YAML 更新此字段。 |
| `dataset_path` | 按需 | benchmark 块内的数据路径 | 非空时，xTask 复制对应的 AISBench 内置数据集配置到用例输出目录，并把其中的数据集 `path` 改成映射后的容器路径，不修改公共 AISBench 源码。 |
| `infer_mode` | ShareGPT 按需 | 多轮对话发送方式 | 支持 `last`、`every`、`every_with_gt`；ShareGPT 固定输入长度必须使用 `last`。省略时保留 AISBench 配置中的默认模式。 |
| `input_len` | 是 | 每个请求的输入 token 长度 | `dataset_name: gsm8k` 时，框架在容器内把问题截断或重复到固定长度；ShareGPT 同时配置 `infer_mode: last` 时，按完整 chat template 严格构造该长度。 |
| `output_len` | 是 | 希望生成的最大 token 数 | 当前会写入 AISBench 后端的 `max_out_len`，因此它才是实际控制输出上限的字段。 |
| `max_out_len` | 兼容 | 历史上的最大输出长度 | 当前 AISBench 生成器不独立使用它；即使同时配置，也由 `output_len` 覆盖后端 `max_out_len`。新用例保持二者相同最直观。 |
| `batch_size` | 是 | AISBench 一批同时处理多少请求 | 也是 `auto_assess` 摸高的目标字段。它与总请求数不是同一回事。 |
| `request_rate` | 是 | 发请求速率 | 会写入 AISBench API 配置。`0` 通常表示不主动限速，具体含义以所装 AISBench 版本为准。 |
| `num_prompts` | 是 | 正式测试总请求/样本数 | GSM8K 或固定长度 ShareGPT 会按此数量生成数据，不足时循环补齐。搜 `benchmark.perf.batch_size` 且未同时搜索 `num_prompts` 时，框架会自动令 `num_prompts = batch_size * 4`。 |
| `num_warmups` | 建议 | AISBench 正式统计前预热多少条 | 传给 `ais_bench --num-warmups`；与 EvalScope 的 `warmup_paras` 不是一套机制。 |
| `temperature`、`top_k`、`top_p`、`seed`、`repetition_penalty`、`ignore_eos` | 按需 | 常见生成参数 | 会先分别写入后端配置。布尔值应写 YAML 布尔值 `true/false`。 |
| `generation_kwargs` | 建议 | 一次性覆盖后端的完整生成参数块 | 只要非空，框架会重写 AISBench 配置中的 `generation_kwargs`，因此它的内容最终优先于上面分散填写的同名生成字段。建议统一把生成参数写在这里。 |
| `baseline` | 按验证目的 | 预期性能基线 | 当前拿解析结果中的 `output_tps_avg` 与它比较。设为 `null` 或不写时，不做基线比较，性能结果存在即可通过这一步。 |
| `threshold` | 配 baseline 时是 | 基线上下允许浮动的比例 | `0.10` 表示结果必须在 `baseline * 0.9` 到 `baseline * 1.1` 之间，不是 10 个绝对单位。取值应在 0 到 1。 |

例如 `batch_size: 8`、`num_prompts: 32` 的意思是“最多按 8 的批量/并发档位跑，总共统计 32 个请求”，不是只发 8 个请求。

### ShareGPT 性能

ShareGPT 数据由使用者提前准备，xTask 不自动下载。默认沿用 AISBench
`sharegpt_gen` 中配置的相对路径，宿主机应放到：

```text
/export/home/models/xllm_test_060/benchmark/ais_bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
```

`download_ais_bench.sh` 将 `benchmark` 目录准备到 `/workspace/benchmark` 后，
AISBench 会按默认相对路径读取
`/workspace/benchmark/ais_bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json`。
YAML 默认不需要填写 `dataset_path`：

```yaml
metrics: perf
dataset_name: sharegpt

benchmark:
  benchmark_type: aisbench
  perf:
    case_type: performance
    api_type: vllm_api_stream_chat_multiturn
    dataset_conf: sharegpt_gen
    output_len: 512
    batch_size: 8
    request_rate: 0
    num_prompts: 100
    num_warmups: 20
    baseline: null
```

如数据无法放在 AISBench 默认目录，仍可在顶层或 `benchmark.perf` 中填写
`dataset_path`。xTask 会生成用例级配置覆盖 `sharegpt_gen` 的 `path`，该兼容方式保留，
但不是 ShareGPT 示例的默认方案。

上面的 `17_single_aisbench_sharegpt_perf.yaml` 保留 AISBench 默认 `every` 模式和原始
ShareGPT 输入长度。严格固定最后一轮输入长度使用独立的
`18_single_aisbench_sharegpt_last_fixed_perf.yaml`，关键配置如下：

```yaml
benchmark:
  perf:
    infer_mode: last
    input_len: 2048
    num_prompts: 100
```

ShareGPT 的固定长度处理是显式启用的：仅当 `infer_mode: last` 且填写正整数
`input_len` 时生效。xTask 使用 `model_path` 对应的 tokenizer 和 chat template，按
AISBench `last` 模式实际发送的内容计算长度，即“所有保留的历史问答 + 最后一条 human
消息 + generation prompt/template token”。超长样本优先删除最早的完整问答轮次，再截断
最后一条 human 内容；不足时重复该内容并补齐。生成后每条都会重新编码校验，未严格等于
`input_len` 时直接终止测试。

`num_prompts` 同时控制生成的对话组数和传给 AISBench 的正式请求数。源数据有效序列不足时，
xTask 会循环复制已处理序列，并生成唯一 ID，最终文件严格包含 `num_prompts` 组。在
`infer_mode: last` 下每组只发送最后一轮请求，因此正常情况下正式请求数就是
`num_prompts`；预热会循环复用这些数据，不要求额外生成 `num_warmups` 组。

处理结果和覆盖了 `path`、`infer_mode` 的数据集配置都写入当前 case 输出目录，不修改
AISBench 源码、公共配置或原始 ShareGPT 文件。服务端必须使用与 `model_path` 相同的
tokenizer/chat template，服务端上报的 `prompt_tokens` 才能与 `input_len` 严格一致。

如果不同时配置 `infer_mode: last` 和 `input_len`，ShareGPT 继续使用原始数据，不执行固定
长度预处理。只配置 `infer_mode` 时，xTask 仅生成用例级配置来切换 AISBench 推理模式。

### ShareGPT Prefix Cache

`22_single_aisbench_sharegpt_prefix_cache_perf.yaml` 展示了默认的单会话公共前缀模式；
需要跨多条 ShareGPT 会话拼接长公共前缀时，可参考
`23_single_aisbench_sharegpt_prefix_cache_packed_perf.yaml`。
Prefix Cache 数据构造必须显式配置 `prefix_cache.enabled: true`，并使用
`infer_mode: last`。`prefix_len` 表示同组请求最少共享的 prompt token 数，xTask 会记录
实际共享长度；`prefix_groups` 控制相互独立的公共前缀组数。

默认的 `prefix_assembly: single` 只从一条 ShareGPT 会话的完整历史轮次构造每组前缀，
保持现有用例行为。当原始数据没有足够长的单条会话时，可显式使用 `packed`：

```yaml
benchmark:
  benchmark_type: aisbench
  perf:
    infer_mode: last
    input_len: 153600
    output_len: 512
    max_out_len: 512
    num_prompts: 40
    num_warmups: 4
    prefix_cache:
      enabled: true
      prefix_source: dataset
      prefix_assembly: packed
      prefix_len: 131072
      prefix_groups: 4
      seed_requests: true
      order: grouped
```

`packed` 模式先按文本规模筛选较长会话，再使用模型实际 tokenizer/chat template 计算
token 长度。xTask 会先为每个组预留不同的、最接近但不超过 `prefix_len` 的长会话作为
anchor，再按 best-fit 顺序补充较长的完整会话；只有整条补充会话无法放入
`input_len` 时，最后一个来源才按完整 human/gpt 轮次截取。达到目标后，仅对最后一条
共享消息做 token 级裁剪，使实际公共前缀尽量精确落在 `prefix_len`；原始 ShareGPT 文件
不会被修改，处理结果仍写入当前 case 输出目录。

每个请求在公共前缀之后的差异化部分也会优先拼入多条较长的 ShareGPT 完整问答，并为
同组请求选择不同的第一条后缀内容；最后一条 human 消息只负责把总 prompt 精确补齐或
截断到 `input_len`，避免用一个很短的问题重复填满全部后缀空间。

生成数据中的 `xtask_prefix_history_source_ids`、
`xtask_prefix_history_source_turns` 和 `xtask_prefix_tokens` 分别记录每组拼接来源、使用轮数
和最终实际共享 token 数。不同组不会复用已选中的 anchor/补充来源，并会在写文件前验证：
同组共享长度不小于 `prefix_len`、组间在 `prefix_len` 之前已分叉、每条最终输入严格等于
`input_len`。

长上下文用例还必须保证模型最大上下文不小于 `input_len + output_len`，并确保 xLLM 的
KV Cache 能同时容纳所有预热前缀和活动请求。配置 `seed_requests: true` 时，
`num_warmups` 必须不小于 `prefix_groups`；`order: grouped` 可降低多组前缀在正式请求期间
被频繁淘汰的概率。

对于后续 AISBench 内置数据集，只要其配置是标准的单数据集配置，通常填写 `dataset_name`、`dataset_conf` 和可选的本地 `dataset_path` 即可，无需增加 xTask 数据集分支。包含多个子数据集或动态构造多个路径的 AISBench 配置，应沿用官方目录布局并省略 `dataset_path`，避免把多个子路径统一覆盖成同一个值。

### AISBench 精度

把 `metrics` 改成 `acc`，并改为 `benchmark.acc`：

```yaml
benchmark:
  benchmark_type: aisbench
  acc:
    case_type: accuracy
    api_type: vllm_api_general_chat
    dataset_conf: gsm8k_gen_0_shot_noncot_chat_prompt
    acc_dataset_len: 200
    batch_size: 8
    output_len: 32768
    request_rate: 8
    baseline: 96.0
    threshold: 0.04
    generation_kwargs: {temperature: 0, ignore_eos: false}
```

AISBench 精度字段如下：

| 字段 | 建议填写 | 通俗解释 | 当前实际行为 |
| --- | --- | --- | --- |
| `case_type` | 是 | 标记这是精度测试 | 建议固定写 `accuracy`；真正执行精度分支的是顶层 `metrics: acc`。 |
| `api_type` | 是 | 请求服务使用的 AISBench 接口配置 | 对话类精度常用 `vllm_api_general_chat`。 |
| `dataset_conf` | 是 | AISBench 精度数据集配置名 | 当前代码只有在它非空时才真正生成精度执行命令；缺少它会直接跳过精度测试。 |
| `dataset_path` | 按需 | benchmark 块内的数据路径 | 数据下载/准备仍按 `dataset_name` 和对应数据集逻辑处理，不要假设这个字段能覆盖所有数据来源。 |
| `acc_dataset_len` | 按需 | 最多测多少条样本 | 适合缩短调试时间。AISBench 的 GSM8K 数据准备会按它截断；`0`/空值的语义取决于数据准备逻辑，正式用例应明确设置。 |
| `batch_size` | 是 | 精度推理批大小 | 写入 AISBench API 配置。不要把它理解成样本总数。 |
| `output_len` | 是 | 单条答案最多生成多少 token | 当前写入 AISBench 后端的 `max_out_len`。 |
| `request_rate` | 按需 | 精度请求发送速率 | 写入 AISBench API 配置。 |
| `num_prompts` | 否 | 限制 AISBench 执行的 prompt 数量 | 只有值为整数时才生成 `--num-prompts` 参数。通常使用 `acc_dataset_len` 更容易理解。 |
| `temperature`、`top_k`、`top_p`、`seed`、`repetition_penalty` | 按需 | 生成采样参数 | 会先写入后端配置。 |
| `generation_kwargs` | 建议 | 最终采用的生成参数块 | 非空时重写 AISBench 的 `generation_kwargs`。精度复现通常建议 `temperature: 0`，并明确 `ignore_eos`。 |
| `baseline` | 是 | 预期精度值 | 当前精度结果必须有有效正数 baseline 才可能判定通过；未配置 baseline 不会自动放行。注意不同数据集的精度可能用百分数或 0 到 1 小数，必须与结果解析器的量纲一致。 |
| `threshold` | 是 | 基线上下允许浮动比例 | 同样按 `baseline * (1 ± threshold)` 校验。 |

`dump_eval_details` 有一个容易踩坑的当前限制：AISBench 脚本生成器从 YAML 顶层读取它，即应写成顶层 `dump_eval_details: true`；写在 `benchmark.acc` 里目前不会生效。

GPQA Diamond 可以直接复用 AISBench 内置的 `gpqa_gen_0_shot_cot_chat_prompt` 配置。使用者需提前准备
`gpqa_diamond.csv`，放到宿主机
`/export/home/models/xllm_test_060/benchmark/ais_bench/datasets/gpqa/`；xTask 保持
`dataset_path` 为空时，会沿用 AISBench 默认的相对路径。参考用例为
`24_single_aisbench_gpqa_acc.yaml`。GPQA 评测结果按百分数记录，`baseline` 应使用实测百分数，
不要写成 0 到 1 的小数。

### AISBench SWE-bench

SWE-bench 是独立评测类型，不能写成 `metrics: acc` 的 GSM8K 精度。新写法与普通精度
格式一致：`metrics: acc`、`benchmark.benchmark_type: aisbench` 和 `benchmark.acc`，
并在 `benchmark.acc` 中携带 SWE 特征字段（`mode`、`dataset`、`runner` 等）和
`case_type: accuracy`。旧写法 `metrics: swebench` + `benchmark.swebench` 仍兼容保留。
判定入口是 `xtask/benchmarks/aisbench/swebench.py` 的 `is_aisbench_swebench_task`：
要求 `benchmark_type: aisbench` 且 SWE 块有效——新格式看 `benchmark.acc` 的
`mode`（`infer`/`eval`/`all`）或 `runner` 字段，旧格式看 `benchmark.swebench`（配合
`metrics: swebench` 或 `dataset_name: swebench`）；普通 GPQA/gedit 精度和性能用例
不会进入这条路径。

**不需要拆成两份 xTask。** 主路径就是一份 YAML、一次执行：

1. xTask 按 `deployment.instances` 在 NPU 机器启动 xLLM（启动参数写法和 `01_single_aisbench_perf.yaml` 一样）。
2. 同一轮调度把 AISBench SWE（mini-swe-agent infer + 官方 harness 打分）派到
   `benchmark.acc.runner.host` 这台 **x86** 机器。
3. NPU 容器不安装 AISBench，也不跑 SWE 镜像。x86 需要能访问 xLLM 的 HTTP 地址。

```yaml
metrics: acc
dataset_name: swebench
benchmark:
  benchmark_type: aisbench
  acc:
    case_type: accuracy
    mode: all                 # infer | eval | all；默认一次做完，不必拆两次
    dataset: verified_mini
    step_limit: 200
    reuse: true
    timeout: 86400
    runner:
      host: 11.87.48.253      # x86；Docker 需已装，AISBench/SWE 由 xTask 自动安装
deployment:
  master_host: 11.87.191.75
  instances:
  - name: master-0
    host: 11.87.191.75
    device_ids: [0, 1]
    local_port: 8800
    serving_extensiable_param: { ... }
```

完整端到端示例：`28_single_aisbench_swebench.yaml`。

xTask 会把主节点上的 AISBench 源码（`DEPENDENCE_PACKAGE_PATH/benchmark`，与
`download_ais_bench.sh` 同一棵树）同步到 x86，并在 x86 本机 `pip install`
AISBench、mini-swe-agent 和 SWE-bench harness。不需要事先手装这三样。
x86 仍需 Docker 可用，`docker ps` 能通过；instance 镜像首次评测时再 pull。
不要把 ARM NPU 机器上的 binfmt 当作生产路径。
如果评测完整 SWE-bench 数据集、x86 上大量实例镜像尚未缓存，
应先在 x86 上预拉取镜像再启动 xLLM，避免 NPU 长时间等待 Docker Hub 下载；
离线预拉取和镜像源网络诊断见 `docs/aisbench_swe_image_prefetch.md`。

SWE-bench 用例设置 `reuse: false` 时，xTask 给 AISBench 显式指定用例工作目录，
新结果与普通 AISBench/EvalScope 一样保存在
`outputs/<case_name>/<YYYYMMDD_HHMMSS>/`，而非默认的
`outputs/<case_name>/outputs/default/<YYYYMMDD_HHMMSS>/`。
历史 `outputs/default/` 结果不会迁移或删除；后处理仅使用本次新建的日期目录，
防止历史结果掩盖当次失败。设置 `reuse: true` 的旧用例继续沿用 AISBench
原有 `outputs/default/` 路径，以免改变历史预测的复用方式。
通过 SWE-bench `extra_args` 显式指定 `--work-dir` 的用例也沿用其自定义路径。
x86 评测结果回传时先下载到用例目录外的临时目录，再合并 AISBench 结果，
不再整体替换主节点的用例目录：本机 xLLM 正在写入的 `xllm_log/<实例>/log/`
会保留，并在运行结束时与新产生的 `shutdown_log/` 一起归档到本次日期目录。

如果 x86 的宿主机 Python 不可用，可以显式启用 xTask 管理的评测控制容器，
而不必预先手动执行 `docker run` 或维护 `python3.11-docker`、`ais_bench-docker` 包装脚本：

```yaml
benchmark:
  acc:
    runner:
      host: 11.87.48.253
      runtime_container:
        image: xllm:a2-x86-cann9-main-0920  # 镜像必须预先在该 x86 主机上可用
        python: /usr/local/python3.11.15/bin/python3
        ais_bench_bin: /usr/local/python3.11.15/bin/ais_bench
```

这个开关只影响 SWE-bench 的 x86 评测控制环境，不影响 NPU xLLM 容器或其他评测。
每次用例自动使用 `<case_name>-swebench_rt-<Unix秒级时间戳>` 创建独立容器，
挂载远端用户的 `~/xtask_swe`、宿主机 Docker CLI 和 `/var/run/docker.sock`，
并自动生成 Python/AISBench 包装入口；用例完成或运行失败后默认删除**本次创建的容器**及包装入口，
保留数据集、预测和打分输出目录。已有同名容器不会被接管或删除。
xTask 不负责安装 Docker 或拉取评测控制镜像。
默认沿用 x86 宿主机 `/etc/resolv.conf` 中的 DNS；如有特殊需求可在
`runner.runtime_container.dns` 中指定 IP 地址列表。
`runner` 也可填写 `python`、`install_dir`、`timeout`、`skip`、`source_root`、
`aisbench_source`、`mini_swe_agent_source`、`swebench_harness_source` 等准备参数。
旧 `benchmark.swebench.setup` 写法保留兼容（包括其旧版共享容器行为），新用例统一写在 `runner` 下。
**挂载 Docker socket 意味着容器内程序可以控制宿主机 Docker，只能在可信的专用 x86 runner 上启用。**
未配置 `runtime_container` 时，保持原有宿主机 Python 或手写包装入口的行为。
若 x86 Docker daemon 的 `put_archive` 在挂载容器文件时错误返回 ENOSPC，
可在该 SWE 用例的 `runner` 中**显式**写 `docker_copy_fallback: true`。
仅此用例的 AISBench 子进程会先尝试原 Docker API；仅命中挂载 ENOSPC 时改用
`docker exec -i ... tar -xf -` 重试。该临时降级不修改共享 Docker daemon、
不影响 VBench 或普通 AISBench；长期仍应修复 x86 的 Docker 环境。
多实例 SWE-bench 在 x86 无法直连 GitHub 时，可在 `runner.runtime_container.proxy`
中配置 `http`、`https` 和指向模型服务的 `no_proxy`；代理只传入 AISBench 评测进程，
不改变依赖安装环境。若该代理偶发中断，可仅在受影响的 SWE-bench 用例设置
`runner.github_raw_retry: true`：仅对 `raw.githubusercontent.com` 的 GET 网络超时、
连接失败或临时 HTTP 429/5xx 重试最多三次（1s、2s、4s），不重试证书错误，
不改变模型 API、普通 AISBench、EvalScope 或其他未启用此开关的 SWE-bench 用例。
x86 Docker 偶发因 overlay 挂载 ENOSPC 不能启动 mini-swe-agent
实例时，可为该 SWE 用例设置 `runner.docker_run_retry: true`：只对此类 Docker
创建失败最多再试两次（2s、4s），其他错误不会自动重试。
mini-swe-agent 默认在全部实例处理结束后统一清理 Docker 实例；在 x86 daemon
容易于连续创建时发生 overlay 挂载错误的专用用例，可设置
`runner.clean_infer_container: true`，仅按当前实例的容器 ID 同步清理已结束的
mini-swe-agent 容器，再开始下一个实例。上述开关均不影响其他评测分支。
全量数据集中的复杂实例可能超过 mini-swe-agent 默认的 2 小时 Docker 容器生命周期，
导致 `Environment stopped before agent finished`。可为这类 SWE 用例显式设置
`runner.agent_container_timeout: 4h`，仅延长该用例 Agent 实例容器的寿命；
它不改变整轮 SWE 超时（`benchmark.acc.timeout`，旧写法 `benchmark.swebench.timeout`）、Agent 的 `step_limit` 或其他类型评测。
评测结果若有 `error_instances > 0` 或 `harness_exit_code != 0`，即使
`baseline: null` 也会判为失败；真正的未解决实例仍按解决率统计，正式回归应设置基线。

只有服务已经在跑时，才用 `external_service` 写法，见 `29_external_aisbench_swebench.yaml`，
语义和 `26_external_aisbench_perf.yaml` 相同。`mode: infer` / `mode: eval` 也可以拆开做，
那是可选的断点续跑，不是默认要求。

### AISBench GEdit 图片编辑精度（LLM-judge）

GEdit（Qwen-Image-Edit）精度走 AISBench 精度分支，但使用 xTask 内置的两段式图片编辑
API：`metrics: acc`、`benchmark.benchmark_type: aisbench`、`benchmark.acc`，关键字段是
`api_type: gedit_image_edit` 和 `dataset_conf: gedit_gen_0_shot_llmjudge`。它的
`benchmark.acc` 不含 `mode`/`runner` 字段，因此不会进入 SWE-bench 路径；
`benchmark_type` 仍是 `aisbench`，也不会命中 VBench 判定。

```yaml
metrics: acc
dataset_name: gedit
benchmark:
  benchmark_type: aisbench
  acc:
    case_type: accuracy
    api_type: gedit_image_edit            # xtask 内置两段式 API（encoder + DiT）
    dataset_conf: gedit_gen_0_shot_llmjudge
    num_prompts: 20                       # 调试样例条数；删除后按 GEdit-Bench 全量（1212）
    batch_size: 1
    output_len: 512
    request_rate: 0
    judge_endpoint: 127.0.0.1:18008       # LLM-judge 端点（host:port）
    baseline: 3.925                       # SC/PQ 两维综合分（0-10），替换为实测值
    threshold: 0.20
    generation_kwargs:
      temperature: 0.01
      ignore_eos: false
```

要点：

- 数据集是 GEdit-Bench（load_from_disk 格式），默认放在
  `<DEPENDENCE_PACKAGE_PATH>/benchmark/ais_bench/datasets/GEdit-Bench`；YAML 不写
  `dataset_path`，需要覆盖时才显式指定 `benchmark.acc.dataset_path`（默认值见
  `xtask/benchmarks/aisbench/generate_aisbench_script.py` 的
  `default_gedit_acc_dataset_path`）。
- `judge_endpoint` 留空时，自服务模式由调度器按 `deployment.judge` 注入
  `127.0.0.1:18008`。
- 打分是 LLM-as-Judge（SC/PQ 两维，0-10），`baseline` 写两维的算术平均。
- 两段式协议与自服务部署（encoder/dit/judge 三服务）见第 4 节
  “GEdit 自服务多服务部署”；完整示例 `37_single_aisbench_gedit_acc.yaml`。
- JoyAI-Image-Edit-Plus：`api_type: gedit_image_edit` 的模型配置按 model/path 含
  `joyai` 自动检测模型族（skip 34 / `input.media_sources` 多图，见上文 image_edit 小节）；
  DiT 4 卡 `sp2*cfg2`；完整示例 `39_single_aisbench_joyai_acc.yaml`。

### EvalScope 性能

```yaml
benchmark:
  benchmark_type: evalscope
  perf:
    case_type: performance
    batch_size: 1
    request_rate: 1
    num_prompts: 6
    min_prompt_length: 760
    max_prompt_length: 760
    output_len: 1024
    max_out_len: 1024
    image_width: 512
    image_height: 512
    image_format: RGB
    image_num: 1
    seed: 42
    baseline: 106.1046
    threshold: 0.10
    warmup_paras: {request_rate: 10, batch_size: 10, num_prompts: 20}
```

EvalScope 性能字段与实际命令参数是一一映射的：

| YAML 字段 | EvalScope 命令 | 通俗解释 |
| --- | --- | --- |
| `case_type` | 无独立命令参数 | 建议写 `performance`，执行分支由 `metrics: perf` 决定。 |
| `request_rate` | `--rate` | 每秒发送多少请求；具体不限速写法以当前 EvalScope 版本为准。 |
| `num_prompts` | `--number` | 本次正式测试总共发送多少个请求。 |
| `batch_size` | `--parallel` | 同时在途的最大请求数，也就是性能测试并发。 |
| `min_prompt_length` | `--min-prompt-length` | 随机输入的最小 token 长度。 |
| `max_prompt_length` | `--max-prompt-length` | 随机输入的最大 token 长度。要固定输入长度就与最小值写相同。 |
| `output_len` | `--min-tokens` | 最少生成 token 数。 |
| `max_out_len` | `--max-tokens` | 最多生成 token 数。要固定输出长度就与 `output_len` 写相同。 |
| `image_width`、`image_height` | 同名 `--image-*` 参数 | 随机视觉输入的图片宽、高。文本模型可以删掉，生成器缺省为 512。 |
| `image_format` | `--image-format` | 随机图片格式，例如 `RGB`。 |
| `image_num` | `--image-num` | 每个请求携带几张图片。 |
| `seed` | `--seed` | 随机数据/请求的种子，固定后更容易复现。 |
| `warmup_paras` | 单独生成一轮预热命令 | 先复制正式配置，再用这里的字段覆盖，例如单独设置预热的 `request_rate`、`batch_size`、`num_prompts`。预热结果会清理，不进入正式统计。 |
| `baseline`、`threshold` | xTask 后处理 | 仍使用 `output_tps_avg` 并按 `baseline * (1 ± threshold)` 校验。 |

文本模型可以删除图像字段；视觉模型保留它们。当前生成器固定向 EvalScope 性能命令传入 `--extra-args '{"ignore_eos": true}'`，所以在 `benchmark.perf` 里另写 `ignore_eos` 目前不能覆盖它。

### EvalScope 精度

```yaml
benchmark:
  benchmark_type: evalscope
  acc:
    case_type: accuracy
    acc_dataset_len: 30
    batch_size: 4
    timeout: 3600
    endpoint: /v1
    baseline: 0.30
    threshold: 0.05
    generation_config:
      temperature: 1
      top_p: 0.95
      top_k: 20
      max_tokens: 30000
      retries: 5
    dataset_args:
      mmmu:
        dataset_id: /xllm_test_060/datasets/mmmu
        subset_list: [Music]
        split: validation
```

EvalScope 精度字段如下：

| 字段 | 建议填写 | 通俗解释 | 当前实际行为 |
| --- | --- | --- | --- |
| `case_type` | 是 | 标记这是精度测试 | 建议固定写 `accuracy`。 |
| `acc_dataset_len` | 按需 | 最多评测多少条 | 非空时映射到 `--limit`；不写则不追加 limit。 |
| `batch_size` | 是 | 一次并行评测多少条 | 映射到 `--eval-batch-size`，默认 1。 |
| `timeout` | 建议 | 单次 EvalScope API 请求/评测超时 | 映射到 `--timeout`，当前代码默认 36000 秒。 |
| `endpoint` | 建议 | OpenAI API 基础路径 | 与服务地址、端口拼成 `http://<host>:<port><endpoint>`，默认 `/v1`。 |
| `generation_config` | 按数据集需要 | 传给模型的生成配置 | 整体传给 `--generation-config`。常见键包括 `temperature`、`top_p`、`top_k`、`max_tokens`、`retries`，具体支持范围由 EvalScope 版本决定。 |
| `dataset_args` | 按数据集需要 | 每个数据集自己的位置、子集和 split | 整体传给 `--dataset-args`。键名通常与顶层 `dataset_name` 对应。 |
| `baseline`、`threshold` | 是 | 精度基线和容许波动 | 量纲必须与 EvalScope 解析出的精度一致，例如解析值为 `0.30` 时不要把 baseline 写成 `30`。 |

顶层 `dataset_name` 会映射到 `evalscope eval --datasets`。`dataset_args` 常见子字段包括：

| 子字段 | 作用 |
| --- | --- |
| `dataset_id` | 数据集在容器内可访问的路径或 EvalScope 可识别的数据集 ID。 |
| `subset_list` | 只评测指定子集，例如 `[Music]`。 |
| `split` | 选择 `validation`、`test` 等数据切分。 |

路径以 `/xllm_test_060` 开头时，多机资源同步会按需检查和补齐相应关键资源。`generation_config`、`dataset_args` 由当前代码转换成命令行字典字符串；值中含复杂引号时应先用小用例验证生成的 `run_evalscope_test.sh`。

### GEdit 图片编辑性能（benchmark_type: image_edit）

图片编辑性能是独立的 `benchmark_type: image_edit`，不经过 `evalscope`/`aisbench` 的
普通分支：`xtask/benchmarks/evalscope/generate_image_edit_script.py` 生成脚本，调用
`xtask/plugins/evalscope/run_gedit_perf.py`（先注册 `gedit_image_gen` API 插件再跑
evalscope perf）压测 `/v1/image/generation`。指标量纲是 MPx/s（产出兆像素每秒）：
结果解析里的 `output_tps_avg` 实际是 MPx/s，`baseline` 直接填 MPx/s 数值。

```yaml
metrics: perf
dataset_name: gedit
benchmark:
  benchmark_type: image_edit
  perf:
    case_type: performance
    num_prompts: 20
    batch_size: 1
    request_rate: -1
    warmup_num: 2
    seed: 42
    timeout: 3600               # 单请求超时（秒）：出图较慢，给足余量
    size: "1328*1328"
    num_inference_steps: 50
    guidance_scale: 3.5
    encoder_url: http://127.0.0.1:18007   # 留空则由调度器按 deployment.encoder 注入
    baseline: 0.0973            # 量纲 MPx/s
    threshold: 0.10
```

- 数据集默认 `<DEPENDENCE_PACKAGE_PATH>/benchmark/ais_bench/datasets/gedit_perf/gedit_perf.jsonl`
  （由 `tools/convert_gedit_to_perf.py` 从 GEdit-Bench 转换生成，`images` 字段存绝对
  路径而非 base64）；YAML 不写 `dataset_path`，需要覆盖时才显式指定
  `benchmark.perf.dataset_path`。
- `encoder_url` 优先级：`benchmark.perf.encoder_url` > 环境变量 `GEDIT_ENCODER_URL` >
  `http://127.0.0.1:18007`。
- `size`/`num_inference_steps`/`guidance_scale` 等经 `--extra-args` 透传为请求体参数；
  数据集行内有同名字段时以行为为准。
- 自服务部署（encoder/dit 服务键、端口约定）见第 4 节；完整示例
  `36_single_image_edit_gedit_perf.yaml`。

**JoyAI-Image-Edit-Plus 模型族差异**（示例 `38_single_image_edit_joyai_perf.yaml`）：

- 插件按 `served_model_name`/`model_path` 含 `joyai`/`joyimageeditplus` 自动检测模型族
  （数据集行/`generation_kwargs` 也可显式传 `model_family: joyai|qwen` 覆盖；缺省 `qwen`
  保持既有 Qwen 用例行为零变化）。
- 两段式协议差异：encoder skip tokens 为 34（Qwen 为 64），system prompt 用 Qwen3-VL
  官方描述式模板；DiT 参考图走 `input.media_sources` 多图（Qwen 走 `input.image` 单图）。
- DiT 并行约束：JoyAI MMDiT 32 头（Qwen 24 头），`sp_size` 必须整除 32，36 号的
  6 卡 `sp3*cfg2` 不适用（32 % 3 != 0）；38/39 号取 4 卡 `sp2*cfg2`（NPU 0-3）。
- 出图分辨率：请求传 `parameters.size`（缺省 `1328*1328`，数据集行自带 `size` 时以行为准）。
  实测（20261103）不给 `size` 走 bucket 机制会退化为 `512*512` 出图，judge 打分崩塌
  （SC/PQ 2.45/6.95 -> 0.85/5.05），故两模型族统一显式传 size。
- 生成参数官方推荐 `num_inference_steps: 30` / `guidance_scale: 4.0`。

<a id="asr-legacy-benchmark"></a>

### JoyAI-ASR 原始评测（benchmark_type: asr，第一阶段）

#### 范围与原始材料

`benchmark_type: asr`、`benchmark.asr.backend: legacy_infer` 使用
`asr_doc/infer.zip` 中的 **原始 `infer_dataset.py` 和 `compute-wer.py`**。
原压缩包不修改；执行前安全解压到本轮输出目录。不是另写一个识别客户端，
也不是已经完成 AISBench 原生适配。当前仅接入原始执行器；AISBench 第二阶段
暂缓实施，后续若接入应沿用数据、评分 profile 和统一结果门禁。

不运行原 `infer.sh` 的自动 pip 安装、固定输出路径和固定 IP；由 xTask 生成
可复现配置后调用原 Python 客户端。仍保留原始并发执行、context 构造、
请求格式、采样参数、编辑距离和归一化规则。

#### 准备与运行

1. 本机准备 `asr_doc/infer.zip`、`asr_doc/data.zip`。或通过 `infer_archive`、
   `dataset_archive` 指定替代路径。压缩包将暂存到用例 `asr_inputs/`，随
   用例同步到 benchmark runner，不需要在所有服务主机单独部署数据集。
2. 测试容器准备 `httpx`（测试验证版本 0.28.1）；缺失会失败，不在线自动安装。
   核心评分仅使用 Python 标准库；`wer_analys.py` 可在有 `kaldialign` 的环境
   对输出 `wer` 离线运行，不是核心执行依赖。
3. TP2 示例：修改模型路径、host、`device_ids`、userid，并确保真实 CTC 权重、
   模型 chat template 和音频前处理完整。环境版本由部署镜像负责。

```bash
python main.py --case_name joyai_asr_xllm_tp2_legacy
# 或先手工部署服务，只测试接口（不管理外部服务生命周期）：
python main.py --case_name 40_external_joyai_asr_legacy_acc_perf
```

TP2 示例与新性能复现 PDF 对齐：CTC、Suffix/6 tokens、schedule overlap=false、
并发 `[1,2,5,10]`、context 和 SSE 开启。不是旧 9 月 Rotary 报告的逐项复现。
不执行 PDF 的 `pkill -9 xllm`，服务生命周期走 xTask 的既有调度/清理。

**当前三个 ASR 用例都是独立 YAML，不存在继承、包含或互相调用。**它们共享同一套
`legacy_infer` 客户端、数据集、评分和归档代码，只是部署模式和运行参数不同。

| 用例 | 服务生命周期 | 并发 × repeats | 正式请求数 | 数值门禁 |
| --- | --- | --- | ---: | --- |
| `40_external_joyai_asr_legacy_acc_perf.yaml` | `.101:8018` 外部服务单轮实测示例 | `[2]` × 1 | 1162 | 错误率 ≤5%，失败请求=0；无性能基线 |
| `41_external_joyai_asr_legacy_repeats.yaml` | 同一外部服务的重复统计、日期归档/报表验证示例 | `[2]` × 2 | 2324 | 同 40，各轮分别检查 |
| `joyai_asr_xllm_tp2_legacy.yaml` | xTask 在配置节点启动 TP2 + CTC + Suffix 服务并清理 | `[1,2,5,10]` × 1 | 4648 | `gates: []` |

表中请求数不含整场开始前的 3 条合成音频预热。所有用例都使用日期归档和报表，
41 不是专门开启归档的开关。`gates: []` 仍检查请求完整性、失败数、评分有效性
和并发/轮次覆盖。40/41 的 5% 是临时质量上限，不是正式验收基线。

外部服务用例不会强制服务器使用 TP2/CTC/Suffix；`provenance` 只是来源记录，
不能代替实际部署证据。自部署 TP2 模板默认也指向 `.101:8018`，执行前必须
确认服务归属、空闲设备和端口，避免
与已有外部模型服务冲突。40/41 的 HTTP 实测不等于自部署 TP2 链路已验证。

#### 配置

```yaml
metrics: acc,perf
benchmark:
  benchmark_type: asr
  asr:
    backend: legacy_infer
    # 默认路径均为 <xTask>/asr_doc/；可指定绝对路径。
    # infer_archive: /path/to/infer.zip
    # dataset_archive: /path/to/data.zip
    dataset_name: ta3_test0925_quiet
    expected_samples: 1162   # 校验原始清单数量；limit 不改变此校验
    limit: 0                # 0=全部；调试时可设 2
    api: audio              # audio=multipart；chat=audio_url JSON
    concurrency: [1,2,5,10]
    repeats: 1
    stream: true            # 仅文字 SSE，不是输入音频实时流式
    use_context: true
    timeout: 300             # httpx/urllib 原超时语义，不是独立整请求 watchdog
    max_audio_seconds: 30   # 越界失败，不能静默过滤或用截断音频夸大吞吐
    cache_condition: service_lifetime
    scoring_profile: joyai_legacy
    warmup:
      synthetic_requests: 3 # 可选 0..3，包内 w2/w8/w20，整场开始前一次
      api: audio            # 对齐原 Shell，亦可设 chat
      exclude_first: 0      # 按索引排除性能统计；精度仍包含全部样本
    generation:
      temperature: 0
      repetition_penalty: 1.05
      language: zh
      # hotwords: 订单、物流  # 全局回退值；优先使用清单中的逐条值
      # top_p: 1             # audio 专用
      # top_k: -1            # audio 专用
      # max_completion_tokens: 3000  # audio 专用，默认不发送
      # max_tokens: 256      # chat 专用，默认 256
      # max_context_chars: 450 # chat 专用，保持原客户端处理
      # chat_prompt: 请转写音频为文字，并给出正确的标点符号。
    # 单位：错误率=百分数(4.03 而非 0.0403)，延迟=ms，吞吐=req/s 或 audio s/s。
    gates:
      - metric: error_rate_pct
        compare: lte
        baseline: 4.5
        threshold: 0.05      # 单边相对容忍，实际 <= 4.5*(1+0.05)
      - metric: e2el_p99_ms
        compare: lte
        baseline: 1000
      - metric: failed_requests
        compare: lte
        baseline: 0
```

ASR 数据来源由 `benchmark.asr.dataset_archive` 和 `dataset_name` 决定，
不是普通 AISBench 的 `benchmark.perf.dataset_path`。`concurrency` 是 ASR
客户端并发，不使用 `batch_size`；`repeats` 表示每个并发档位的完整数据集重复次数。

以上门禁数值仅示范，**不代表验收基线**。每个并发、每个 repeat 的所有门禁
采用 AND；改善不会因越出下界失败；0% 错误率合法；缺失结果不能用 0 替代。
所有运行必须请求完整且无请求失败，即使未配置 gates 也会校验。

不支持 `seed` 配置：原 audio 客户端不发送，原 chat 客户端固定为 4419。
chat 的 temperature 固定 0；对不生效的 audio-only 参数会拒绝配置，防止误记。
模型识别 ID、参考文本、context 每条对应；参考答案不送给模型。

#### 缓存、预热和重复

每个用例内部并发扫描和 repeats 共用同一服务生命周期，客户端不重启 xLLM。
`enable_prefix_cache=false`、`speculative_suffix_max_cached_requests=0` 不代表音频
编码缓存关闭。后续重复音频可能命中缓存。禁止把配置命名为 `cold` 而未真正
保证冷缓存。若需要每轮重启，使用独立正常 xTask 用例，各用例只测一个并发
及一个 repeat；外部服务模式由外部操作者负责重启。

初始预热调用若 HTTP 失败立即失败。预热得到空文本允许，因为它是合成音频；
正式集参考非空的空预测判请求失败。`exclude_first` 只排除性能，不隐藏失败。

#### 评分与指标口径

* `joyai_legacy` 原命令：`compute-wer.py --char=1 --v=1`；中文按字、拉丁片段
  按段，保留原大小写/标点/标签规则。指标名为 `error_rate_pct`，**不伪称纯 CER
  或标准英文 WER**。语料级 `(S+D+I)/N`，记录 S/D/I/N、脚本 SHA256。
* 同一份包内历史预测可复算：N=14080，S=312，D=99，I=156，错误率约 4.027%。
  追加 `--cer=1` 会改变计数，不能与原 profile 混用；标准评分属于第二阶段。
* `e2el_*_ms`：保持原客户端计时边界；chat 在读取/Base64 编码后计时；
  multipart 包含音频上传。分位数为原脚本线性插值。
* `legacy_ttft_*`：原 choices-bearing 分片计时，并不保证首个文本或首 token。
* `first_text_*`：新增首个**非空文本 SSE**时延；同步响应不伪造首文本指标。
* `legacy_rtf`：总请求延迟 / 总音频时长，不是并发整场耗时。
* `request_throughput` / `audio_throughput`：测量窗口内成功请求数 / 成功音频秒数
  除以窗口墙钟秒数；窗口为首个被统计请求开始到最后结束。
* `simulate_realtime=true` 沿用原数据分片+休眠策略，不等于输入流式 ASR；
  尾部休眠不包括在新增吞吐窗口中。
* 推测解码 Prometheus 计数若可用，仍保存在原 `latency_summary.txt`。

客户端保持原请求，但额外观测 SSE：无正常结束标记、错误事件、无效 JSON、
非空标注对应空转写均判失败。失败结果原 scorer 按空转写计漏字，同时测试
失败；不重试，不隐式过滤。参考/预测 ID 必须完整一致，不沿用评分器的交集
静默跳过行为。

#### 输出与溯源

```text
outputs/<case>/
├── run_asr_test.sh
├── asr_inputs/                    # 用例暂存原压缩包，runner_config.json 权限 600
└── YYYYMMDD_HHMMSS/               # 每次调用独立，不覆盖上一轮
    ├── config.json                # 参数、服务配置、归档哈希、Python/httpx 信息
    ├── asr_result.json            # 所有并发/repeat 摘要和最终门禁，原子更新
    ├── execution_result.json      # xTask 最终状态、退出码、耗时、归档主机、报表是否生成
    ├── runs.csv                   # 每个并发/repeat 的计数、错误率和性能
    ├── summary_by_concurrency.json # 各并发的有效轮次、均值、中位数、标准差
    ├── task.yaml / source_task.yaml # 实际配置与源 YAML（认证信息打码）
    ├── runner_config.json        # 本轮客户端参数，不再依赖下一轮覆盖的配置
    ├── run_asr_test.sh / commands.json # 本轮执行脚本及服务/测试命令
    ├── benchmark_stdout.log / benchmark_stderr.log
    ├── xllm_log/                 # 仅自部署模型：本轮推理日志、启动/清理材料
    ├── assets/                    # 原客户端、评分脚本和解压数据
    ├── c1_r1/
    │   ├── samples.json           # ID、音频路径、时长、逐文件哈希
    │   ├── requests.jsonl         # 实际请求配置(音频 Base64省略)、原预测、状态、时延
    │   ├── summary.json           # S/D/I/N、性能、完整性、文件路径
    │   ├── text.hyp / text.ref / text.ctc / err.log / wer
    │   └── latency.tsv / latency_summary.txt
    └── c2_r1/ ...
```

认证 key 在归档配置中打码；原 client 代码/数据 SHA256 自动记录。模型权重和
服务 commit 无法通过客户端可靠自动证明，可在 `benchmark.asr.provenance` 明确
填写并随配置归档；服务日志与启动命令走 xTask 已有归档。
Excel 为各并发/repeat 展开错误率、S/D/I/N、P99、RTF、吞吐；不把样本文本或音频
塞进单元格。注意数据及 requests.jsonl 含业务内容，使用合适的归档权限。

##### 日期归档规则

单用例执行在 `outputs/<case>/YYYYMMDD_HHMMSS/` 独立归档；使用 `--case_group`
时为 `outputs/<group>/<case>/YYYYMMDD_HHMMSS/`。若 group 名与 case 名相同，
相邻同名目录是既有布局，不是重复的日期层。同一用例重复执行不覆盖历史日期目录。同秒运行发生冲突时选取下一个可用秒级目录。用例内的并发及
repeat 分别存入 `c<N>_r<M>/`。不额外增加一层日期目录，也不改变既有 Excel 汇总。

* 成功、请求失败、评分失败、预热失败都保留日期目录；服务就绪失败也会创建
  失败归档。中途被终止时读取已落盘请求，补 `state: interrupted` 的失败摘要，
  不补造评分，不把部分数据标成完整成功。若整个 xTask 控制进程也被强杀，
  原子更新的 runner 初始/中间摘要仍在；控制端最终摘要需要恢复后处理才能生成。
  损坏的顶层 JSON 保存为 `asr_result.corrupt.json`；已完成 repeat 的独立摘要可以
  恢复，但整场仍标记 interrupted/failed，不把缺失请求或损坏结果判通过。
* 清理容器前，将本轮容器创建的日期目录所有权交还调用者，再归档命令、日志。
  不通过全员可写权限解决业务数据归档。自部署服务额外归档模型推理日志；外部
  服务只归档客户端证据，不能凭客户端抓取远端服务日志。
* JSON 的 `config_file`、各轮 `files` 相对于所在摘要目录保存；后处理仅在内存
  中解析为宿主绝对路径。旧 `/workspace/outputs/...` 引用可兼容读取。原始执行
  命令和 samples 中的源路径保留为溯源信息，不保证可跨环境重新执行。
* `execution_result.json` 分开记录 `artifact_host/artifact_path` 和 `service_endpoint`。
  Excel 结果位置用归档所在主机，不再把外部模型服务 IP 当作文件所在主机。
* 归档内脚本和配置首次写入后不再被下一轮替换；执行状态和耗时在 xTask
  finalize 时更新。`asr_result.json` 表示 benchmark 结果；`execution_result.json`
  表示包含准备/归档的完整 xTask 执行状态，归档失败会标明失败。


#### 轮次汇总、搜参与可选看板

* `runs.csv` 保留每个并发/repeat 的计数、错误率、延迟和吞吐；失败轮次不删除。
  `summary_by_concurrency.json` 按并发分别列 total/valid/invalid 轮次及有限数值的
  mean/median/min/max/stddev。不同并发不混算，失败/不完整轮次不参与有效统计。
  统计中的 P99 中位数是“各轮 P99 的中位数”，不是合并全部请求后重新计算的 P99。
* Excel 同时展开原始各轮指标和各并发重复统计。每轮门禁仍单独 AND 检查，不能
  用平均数掩盖一个失败轮次。0% 错误率合法，缺失/NaN 不算 0。
* 远端初次和复用 session 都只回传本次新增的日期目录，不下载覆盖整个用例根目录；
  后续 trial 只刷新执行脚本和 `asr_inputs/`，不会删除远端上次日期目录或控制端
  其他服务节点的启动、清理日志。
* Excel 写入失败或抛异常会在本轮 `execution_result.json` 写
  `report_generated: false` 和失败原因，不能仅有 benchmark 成功就宣称报表成功。
* `para_set` exhaustive/TPE 读取 ASR 原生指标：单并发可直接使用
  `audio_throughput`、`error_rate_pct`、`e2el_p99_ms`；多并发需明确
  `c2.audio_throughput` 等带并发前缀的指标。重复执行取有效轮中位数。
  默认 objective 是 `audio_throughput` maximize，不使用 token throughput。
  ASR 用例失败时不能作为最优可行候选。

##### 自动并发摸高（可选）

ASR 不套用 LLM 的 TTFT/TPOT 目标。若运行 `--assess_test_type auto_assess`，需要
明确一个 ASR 延迟上限；每个 trial 只允许一个并发。例子（值仅示范）：

```yaml
benchmark:
  benchmark_type: asr
  asr:
    concurrency: [2]
    auto_assess:
      metric: e2el_p99_ms
      max: 1000          # ms，不代表已验收基线
      low: 0
      high: 16
```

生成 YAML 时修改 `benchmark.asr.concurrency`，不是无效的 `benchmark.perf.batch_size`。
缺失 ASR target、缺失指标、请求/质量失败时明确停止，不无条件循环。
摸高报告增加 `selected_concurrency`（兼容保留旧报告的 selected_batch_size）。重复 trial
使用同一服务 session，缓存条件仍是 service_lifetime，不是冷缓存；穷举并发更适合
延迟不单调或容易受缓存影响的测试。

##### bench-reporter（可选）

`XTASK_BENCH_REPORT=1` 时走 ASR 专用适配：一个并发/repeat 一条 payload。
通用字段只发送真正存在的 E2E/请求吞吐；S/D/I/N、错误率、音频吞吐、RTF、
首文本等放在 `toolSpecificConfig.asr`。不把错误率转换成 accuracy，不编造
TPOT/token/s，不套用 LLM SLA。归档保存 `dashboard_payloads.json`，发送失败沿用
可选 reporter 的本地 spool。服务端是否支持显示 ASR 扩展字段需另行验证；本次
只验证构造与 transport 调用，不声明远端 UI 已完成 ASR 接入。硬件数、server
版本未知就保留未知，不从报告文档推断已验证 TP。

#### 验证和第二阶段

```bash
python -m pytest -q tests/test_asr_benchmark.py tests/test_asr_reporting.py
```

测试使用本地模拟 HTTP/SSE 服务和包内原 client/scorer，覆盖同步/audio/chat、
并发/repeat、断流、HTTP 错误、空结果、评分 ID 对齐、门禁及 Excel 展开。
它验证客户端/工程接入，不代表真实 NPU 模型精度或性能验收。
AISBench API/Dataset/Evaluator 是暂缓的第二阶段工作，当前没有实现。后续可替换
执行器而保留数据、profile 和结果契约。阶段一会明确拒绝 `backend: aisbench`，
不虚报实现。

### Benchmark 基线校验规则

以下普通 LLM/图像编辑规则不适用于 `benchmark_type: asr`；ASR 使用
`benchmark.asr.gates`，并对每个并发/repeat 检查完整性与所有门禁，详见上方 ASR 小节。

- 性能：比较 `benchmark_perf_result.summary.output_tps_avg`。不配置有效 `baseline` 时，只要性能结果成功解析，基线步骤直接通过。
- 精度：比较解析出的 `benchmark_acc_result`。缺少有效 `baseline` 时，当前代码不会把精度用例判定为通过。
- `threshold` 是比例。`baseline: 100`、`threshold: 0.05` 的合法范围是 95 到 105。
- 因此，不同 benchmark 或数据集之间复制精度 baseline 前，先确认结果量纲是“百分数”还是“0 到 1”。
- 图像编辑 gedit（LLM-judge 0-10 分制）：baseline 为 SC/PQ 两维综合分（算术平均），
  如 36 号 `0.0973`（吞吐量纲，另见上文）、37 号 `3.925`、39 号 `4.56`（两次干净轮
  20261103_010342=4.70、20261103_132652=4.425 的均值，threshold 0.20）。gedit judge 打分
  波动较大（20 样本 + judge 非零温度），回填时以多次无外部干扰干净轮均值为准，避开与
  其他任务共卡的时间窗。

## 4. deployment 部分

### 单机

只保留一个 `instances` 元素：

```yaml
deployment:
  master_host: 11.87.191.75
  benchmark: {runner_instance: master-0, health_path: /health}
  instances:
  - name: master-0
    host: 11.87.191.75
    role: master
    rank_start: 0
    device_ids: [0, 1]
    local_port: 8800
    serving_extensiable_param:
      max_seqs_per_batch: 10
      enable_chunked_prefill: true
      enable_schedule_overlap: true
      enable_prefix_cache: false
      enable_shm: true
      block_size: 128
```

单机也使用 `deployment.instances`，只是列表中只有一个元素。这样单机、多机共用同一套调度逻辑。

### 普通实例字段字典

以下字段同时适用于单机和普通多机混部：

| 字段 | 新用例是否填写 | 通俗解释 | 当前行为和注意事项 |
| --- | --- | --- | --- |
| `deployment.master_host` | 是 | 这个普通 xLLM 服务的主节点 IP | benchmark endpoint 会取该 IP，并在相同 host 的实例中选择 `master`/`standalone` 的 `local_port`。它必须能匹配一个实例。 |
| `deployment.sync_workspace` | 多机建议 | 执行前是否把当前 xTask 工程同步到远端节点 | 默认 `true`。这是工程代码同步，不是模型目录同步。单机无需关心。 |
| `deployment.benchmark.runner_instance` | 建议 | 在哪个实例的容器里执行 benchmark | 值必须对应某个 `instances[].name`。不写时优先选 `master`、`standalone`、`prefill`，再退回第一个实例。 |
| `deployment.benchmark.health_path` | 建议 | 统一服务的健康检查路径 | 默认 `/health`，与推导出的 endpoint 拼接。不要在这里写完整 URL。 |
| 顶层/`instances[].hccl_if_port` | 同节点混部是 | 每个物理 runtime 的 HCCL 基础端口 | 顶层值是默认 43439。同一 host 上多个 `instances` 必须分别覆盖成不同端口；跨 host 的实例可以复用。 |
| 顶层/`instances[].progress_conn_port` | 端口冲突时填写 | 普通服务整组共用的主节点协调端口 | 优先使用 master/standalone 实例的显式值，其次使用顶层值，默认 9590；框架会把最终值统一下发给本组所有 runtime。启用 `enable_shm` 时，同一物理 host 上的不同逻辑组不能复用该端口。 |
| `instances[].name` | 是 | 物理启动实例的唯一名字 | 用于输出目录、脚本名、容器名、benchmark runner 选择；同一用例内不能重复。它不是模型名。 |
| `instances[].host` | 是 | 这个实例在哪台服务器启动 | 多机通过 SSH 连接非本机 host。相关 SSH 配置由工程的 HostExecutor/SSH 配置读取。 |
| `instances[].role` | 是 | 这个节点在普通多机服务中的身份 | 普通部署建议主节点写 `master`，其余写 `worker`；单机写 `master` 或 `standalone`。PD 角色另见后文。 |
| `instances[].rank_start` | 多实例建议 | 这个节点第一张卡的全局 rank | 普通部署的所有实例属于同一个 rank 组，后一节点要接着前面节点的卡数累计，例如 16 卡节点依次写 0、16、32。省略时按 YAML 顺序自动累计，显式写出更容易核对。 |
| `instances[].device_ids` | 是 | 这个节点实际占用哪些 NPU | 可写不连续卡号，例如 `[0, 2, 4, 6]`。框架按列表长度计算本节点卡数。 |
| `instances[].num_devices` + `start_device_id` | 仅兼容 | 用“起始卡号 + 连续卡数”代替 `device_ids` | 只有未写 `device_ids` 时使用。新用例优先写清楚 `device_ids`，不要两种方式混用。 |
| `instances[].local_port` | 是 | 该实例的 xLLM 本地监听端口起点 | 一个 N 卡 runtime 实际占用 `local_port` 到 `local_port + N - 1`。普通服务的 benchmark endpoint 使用主节点实例的起始端口。默认回退为顶层 `port`，再回退到 8800。 |
| `instances[].service.port` | 仅兼容 | `local_port` 的旧写法 | 只在未配置 `local_port` 时使用。 |
| `instances[].ipc_mode` | 特殊场景按需 | 容器 IPC 命名空间模式 | 同一 host 有多个 runtime 时框架默认使用 `host`。同一逻辑实例的同 host runtime 启用 `enable_shm` 后必须共享 IPC，不能配置 `private`。 |
| `instances[].shm_size` | 私有 IPC 按需 | 容器私有 `/dev/shm` 上限 | 仅在显式使用私有 IPC 时有意义；需按模型和并发量自行设置。共享 host IPC 时通常不填写。 |
| `instances[].serving_extensiable_param` | 按模型需要 | 传给这个 xLLM 实例的启动参数 | 详见“xLLM 启动参数”小节。实例级字段优先于兼容的顶层同名块。 |
| `instances[].readiness` | 按需 | 自定义如何判断该实例已经启动 | 不写时 xTask 根据角色检查日志关键字或进程；可改成命令或 HTTP 检查。 |

`readiness` 支持三种常见写法：

```yaml
# 1. 自定义命令，退出码为 0 就算就绪
readiness:
  type: command
  command: pgrep -f '[x]llm'

# 2. HTTP 返回 200 就算就绪
readiness:
  type: http
  url: http://127.0.0.1:8800/health

# 3. 继续使用日志检查，但自定义任一成功关键字
readiness:
  success_keywords: [Brpc Server started on port, 服务启动完成]
```

`success_keywords` 是“匹配到任意一个即成功”。远端检查默认等待时长可由兼容字段 `serving_ready_timeout` 控制，默认约 320 秒。

### 普通混部（跨节点或同节点）

在单机 YAML 上增加 worker；两台机器仍然是一个 xLLM 服务：

```yaml
deployment:
  master_host: 11.87.191.75
  sync_workspace: true
  benchmark: {runner_instance: master-0, health_path: /health}
  instances:
  - {name: master-0, host: 11.87.191.75, role: master, rank_start: 0,
     device_ids: [0,1,2,3], local_port: 8800,
     serving_extensiable_param: {max_seqs_per_batch: 10}}
  - {name: worker-0, host: 11.87.191.121, role: worker, rank_start: 4,
     device_ids: [0,1,2,3], local_port: 8800,
     serving_extensiable_param: {max_seqs_per_batch: 10}}
```

`rank_start` 是全局卡编号的起点；后一个节点要接着前一个节点累计。多机默认会从主节点同步工程；`xllm_test_060` 只按需检查关键资源，不会无条件整目录复制。模型目录仍需提前放到每台机器。

多个 runtime 也可以放在同一个物理 host，例如卡 `[0,1]` 和 `[2,3]` 分别启动两个容器。它们仍属于同一个普通 xLLM 服务，因此共享一个 `progress_conn_port`；但每个 runtime 必须配置不同的 `hccl_if_port`，并使用互不重叠的 `device_ids` 和 `local_port` 端口段。启用 `enable_shm` 时，xLLM 会在同 IP runtime 之间使用共享内存通信，因此这些容器必须共享 host IPC；框架会自动使用 `ipc_mode: host`。完整写法见 `19_samehost_mixed_aisbench_perf.yaml`。

普通部署的 ranktable 规则是：所有 `deployment.instances` 组成同一个 `default` 组；实例数大于 1 时，框架按物理 host 汇总本组卡号，同一 host 只生成一个 server 条目，同时保留每个 runtime 的全局 rank。只有一个实例时不自动生成 ranktable。

### 统一的 xLLM-Service 配置

PD 分离和普通实例统一路由均使用 `deployment.xllm_service` 描述 etcd 与
`xllm_master_serving` 的 IP、端口、启动重试和二进制兼容参数。部署模式仍分别为
`pd_disaggregated`、`service_routed`；模型计算拓扑仍使用 `pd_instances`、
`service_instances`，不能把它们当成 Service 进程列表。

历史 `deployment.pd_control`、`deployment.service_control` 在配置边界转换为
`deployment.xllm_service`，旧配置仍可读取；不要同时填写新旧配置块，即使内容相同
也会报冲突。参数搜索中的旧 dotted path 同样转换到新路径。

旧二进制保留单 Service 流程及历史 `pd_control/`、`service_control/` 日志目录。
新二进制已支持第一阶段双角色部署：Coordinator/Router 是独立进程，复用本用例在
对应 IP 上的模型引擎容器，不创建额外容器，也不参与设备/rank/world_size 计算。
详见下一节；仅改字段名与启用双角色部署是两个不同的变更。

### 新版 Service 第一阶段：复用引擎容器

现有扁平 `xllm_service` 无需批量重写。`service_binary_profile: auto` 探测到角色参数后，
自动在控制节点启动 1 个 Coordinator 和 1 个 Router：

| 监听项 | 扁平字段/默认派生 | 说明 |
| --- | --- | --- |
| Router HTTP | `http_server_port`（原值不变） | benchmark 仍访问原业务端口 |
| Coordinator RPC | `rpc_server_port`（原值不变） | Engine 从 etcd 发现控制面 |
| Coordinator HTTP | `coordinator_http_server_port`，默认 HTTP + 10000 | 可覆盖，不能冲突或越界 |
| Router RPC | `router_rpc_server_port`，默认 RPC + 10000 | 可覆盖，不能冲突或越界 |

`legacy` 不派生新监听端口，继续单 Service；`refactor` 使用新流程。`auto` 在规划阶段
为可能的新二进制检查派生端口，若冲突需明确修改端口或选择实际的 legacy profile，
不能跳过冲突检查。端口不会因占用而静默漂移。

需按 IP、数量或承载容器定制时，使用以下列表（本节配置已实现）：

```yaml
deployment:
  mode: pd_disaggregated
  xllm_service:
    service_binary_profile: auto
    etcd:
      host: 11.87.191.75
      client_port: 3389
      peer_port: 3390
      container_instance: prefill-0-node-0
    etcd_namespace: ""
    control_plane_shard_id: default
    startup_attempts: 2
    startup_wait_seconds: 3
    master_key_wait_seconds: 30
    coordinators:
      - name: coordinator-0
        host: 11.87.191.75
        http_server_port: 48888
        rpc_server_port: 38889
        container_instance: prefill-0-node-0
    routers:
      - name: router-0
        host: 11.87.191.121
        http_server_port: 38888
        rpc_server_port: 48889
        container_instance: decode-0-node-0
  benchmark:
    runner_instance: prefill-0-node-0
    router_instance: router-0
    health_path: /readyz
  # pd_instances 仍是独立的模型计算拓扑，省略部分沿用实际用例配置。
```

- `container_instance` 是展开后的 **node 名称**，不是 P/D 逻辑组名或实际 Docker 容器名。
  指定 IP 必须匹配本例已有引擎节点。显式列表遇到同 IP 多个容器时必须指定；扁平
  兼容模式优先选择该 host 上的 benchmark runner。第一阶段不支持无引擎节点的新 IP。
- 每类列表至少 1 个实例，名称全局唯一；`etcd`、`remote`、`archive` 是保留名。
  一个 shard 只有一个 active Coordinator leader，多个 Coordinator 是候选副本；Router
  可多个，不要求 1:1。多个 Router 必须用 `deployment.benchmark.router_instance` 明确
  选择一个业务入口，本阶段不自动增加 LB 或向所有 Router 平分请求。
- 列表内分别配置 HTTP/RPC 端口，不再同时填写扁平 `http_server_port`、`rpc_server_port`、
  `service_deployment_role` 或两个派生端口字段，避免歧义。共享 etcd 只启动一次，
  `etcd.host` 与成员 host 可不同；嵌套与旧扁平 etcd 端口同时填写时必须一致。
- Service 与 Engine 共用 `etcd_namespace`；第一阶段只支持 `control_plane_shard_id: default`。
  不支持外部 etcd 接管、named shard 模型订阅、动态扩缩容；`external_service` 模式仍不
  启动/清理这些组件。内部 token 统一从共享字段或 `XLLM_INTERNAL_TOKEN` 解析。
- 检查顺序：etcd → Coordinator 进程及本例 leader → Router 进程 → Engine 注册后
  Router `/readyz` 与 `/v1/models` 的实际模型。仅有 `/health` 200 不算新 Service 业务就绪。
  这些检查不等价于已经验证真实推理成功，性能/精度仍由 benchmark 验证。
- 每次执行采用独立运行 ID，日志写入 `xllm_service/<run-id>/<name>/xservice.log`，etcd
  日志独立；远端日志按 host 归档。每个 benchmark 轮次保存自己的日志快照，不覆盖历史轮次。
- 通过 PID、Linux start-time、PID namespace、executable 及 pidfd 清理本例拥有的进程，
  不用进程名批量 pkill。重试某个 Router/Coordinator 不删除 etcd 数据或影响其他成员。
  容器内需 Python 3.9+、Linux pidfd 支持（当前测试镜像满足）；未验证支持的运行环境
  应先做进程 helper 的 CPU 检查。`CLEAN_RESULTS=false` 保留完整运行环境。

### 多个普通实例统一路由

当目标是启动多个彼此独立的普通 xLLM 实例，并由一组 `xllm-service + etcd` 统一分发请求时，使用 `mode: service_routed`。不要使用普通 `instances`：普通 `instances` 的多个元素始终属于同一个分布式 xLLM 实例。

```yaml
deployment:
  mode: service_routed
  master_host: 11.87.191.83
  xllm_service:
    host: 11.87.191.83
    etcd_client_port: 13389
    etcd_peer_port: 13390
    http_server_port: 13888
    rpc_server_port: 13889
  benchmark: {runner_instance: default-0-node-0, health_path: /health}
  service_instances:
  - name: default-0
    group_master_host: 11.87.191.83
    progress_conn_port: 9590
    nodes:
    - {name: default-0-node-0, host: 11.87.191.83,
       device_ids: [0,1], local_port: 48000, hccl_if_port: 43439}
  - name: default-1
    group_master_host: 11.87.191.83
    progress_conn_port: 9690
    nodes:
    - {name: default-1-node-0, host: 11.87.191.83,
       device_ids: [2,3], local_port: 48100, hccl_if_port: 43539}
```

每个 `service_instances[]` 是一个完整的普通逻辑实例，角色保持 xLLM `DEFAULT`；它可以只有一个 node，也可以由同一 host 或不同 host 上的多个 node 组成。xTask 自动为每个实例注入 `--enable_service_routing=true` 和统一的 `--etcd_addr`，benchmark 只访问 `xllm_service.host:http_server_port`。

| 字段 | 作用和约束 |
| --- | --- |
| `deployment.mode` | 固定写 `service_routed`，只在明确选择此模式时启用普通实例统一路由。 |
| `xllm_service.host` | etcd 和 `xllm_master_serving` 所在 host，必须对应一个实际 node。 |
| `xllm_service.etcd_client_port` / `etcd_peer_port` | etcd 客户端和 peer 端口，默认 3389/3390。 |
| `xllm_service.http_server_port` / `rpc_server_port` | 统一 HTTP 入口和内部 RPC 端口，默认 38888/38889。 |
| `xllm_service.service_deployment_role` | 扁平新模式允许 `COORDINATOR`（自动补 Router）；不能用唯一 `ROUTER` 替代控制面。列表内角色由所在列表确定。 |
| `xllm_service.service_binary_profile` | 默认 `auto`；`legacy` / `refactor` 在普通实例路由与 PD 模式下使用相同语义。 |
| `xllm_service.master_key_wait_seconds` | 控制面启动后等待 MASTER key 或 `/readyz` ready 的最长秒数，默认 30。 |
| `service_instances[].name` | 独立普通逻辑实例名，全局唯一。 |
| `service_instances[].group_master_host` | 本逻辑实例的主节点，必须对应本组某个 node；默认取本组第一个 node。 |
| `service_instances[].progress_conn_port` | 本逻辑实例所有 node 共用；同 host 上的不同逻辑实例必须不同。 |
| `service_instances[].nodes` | 组成该逻辑实例的物理 node；卡号、本地端口段和同 host 的 HCCL 端口不能冲突。 |
| `deployment.benchmark.runner_instance` | 填实际 `nodes[].name`，用于选择执行 benchmark 的容器。 |

`enable_service_routing`、`etcd_addr`、`enable_disagg_pd` 和 `instance_role` 由框架管理，不应放入 `serving_extensiable_param`。不同普通逻辑实例分别计算 world size、rank 和 ranktable，不会与其他逻辑实例合并。完整真实用例见 `21_service_routed_samehost_multi_instance_perf.yaml`。

### 工程与测试资源同步

普通多机、普通实例统一路由和 PD 多机共用下面的前置处理：

```yaml
deployment:
  sync_workspace: true
  resource_sync:
    enabled: true
    source: /export/home/models/xllm_test_060
    target: /export/home/models/xllm_test_060
    max_workers: 4
    timeout: 3600
```

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `sync_workspace` | `true` | 把主节点当前 xTask 工程及当次输出所需目录同步到用例涉及的远端 host。设为 `false` 时，远端必须已经有同路径、同版本工程。 |
| `resource_sync` | 启用 | 负责 `/export/home/models/xllm_test_060` 中 benchmark 运行所需资源的快速检查与增量补齐。也可以整体写 `false` 关闭。 |
| `resource_sync.enabled` | `true` | 是否启用测试资源检查。 |
| `resource_sync.source` | `/export/home/models/xllm_test_060` | 主节点资源根目录。 |
| `resource_sync.target` | 与 `source` 相同 | 远端服务器上的资源根目录，通常保持同路径，便于容器挂载到 `/xllm_test_060`。 |
| `resource_sync.max_workers` | `4` | 最多并行检查/同步多少台远端主机。 |
| `resource_sync.timeout` | `3600` | 单台主机资源同步操作允许的最长秒数。 |

这里的“增量补齐”不是比较整个大目录：框架先根据 benchmark 和 YAML 中出现的 `/xllm_test_060/...` 路径推导本次真正需要的组件，再检查少量稳定关键文件。关键文件都在就跳过；缺少时只同步对应组件。

### xLLM 启动参数 `serving_extensiable_param`

这个字段名保留了工程的历史拼写：`serving_extensiable_param`。框架不会维护一份完整的 xLLM 参数白名单，而是把其中每一项转换为 `--key=value` 传给当前安装的 xLLM。因此“YAML 能解析”不等于“当前 xLLM 版本一定支持”，最终以生成的 `start_xllm_serv_*.sh` 和 xLLM 启动日志为准。

常用字段如下：

| 字段 | 通俗作用 | 使用提醒 |
| --- | --- | --- |
| `max_memory_utilization` | 限制 xLLM 最多使用多大比例的设备内存 | 常写 0 到 1 的小数，例如 `0.85`。 |
| `max_seqs_per_batch` | 单批最多容纳多少条序列 | 常见搜参项；不等于 benchmark 总请求数。 |
| `max_tokens_per_batch` | 单批最多容纳多少 token | 需要与输入/输出长度、显存容量一起考虑。 |
| `enable_chunked_prefill` | 是否把较长 Prefill 拆块执行 | 布尔值写 `true/false`。 |
| `enable_schedule_overlap` | 是否开启调度重叠优化 | 是否有收益与模型、版本和部署模式有关。 |
| `enable_prefix_cache` | 是否开启前缀缓存 | 输入无公共前缀时未必有收益。 |
| `enable_shm` | 是否启用共享内存相关路径 | 多进程配置通常沿用已验证用例。 |
| `enable_heterogeneous_pd` | 是否允许 P/D 使用不同并行拓扑 | 非 MLA 模型的 P/D 卡数或并行规模不一致时，P 和 D 两侧都要设为 `true`；20 号用例保持各逻辑组 TP2 以满足 Qwen3-8B 的显存要求，同时覆盖同 host 多逻辑实例和同一逻辑实例多 node。 |
| `block_size` | KV cache block 大小 | 改动可能影响显存和调度效率。 |
| `dp_size` | 数据并行规模 | 必须与实际卡数、模型并行配置匹配。 |
| `ep_size` | 专家并行规模 | MoE 模型常用，必须与模型及总卡数匹配。 |
| `enable_graph` | 是否开启图模式 | 不支持的模型/形状应关闭。 |
| `enable_graph_mode_decode_no_padding` | Decode 图模式是否使用 no-padding | 一般与 `enable_graph` 配套，并以已验证模型配置为准。 |
| `draft_model` | 投机推理 draft/MTP 模型路径 | 该路径同样要在使用它的每台服务器上存在。 |
| `num_speculative_tokens` | 每轮投机生成 token 数 | 需要与 `draft_model` 和当前 xLLM 版本匹配。 |
| `speculative_algorithm` | 选择投机推理算法 | 例如旧用例中的 `Eagle3`；必须与 draft 模型和当前 xLLM 版本匹配。 |
| `max_concurrent_requests` | xLLM 服务端允许同时处理的请求上限 | 与 benchmark 的并发不是同一层：前者限制服务，后者决定测试端施加多少并发。 |
| `backend: vlm` | 告诉 xLLM 使用视觉模型后端 | 视觉模型常见配置。 |
| `max_encoder_cache_size` | 视觉/编码器缓存容量 | 需结合图片数量和设备内存调整。 |
| `limit_image_per_prompt` | 单请求最多允许多少张图片 | 应不小于 benchmark 的 `image_num`。 |
| `enable_intralayer_addnorm` | 层内 addnorm 优化开关 | 属于版本/模型相关优化，优先从已验证同模型用例复制。 |
| `enable_interlayer_addnorm` | 层间 addnorm 优化开关 | 同上。 |
| `enable_split_rmsnorm_rope` | 拆分 RMSNorm/RoPE 优化开关 | 同上。 |
| `rank_tablefile` | 手工指定 ranktable 文件 | 如果一个 rank 组的实例都已配置手工路径，框架跳过自动生成；对未手工配置的实例，自动流程会注入合并后的路径。 |

YAML 布尔值建议写小写 `true/false`，列表按 YAML 列表写。不要把复杂 Python 表达式当作值传入。

### PD 分离

PD 可以简单理解为：Prefill 负责“读输入”，Decode 负责“生成输出”，两者组成一个服务。一个逻辑 P/D 实例下面可以再放多个服务器节点：

```yaml
deployment:
  mode: pd_disaggregated
  master_host: 11.87.191.75
  xllm_service:
    host: 11.87.191.75
    etcd_client_port: 3389
    http_server_port: 38888
    rpc_server_port: 38889
  benchmark: {runner_instance: prefill-0-node-0, health_path: /health}
  pd_instances:
  - name: prefill-0
    role: prefill
    group_master_host: 11.87.191.75
    progress_conn_port: 9590
    transfer_port_base: 26000
    disagg_pd_port: 7777
    nodes: [{name: prefill-0-node-0, host: 11.87.191.75,
             device_ids: [0,1], local_port: 48000, hccl_if_port: 43439}]
  - name: decode-0
    role: decode
    group_master_host: 11.87.191.121
    progress_conn_port: 9690
    transfer_port_base: 27000
    disagg_pd_port: 7787
    nodes: [{name: decode-0-node-0, host: 11.87.191.121,
             device_ids: [0,1], local_port: 48000, hccl_if_port: 43439}]
```

`xllm_service.host` 是 etcd 和统一 HTTP 服务所在节点，必须对应一个实际 node。`pd_instances[].name` 只标识逻辑 P/D 实例，`role` 只表示它是 P 还是 D；不要再写 `pd_group`。同一个逻辑实例有两个及以上 node 时，才为这一组生成和合并 ranktable；不同 P/D 组互不合并。

### PD 顶层和控制面字段

| 字段 | 新用例是否填写 | 通俗解释 | 当前行为和注意事项 |
| --- | --- | --- | --- |
| `deployment.mode` | 是 | 声明这是 PD 分离服务 | 固定写 `pd_disaggregated`。PD YAML 只允许层级 `pd_instances` 写法，不能再用扁平 `instances` 写 P/D。 |
| `deployment.master_host` | 是 | 整个任务的主控 host | 一般与 `xllm_service.host` 一致，并应是实际 node 之一。 |
| `xllm_service.host` | 是 | etcd 和 `xllm_master_serving` 在哪台服务器的服务容器内启动 | 必须能在展开后的某个 node host 中找到对应 runtime。统一 benchmark endpoint 也使用这个 IP。 |
| `xllm_service.etcd_client_port` | 建议 | P/D 实例注册和查询 etcd 的端口 | 默认 3389，并传给 xLLM 的 `--etcd_addr`。 |
| `xllm_service.etcd_peer_port` | 建议 | etcd 自身 peer 监听端口 | 默认 3390。单节点 etcd 也需要该监听端口不冲突。 |
| `xllm_service.http_server_port` | 是 | 对外统一 xLLM 服务端口 | benchmark 最终请求 `http://<xllm_service.host>:<http_server_port>`，默认 38888。它不是 node 的 `local_port`。 |
| `xllm_service.rpc_server_port` | 建议 | `xllm_master_serving` 内部 RPC 端口 | 默认 38889，确保控制节点端口未占用。 |
| `xllm_service.startup_attempts` | 否 | 控制面启动失败最多重试几次 | 默认 2；每次重试前会清理旧的 etcd 和 `xllm_master_serving` 进程。 |
| `xllm_service.startup_wait_seconds` | 否 | 每次启动控制面后先等多少秒再检查 | 默认 3。它不是全部 P/D 实例注册完成的超时。 |
| `xllm_service.master_key_wait_seconds` | 否 | 等待控制面锚点就位的最长秒数 | 默认 30。框架优先用 `etcdctl`（gRPC）轮询 etcd 的 `XLLM:SERVICE:MASTER`，回退到 etcd v3 REST 网关 `/v3/kv/range`；refactor 二进制还接受 `http://<host>:<http_server_port>/readyz` 返回 `status=ready`。 |
| `xllm_service.service_deployment_role` | 否 | 扁平兼容模式角色 | 新二进制默认补 1C+1R；角色列表不能再填此字段。 |
| `xllm_service.service_binary_profile` | 否 | 控制面二进制兼容模式 | 默认 `auto`：启动前探测 `xllm_master_serving --help` 是否支持 `--service_deployment_role`。`legacy` 对应 GitHub 旧版（不传 role flag；MASTER key 探测为**非阻塞**告警——旧版二进制启动即自动写 MASTER，但用 3s 短租约，硬性门控易在租约边界抖动，故沿用历史的 `pgrep + etcd /health + 无 fatal 日志` 基础健康检查）；`refactor` 对应 JD 新版（传 `COORDINATOR`，MASTER key/`/readyz` 为**硬性门控**，用于捕获默认 ROUTER 不选举的回归）。 |
| `xllm_service.master_service_key` | 否 | 控制面锚点 etcd key | 默认 `XLLM:SERVICE:MASTER`。仅在自定义 `control_plane_shard_id` 时可能需要改成带 shard 前缀的 key。 |
| `xllm_service.transfer_port_base` | 仅兼容 | 所有 P/D 逻辑组的 transfer 端口默认值 | 默认 26000。旧用例继续有效；同 host 多 P/D 组的新用例应改在各 `pd_instances[]` 上分别配置。 |
| `xllm_service.no_proxy_hosts` | 否 | 额外加入 `NO_PROXY` 的 host 列表 | localhost、控制面和所有实例 host 会自动加入；这里只需补充环境中特殊地址。 |
| `deployment.benchmark.runner_instance` | 建议 | 在哪个物理 node 的容器里跑 benchmark | 层级 PD YAML 中应填写一个实际的 `nodes[].name`，例如 `prefill-0-node-0`，不是逻辑实例名 `prefill-0`。不写时框架会自动选择；写了却不匹配任何 node 时会退回首个 runtime，但配置含义会变得不明确。 |
| `deployment.benchmark.health_path` | 建议 | 统一服务健康检查路径 | 默认 `/health`，拼在 `xllm_service` 的 HTTP endpoint 后。 |

注意：`deployment.benchmark.endpoint` 即使手工填写，当前服务组逻辑也不会把它作为权威地址；普通部署从 `master_host + local_port` 推导，PD 从 `xllm_service.host + http_server_port` 推导。**仅当顶层 `external_service` 开启时**，该字段（以及 `external_service.endpoint`）才成为被测服务地址。

xLLM-Service 在 `service architecture refactor`（约 `55905ea` / `3fe20c6`）之后引入了 `COORDINATOR` / `ROUTER` 双角色：二进制默认是 `ROUTER`，只有 `COORDINATOR` 会通过 `elect_master()` 写入 `XLLM:SERVICE:MASTER`。GitHub 旧版 main（如 `72ab598` / `73d5e00`）没有该 flag，启动时会自动写 MASTER（用 3s 短租约 + 后台 keepalive 续约）。xTask 默认 `service_binary_profile=auto` 自动探测并兼容两类二进制：探测到 refactor 二进制时注入 `--service_deployment_role=COORDINATOR` 并把 MASTER key/`/readyz` 作为**硬性**控制面门控（捕获默认 ROUTER 不选举的回归）；探测到 legacy 二进制时不传 role flag，MASTER key 探测退化为**非阻塞**告警，沿用历史的基础健康检查（`pgrep + etcd /health + 无 fatal 日志`），避免 3s 租约边界抖动误杀旧版二进制。也可显式设为 `legacy` 或 `refactor`。

### PD / Service 路由场景编译 xLLM-Service

PD 场景的 `xllm_master_serving` 可以由 xTask 单独自动编译和归档，不改变原有
xLLM wheel 的编译、归档和加载流程：

```bash
bash tools/prepare_xllm_service.sh
```

源码默认准备在 `xllm-service/`；也可以通过 `XLLM_SERVICE_SOURCE_DIR` 指定已有
checkout，通过 `XLLM_SERVICE_REPO_URL` 指定仓库地址。脚本会执行 Service 官方要求的
`prepare.sh`、CMake 和 `xllm_master_serving` 构建，并将产物原子切换到：

```text
pkg/xllm/service_binary/current/xllm_master_serving
pkg/xllm/service_binary/current/librust_tokenizers.so
```

`download_xllm.sh` 会优先加载该 `current` 版本；旧的
`pkg/xllm/service_binary/xllm_master_serving` 直放布局仍兼容。ETCD 不由该编译流程
生成：xLLM-Service 的 `third_party/etcd_cpp_apiv3` 是 C++ 客户端库，不包含 `etcd`
或 `etcdctl` 服务端程序。PD 启动所需的 ETCD 仍按原逻辑从本地归档或 `xllm_pack`
加载。

运行时依赖准备会按部署模式选择安装范围：普通实例不会安装
`xllm_master_serving` 或 ETCD；PD / `service_routed` 只在
`xllm_service.host` 对应的控制面容器中安装，其他 runtime 只准备 xLLM wheel
及其依赖。直接手工执行 `download_xllm.sh` 时，未设置开关仍保留旧行为；如需显式
跳过控制面组件，可使用 `XLLM_INSTALL_SERVICE=0 bash tools/download_xllm.sh`。

`download_xllm.sh` 需要的 `xllm_pack.tar.gz` 会优先从服务器本地归档获取，不再
每次从外网 wget（约 585MB）。脚本按顺序探测以下位置：

1. 环境变量 `XLLM_PACK_LOCAL_DIR` 指定的目录；
2. `/export/home/models/xllm_test_060`（宿主机目录在容器内同路径直挂时）；
3. `/xllm_test_060`（资源同步挂载布局）。

只要任一位置存在 `xllm_pack.tar.gz` 就直接解压使用；本地归档缺失、损坏或解压
后没有生成 `xllm_pack` 目录时，自动回退到原来的外网 `wget` 下载，保证行为
向后兼容。要固定使用某个归档目录，可在执行前导出：

```bash
export XLLM_PACK_LOCAL_DIR=/export/home/models/xllm_test_060
```

定时任务如需同时编译 xLLM 和 Service，可显式执行：

```bash
bash tools/cron_daily_build.sh build service_build test
```

默认 `build` 阶段仍只执行原有 `prepare_xllm.sh`，避免 Service 的 CMake/Rust 构建
改变既有 xLLM 构建任务。仓库、分支、提交号日志和定时任务的完整说明见第 7 节。

### `pd_instances` 和 `nodes` 字段

| 字段 | 新用例是否填写 | 通俗解释 | 当前行为和注意事项 |
| --- | --- | --- | --- |
| `pd_instances[].name` | 是 | 一个逻辑 P 或 D 实例的唯一名字 | 例如 `prefill-0`、`decode-1`。框架自动用它生成内部 `pd_group`；禁止手工写 `pd_group` 或 `group`。 |
| `pd_instances[].role` | 是 | 这个逻辑实例负责 P 还是 D | 只能是 `prefill` 或 `decode`。`name` 负责区分第几个实例，`role` 负责说明功能，两者不重复。 |
| `pd_instances[].group_master_host` | 多 node 建议 | 这个逻辑 P/D 实例内谁是主节点 | 默认取 `nodes` 的第一个 host。组内 world size、启动主从关系和 readiness 都使用它；它必须对应本组实际 node。 |
| `pd_instances[].hccl_if_port` | 仅作默认值 | 本逻辑实例各 node 的 HCCL 默认端口 | 默认继承顶层值或 43439。HCCL 实际是 node 级配置；同一 host 上有多个 node 时，必须在各 `nodes[].hccl_if_port` 覆盖成不同值。 |
| `pd_instances[].progress_conn_port` | 同 host 多组是 | 本逻辑实例的主节点协调端口 | 默认继承顶层值或 9590。同一逻辑实例的 node 必须一致；多个逻辑组的 `group_master_host` 相同时必须不同。启用 `enable_shm` 的逻辑组只要共享任一物理 host，也必须使用不同值，避免共享内存对象重名。 |
| `pd_instances[].transfer_port_base` | 同 host 多组是 | 本逻辑实例的数据传输端口起点 | 默认回退 `xllm_service.transfer_port_base` 或 26000。实际端口为 `transfer_port_base + rank`，因此同 host 的不同逻辑组必须使用不重叠端口段。 |
| `pd_instances[].disagg_pd_port` | 同 host 多组是 | 本逻辑实例的 PD 调度 bRPC 监听端口 | 默认 7777，并传给 xLLM 的 `--disagg_pd_port`。同一逻辑实例的所有 node 必须一致；不同 P/D 逻辑实例的 `group_master_host` 相同时必须不同。 |
| `pd_instances[].ipc_mode` / `shm_size` | 特殊场景按需 | 本逻辑实例各 node 的容器 IPC 配置 | 同 host 存在多个 runtime 时默认使用 host IPC。同一逻辑实例需要在同 host node 间使用共享内存时禁止配置 `private`；不需要跨容器共享内存时可显式配置 private IPC 和相应 `shm_size`。 |
| `pd_instances[].serving_extensiable_param` | 是 | 本逻辑实例所有 node 默认共用的 xLLM 参数 | P 和 D 可以写不同参数块。 |
| `pd_instances[].nodes` | 是 | 组成这个逻辑实例的物理启动 node 列表 | 至少一个 node。多个 node 可以使用同一个物理 host，只要卡号和本地端口段不重叠。 |
| `nodes[].name` | 建议 | 物理启动节点的唯一名字 | 不写时自动生成 `<逻辑实例名>-node-<序号>`。所有 P/D node 名全局不能重复。它会成为实际脚本、容器和输出目录名。 |
| `nodes[].host` | 是 | 物理服务器 IP | 必须可 SSH 访问，并具备模型、NPU 和容器环境。 |
| `nodes[].device_ids` | 是 | 该 node 使用的卡号 | 同一 P/D 组的 world size 等于组内所有 node 的卡数之和。 |
| `nodes[].rank_start` | 可省 | 该 node 在本 P/D 组内的起始 rank | 省略时框架按 node 顺序从 0 累计。注意 P 组、D 组分别从各自的 0 开始，不跨组累计。 |
| `nodes[].local_port` | 是 | 该物理 node 的 xLLM 本地端口起点 | 它用于 node 自身启动，不是对外统一 endpoint。N 卡 node 占用从该值开始的 N 个连续端口；同一服务器上多个 node 的端口段不能重叠。 |
| `nodes[].hccl_if_port` | 同 host 多 node 是 | 该物理 node/runtime 的 HCCL 基础端口 | 默认继承逻辑实例、顶层或 43439。同一 host 上的所有 runtime 必须不同；不同 host 可使用相同值。 |
| `nodes[].progress_conn_port` | 不建议 | 逻辑实例协调端口的 node 级写法 | 应统一写在 `pd_instances[]`。node 即使显式填写也必须与本逻辑实例其他 node 完全一致，否则配置校验会拒绝。 |
| `nodes[].serving_extensiable_param` | 按需 | 只覆盖这个 node 的 xLLM 参数 | 先继承逻辑实例参数，再用 node 内同名键覆盖。只把确实需要差异化的键放在 node 层。 |
| `nodes[].readiness` | 按需 | 自定义该 node 的启动判断 | 与普通实例的 `readiness` 格式相同。 |

一个 xLLM PD 用例始终只有一个对外 endpoint，但可以有多个 Prefill 实例、多个 Decode 实例，每个 P/D 实例又可以由一个或多个 node 组成：

```text
一个 endpoint
├── prefill-0
│   ├── node-0
│   └── node-1
├── prefill-1
│   └── node-0
└── decode-0
    ├── node-0
    └── node-1
```

ranktable 以逻辑实例为组分别处理。一个逻辑组内即使多个 node 位于同一物理 host，也会按 host 合并成一个 server 条目并保留各 node 的 rank。上例只给 `prefill-0` 和 `decode-0` 自动生成 ranktable；单 node 的 `prefill-1` 跳过自动生成。P/D 各组之间不合并。

同一物理 host 上所有 runtime 的 `device_ids`、xLLM 监听端口段和 PD transfer 端口都必须互不重叠；`xllm_service` 的 etcd、HTTP、RPC 端口也会参与冲突检查。多个 P/D 逻辑组共机的完整写法见 `20_pd_samehost_multi_instance_perf.yaml`。

`startup_policy` 虽然在部分历史 YAML 中出现，但当前调度代码始终并行启动所有实例，它暂时不能真正切换串行/并行策略。新用例不需要依赖这个字段。

### GEdit 自服务多服务部署（deployment.encoder / dit / judge）

gedit（Qwen-Image-Edit）两段式协议需要 Text Encoder、DiT（精度用例还需要 LLM judge）
等多个独立 xLLM 服务。自服务模式把这些服务键写在 `deployment` 下，与
`deployment.benchmark` 同级（调度器自动拉起各服务容器，评测结束后自动回收；
benchmark 默认在 DiT 服务容器内执行）：

```yaml
deployment:
  encoder:
    device_ids: [7]
    port: 18007                       # Text Encoder 单卡固定端口
    master_node_addr: 127.0.0.1:29798
    health_path: /health
    hccl_if_port: 43438
    serving_extensiable_param:
      task: embed
      enable_return_mm_full_embeddings: 1
      # ...其余参数见 36/37 号用例
  dit:
    device_ids: [0, 1, 2, 3, 4, 5]
    port: 19000                       # 起始端口；每卡一个进程，按 rank 连续自增
    master_node_addr: 127.0.0.1:19757
    health_path: /health
    hccl_if_port: 43432
    readiness_timeout: 1800           # DiT 6 卡加载较慢（默认 900s）
    serving_extensiable_param:
      backend: dit
      cfg_size: 2
      sp_size: 3
      # ...
  judge:                              # 仅精度用例（37）需要
    device_ids: [12]
    port: 18008
    master_node_addr: 127.0.0.1:29810
    model_path: /export/home/models/Qwen2.5-VL-7B-Instruct
    hccl_if_port: 43500
    serving_extensiable_param:
      model_id: Qwen2.5-VL-7B-Instruct
      # ...
```

生效条件（`xtask/core/gedit_service.py`，对其他用例零影响）：

- `benchmark_type: image_edit`（36 号性能用例），或 `benchmark_type: aisbench` 且
  `benchmark.acc.api_type` 以 `gedit` 开头（37 号精度用例）；其他 `benchmark_type`
  即使出现这些键也不会被消费。
- `deployment` 至少出现 encoder/dit/judge 中的一个。
- `external_service` 关闭；外部模式沿用旧的仅探测路径，不自启服务。

端口与挂载约定：

- 端口按“每卡一个 xLLM 进程、端口 = `port` + rank”连续自增：DiT 写 `port: 19000`
  即占用 19000-19005，无需逐卡列举；encoder 固定 18007，judge 固定 18008。
- 容器自动挂载 `/etc/hccn.conf`、Ascend driver、add-ons、`npu-smi` 和
  `pkg/xllm/vendors/custom_xllm_math`（缺一会出现 aclInit 507008 core dump），
  YAML `volumes` 可在此基础上追加。
- `hccl_if_port`（`HCCL_IF_BASE_PORT`）经验规则：同一 host 上多个服务的 base 间隔
  至少 60，避免一个服务的 base 落进另一个服务的 NIC 通道窗口。已验证组合是
  DiT 43432 / encoder 43438 / judge 43500；judge 曾用 43442 落进 DiT 的窗口，
  导致 DiT 首次 allreduce 报 aicpu ERR02005 / acl 507018。

## 5. 搜索写法

### auto_assess：只摸高 batch_size

本小节描述普通 LLM 的 TTFT/TPOT 与 `benchmark.perf.batch_size`。ASR 改用
`benchmark.asr.auto_assess` 的显式延迟上限和 `concurrency`，不套用这些参数；
配置见第 3 节“JoyAI-ASR 原始评测”的“自动并发摸高”。

```yaml
benchmark:
  perf:
    batch_size: 40
    bs_assess: {low_thrld: 8, high_thrld: 72}
```

```bash
python main.py --case_group <case> --assess_test_type auto_assess \
  --target_ttft 1000 --target_tpot 50
```

字段和命令行参数含义：

| 配置 | 通俗解释 | 默认/当前行为 |
| --- | --- | --- |
| `benchmark.perf.batch_size` | 第一次真实测试从哪个 batch size 开始 | 必须位于合理的上下界内。后续值由算法自动生成。 |
| `benchmark.perf.bs_assess.low_thrld` | 摸高区间下界 | 不写时默认 0。它是搜索边界，不一定会作为一个测试点执行。 |
| `benchmark.perf.bs_assess.high_thrld` | 摸高区间上界 | 不写时默认初始 `batch_size * 2`。同样不保证直接执行该值。 |
| CLI `--target_ttft` | 允许的 P90 首 token 时延上限，单位 ms | 默认值很大，建议真实摸高时明确填写。 |
| CLI `--target_tpot` | 允许的 P90 每输出 token 时延上限，单位 ms | 默认 100 ms。 |

每轮读取 `ttft_p90` 和 `tpot_p90`：两项都优于目标时提高下界；任一项不满足时降低上界；然后取新上下界的整数中点继续，直到区间收敛。`tpot_p90` 与目标相差不超过 1 ms 时也会提前停止。

它只适用于性能测试，并且只改变 benchmark 的 `batch_size`。同一基础用例的一轮 `auto_assess` 会复用已经启动的容器、xLLM 进程、普通多机/PD 控制面和服务拓扑；每轮只重新生成并执行 benchmark。因此：

- 单机、普通多机、PD 都能使用同一套摸高逻辑。
- 摸高期间不能改变模型、卡、xLLM 启动参数、P/D 数量或 node 数量。
- scheduler session 发生异常后会停止整轮摸高，最后统一清理服务。

### para_set exhaustive：所有组合都跑

```yaml
assess_test_param:
  benchmark.perf.batch_size: [1, 2, 4]
  deployment.instances[*].serving_extensiable_param.max_seqs_per_batch: [8, 16]
assess_search:
  strategy: exhaustive
  objective: {metric: output_tps_total, direction: maximize}
```

上例会运行 `3 × 2 = 6` 组。每组都会重新启动服务；结果目录使用 `_sg1`、`_sg2`，并保存本组实际参数。

`exhaustive` 就是笛卡尔积全量测试。三个参数的候选数量分别为 2、3、4 时，总组合数为 `2 × 3 × 4 = 24`。没有配置 `assess_search` 时，`para_set` 也默认采用 `exhaustive`。

| `assess_search` 字段 | exhaustive 下是否需要 | 作用 |
| --- | --- | --- |
| `strategy` | 可省 | 明确写 `exhaustive` 最易读。 |
| `objective.metric` | 可省 | 指定用哪个性能结果排最佳组合，默认是 `output_tps_total`。也可以显式指定其他 summary 指标。 |
| `objective.direction` | 可省 | `maximize` 表示越大越好，`minimize` 表示越小越好，默认 `maximize`。 |
| `constraints` | 按需 | 先判断一组结果是否满足上下限；不满足的组合会标为 `constraint_failed`，不会成为可行最优组合。 |

每组参数会独立走正常调度生命周期，即准备、启动、benchmark、清理；它不像 `auto_assess` 那样复用 xLLM session，因为 `para_set` 允许搜索 xLLM 启动参数和拓扑。

### para_set TPE：最多跑指定组数

```yaml
assess_test_param:
  benchmark.perf.batch_size: [1, 2, 4, 8, 16]
  deployment.instances[*].serving_extensiable_param.max_seqs_per_batch: [8, 16, 32]
assess_search:
  strategy: tpe
  max_trials: 6
  startup_trials: 2
  seed: 42
  objective: {metric: output_tps_total, direction: maximize}
  constraints:
    ttft_p90: {max: 1000}
    tpot_p90: {max: 50}
```

TPE 不是凭空生成任意数字，它仍从 `assess_test_param` 提供的离散候选中选择组合，只是不一定把笛卡尔积全部跑完。

TPE 会把目标指标纳入持久化 study 名称；例如默认目标切换为 `output_tps_total` 后，不会继续复用旧的 `output_tps_avg` 搜索 trial。

| `assess_search` 字段 | 默认值 | 通俗解释 |
| --- | --- | --- |
| `strategy` | `exhaustive` | 使用 TPE 时必须写 `tpe`。 |
| `max_trials` | 20 | 最多实际测试多少种不重复参数组合。最终次数不会超过笛卡尔积的总组合数。 |
| `startup_trials` | `min(5, max_trials)` | 开始的多少轮先做探索，不依赖已完成结果建立的 TPE 判断。必须是非负整数且不能大于 `max_trials`。 |
| `seed` | 42 | TPE 采样随机种子。固定配置和种子有助于复现选择顺序，但真实性能波动仍可能改变后续选择。 |
| `objective.metric` | `output_tps_total` | TPE 学习和挑选参数的目标指标。 |
| `objective.direction` | `maximize` | `maximize` 找最大值，`minimize` 找最小值。吞吐通常 maximize，时延通常 minimize。 |
| `constraints` | 空 | 对一个或多个结果指标设置 `min`、`max`。违反约束的 trial 会被重罚并标记为不可行。 |

例如“TTFT P90 不超过 1000 ms、TPOT P90 不超过 50 ms，在满足条件的组合里最大化输出吞吐”就是：

```yaml
assess_search:
  strategy: tpe
  max_trials: 20
  startup_trials: 5
  seed: 42
  objective:
    metric: output_tps_total
    direction: maximize
  constraints:
    ttft_p90: {max: 1000}
    tpot_p90: {max: 50}
```

`objective.metric` 与 `direction` 的关联很直接：框架先从每组 benchmark 的 summary 中按 `metric` 取数，再根据 `direction` 比较大小。约束不满足时，这个 objective 数值仍会记录，但该组不算可行最优组合。

### `assess_test_param` 路径怎么写

推荐总是写完整路径，框架会严格检查路径已经存在；拼错字段不会静默生成一组未修改的 YAML。

| 写法 | 选择范围 | 示例 |
| --- | --- | --- |
| 普通点路径 | 普通字典中的一个现有字段 | `benchmark.perf.batch_size` |
| `instances[*]` | 普通部署的全部实例 | `deployment.instances[*].serving_extensiable_param.max_seqs_per_batch` |
| `instances[0]` | 普通部署列表中下标为 0 的实例 | `deployment.instances[0].device_ids` |
| `instances[name=...]` | 普通部署中指定名字的实例 | `deployment.instances[name=worker-0].serving_extensiable_param.enable_graph` |
| `instances[role=...]` | 普通部署中指定角色的所有实例 | `deployment.instances[role=worker].serving_extensiable_param.max_seqs_per_batch` |
| `pd_instances[*]` | 全部逻辑 P/D 实例 | `deployment.pd_instances[*].serving_extensiable_param.max_seqs_per_batch` |
| `pd_instances[0]` | 下标为 0 的逻辑 P/D 实例 | `deployment.pd_instances[0].serving_extensiable_param.enable_graph` |
| `pd_instances[name=...]` | 指定逻辑 P/D 实例 | `deployment.pd_instances[name=prefill-0].serving_extensiable_param.max_seqs_per_batch` |
| `pd_instances[role=...]` | 所有 Prefill 或所有 Decode 实例 | `deployment.pd_instances[role=decode].serving_extensiable_param.max_seqs_per_batch` |

候选值必须是非空列表。字段本身是列表时，要把“每个完整列表”作为一个候选：

```yaml
assess_test_param:
  # 搜索 master-0 整体使用 2 卡还是 4 卡
  deployment.instances[name=master-0].device_ids:
  - [0, 1]
  - [0, 1, 2, 3]
```

这就是“支持整体搜索 `device_ids`，不支持 `device_ids[0]`”的含义：框架可以把整个列表从 `[0, 1]` 换成 `[0, 1, 2, 3]`，但通用路径解析器不能只修改卡号列表里的第一个元素。同理，不支持 `deployment.pd_instances[0].nodes[0].device_ids[0]` 这种任意深度列表索引。

还需注意：

- 选择器只专门支持 `deployment.instances[...]` 和 `deployment.pd_instances[...]` 这两层；`nodes[0]` 不是通用选择器。
- 目标字段必须已经出现在基础 YAML 中。不能用搜参路径临时发明一个新的 xLLM 参数。
- 旧式裸字段名如 `batch_size`、`max_seqs_per_batch` 仍兼容，但存在“同时修改多个匹配实例”的隐式行为，新用例不要依赖它。
- 搜 `benchmark.perf.batch_size` 时，如果没有显式同时搜索 `benchmark.perf.num_prompts`，每个候选自动设置 `num_prompts = batch_size * 4`；显式搜索 `num_prompts` 时，以显式候选为准。
- 也可以把点路径改写成层级字典，但完整点路径更容易复制、检查和记录。

层级写法示例：

```yaml
assess_test_param:
  deployment:
    pd_instances[name=prefill-0]:
      serving_extensiable_param:
        max_seqs_per_batch: [8, 16]
```

### PD 拓扑搜索

用 `assess_topologies` 预先列出完整合法拓扑，再把名字放进 `assess_test_param.deployment_topology`：

```yaml
assess_topologies:
  topo_1p1d: {pd_instances: [...]}
  topo_2p1d_d2n: {pd_instances: [...]}
assess_test_param:
  deployment_topology: [topo_1p1d, topo_2p1d_d2n]
  benchmark.perf.batch_size: [1, 2]
  deployment.pd_instances[role=prefill].serving_extensiable_param.max_seqs_per_batch: [8, 16]
assess_search: {strategy: exhaustive, objective: {metric: output_tps_total, direction: maximize}}
```

拓扑候选必须各自能独立运行；不要分别笛卡尔搜索 P 数量、D 数量和 node 数量，否则很容易组合出没有 IP、端口冲突或 rank 不合法的配置。

拓扑搜索的完整规则：

- `assess_topologies` 只用于层级 PD YAML，并且基础部署不能再写扁平 P/D `deployment.instances`。
- 每个拓扑名字下面目前只允许一个字段：`pd_instances`。`xllm_service`、`master_host`、资源同步、benchmark runner 等公共配置仍放在基础 `deployment` 中。
- `assess_test_param.deployment_topology` 的每个值必须正好匹配一个 `assess_topologies` 名字。
- 每个候选拓扑都要写完整的 P/D 实例、node、host、卡和端口，且必须单独合法。
- 框架先替换本轮完整拓扑，再应用同一轮的普通搜索参数。因此与拓扑联合搜索的路径必须在每个候选拓扑里都能匹配。
- `deployment.pd_instances[role=prefill]...` 会修改该拓扑里的所有 Prefill 逻辑实例；如果 2P 拓扑需要两个 P 使用不同值，应改用 `[name=prefill-0]`、`[name=prefill-1]`，并确保所有拓扑都有对应名字，或者把差异直接写进各拓扑。

P/D 实例个数、每个实例的 node 数、每个 node 的卡数可以搜索，但稳妥方式是把它们作为一个完整 `deployment_topology` 候选一起切换。这样 IP、端口、rank 和 xLLM 并行参数能作为一套合法配置管理。

### 搜索目标和约束可用指标

普通 LLM 的 `objective.metric` 和 `constraints` 的键来自 benchmark 性能结果的
`ResultSummary`。ASR 使用原生 `audio_throughput`、`error_rate_pct`、`e2el_p99_ms`
等指标，单并发重复轮取有效中位数，多并发需 `c2.audio_throughput` 等前缀；默认
最大化 `audio_throughput`，不是 `output_tps_total`。ASR 的详细规则见第 3 节。

普通 LLM 当前可用字段为：

| 指标 | 含义 |
| --- | --- |
| `total_requests` | 成功解析到的总请求数 |
| `failed_requests` | 失败请求数 |
| `concurrency` | 结果记录的并发值 |
| `max_concurrency` | 结果记录的最大并发值 |
| `request_throughput` | 每秒完成请求数 |
| `input_tokens_avg` | 平均输入 token 数 |
| `output_tokens_avg` | 平均输出 token 数 |
| `output_tps_total` | 汇总输出 token 吞吐，`para_set` 默认最大化该指标 |
| `output_tps_avg` | 平均输出 token 吞吐，可显式作为其他搜索目标 |
| `summary_tps_total` | 汇总总 token 吞吐 |
| `ttft_avg`、`ttft_p90`、`ttft_p99` | 首 token 时延的平均值/P90/P99 |
| `tpot_avg`、`tpot_p90`、`tpot_p99` | 每输出 token 时延的平均值/P90/P99 |
| `e2el_avg`、`e2el_p90` | 端到端时延的平均值/P90 |

不要使用 Excel 为方便展示而产生的列名或旧别名，例如 `output_token_throughput`；搜索器按上表中的内部字段精确取值。

### 搜索输出怎么看

每个参数组合使用短名字 `<基础用例名>_sg1`、`_sg2`，不会把所有参数和值拼进目录名。每个 `_sgN` 目录下的 `assess_test_param.yaml` 保存：

- `current_combination`：本轮实际选择的参数。
- `assess_test_param`：基础用例中的完整搜索空间，只包含搜索相关内容。
- `resolved_topology`：使用拓扑搜索时，本轮最终采用的完整 PD 拓扑。

汇总文件：

- `search_summary.yaml`：策略、目标、约束、执行数量、可行数量、最佳组合和全部 trial 摘要。
- `search_trials.csv`：一行一个实际执行组合，便于排序和二次分析。
- TPE 还会保存 `search_state.db`，用于记录 Optuna study 状态。

常见计数字段含义：

| 字段 | 含义 |
| --- | --- |
| `planned_trials` | exhaustive 事先计划的组合数；auto_assess 是初始队列计划数，不代表最终动态轮数。 |
| `executed_trials` | 已经真正执行过的 trial 数。 |
| `completed_trials` | 没有调度/结果缺失错误的 trial 数；它可能包含违反约束的 trial。 |
| `failed_trials` | 启动、benchmark 或结果解析等原因导致失败的 trial 数。 |
| `constraint_failed_trials` | 测试执行完成，但一个或多个 `constraints` 不满足的 trial 数。 |
| `feasible_trials` | 执行成功并满足全部约束、可参与最佳组合选择的 trial 数。 |

## 6. 保留字段和当前限制

下面这些字段在历史配置或代码接口中仍有保留，但当前 YAML 用例不依赖它们，新用例无需填写，本文不再展开：

- AISBench 性能：`mode`、`summarizer`、`concurrency`、`total_requests`、`timeout`、`warmup_requests`。
- AISBench 通用模板：`dump_extract_rate`、`enable_debug`、`merge_ds`、`max_num_worker`、`pressure`。
- 历史顶层字段：`additional_config`，以及与实例配置重复的顶层 `block_size`、`port`、`num_devices`、`start_device_id`、`world_size`。
- 部署兼容字段：`deployment.startup_policy`、`deployment.benchmark.endpoint`、`instances[].service.port`。

现有 YAML 中仍能看到、但写新用例时需要注意的限制只有以下几项：

- AISBench 性能输出上限实际取 `perf.output_len`；保留的 `perf.max_out_len` 建议与它写成相同值。
- AISBench 精度输出上限实际取 `acc.output_len`；`acc.input_len`、`acc.max_out_len` 仅保留。
- AISBench 精度的 `ignore_eos`、思考模式等生成参数应放在 `generation_kwargs`，不要依赖 `acc` 下的同名直接字段。
- `benchmark.perf/acc.dump_eval_details` 仅保留；当前 AISBench 精度如需明细，应使用顶层 `dump_eval_details: true`。
- EvalScope 性能当前固定 `ignore_eos: true`。
- ASR 当前仅支持 `benchmark.asr.backend: legacy_infer` / `scoring_profile: joyai_legacy`；
  AISBench ASR 后端和标准 CER/WER profile 尚未实现，不把文字 SSE 当成输入流式 ASR。
- 多实例当前始终并行启动，服务 endpoint 始终由部署拓扑自动推导。

搜参不支持 `nodes[0]`、`device_ids[0]` 等任意深度列表路径，这属于搜索路径规则，详见第 5 节。

## 7. 编译与定时执行

### 7.1 单独编译 xLLM

在 xTask 工程根目录执行：

```bash
bash tools/prepare_xllm.sh
```

编译日志会打印到当前终端，也可以重定向：

```bash
bash tools/prepare_xllm.sh > outputs/xllm_compile.log 2>&1
```

编译成功后，xLLM wheel、Ascend `vendors/` 和运行时动态库会自动归档到
`pkg/xllm/`，后续用例通过 `download_xllm.sh` 自动加载，不需要手工复制。

如果本轮编译生成了独立的 `flash_linear_attention_npu*.whl`，还会将其归档到
通用三方包目录 `pkg/xllm/third_whl/`。后续其他三方 wheel 也可以放在该目录，
无需修改运行环境安装脚本或按包名配置白名单。

运行容器初始化时，`download_xllm.sh` 在安装 xLLM 后使用同一个 Python，
一次离线安装 `third_whl/` 直属目录中的全部 `*.whl` 文件
（`--force-reinstall --no-deps --no-index`），不自动下载依赖或升级镜像中的
torch、torch_npu 等依赖。目录不存在或没有 wheel 时自动跳过，原流程不变。
FLA wheel 自带 Python 模块和 OPP，无需手工合并进 xLLM wheel。

旧版本没有 FLA 子模块、跳过 FLA 编译或未生成 FLA wheel 时，只跳过 FLA
归档。编译前仅清理源码 `dist/` 下的历史 FLA wheel；本轮成功后若没有新 FLA
wheel，会仅清掉 `third_whl/` 中旧的 `flash_linear_attention_npu*.whl`，保留
其他三方 wheel，后续运行时仍会安装这些包。编译失败时保留上一轮已归档包。
FLA 编译目录若存在多个版本/芯片的 wheel，归档步骤仍明确报错；通用归档目录
支持多个三方 wheel，安装时全部交给 pip，版本冲突或安装失败会报错退出。

脚本还会在启动 Docker 编译前打印本次实际使用的仓库、分支和完整 commit：

```text
[source-version] xLLM repo: ...
[source-version] xLLM branch: ...
[source-version] xLLM commit: <commit-sha>
```

编译源码的仓库和分支可以分别通过环境变量或位置参数指定：

```bash
# 位置参数优先于 XLLM_BRANCH
bash tools/prepare_xllm.sh release/v0.11.0

# 等价写法
XLLM_BRANCH=release/v0.11.0 bash tools/prepare_xllm.sh
```

默认仓库地址为：

```text
git@coding.jd.com:xLLM_AI/xllm.git
```

需要使用其他仓库时，将下面的占位地址替换为实际存在且当前用户有权限访问的仓库：

```bash
# 仅示例，不能直接使用 team/xllm.git
XLLM_REPO_URL=git@coding.jd.com:<组织>/<项目>.git \
XLLM_BRANCH=release/v0.11.0 \
bash tools/prepare_xllm.sh
```

未指定分支时，已有 checkout 更新当前分支；新目录 clone 远端默认分支。
如果已有 `xllm/` 的 `origin` 与期望仓库地址不同，脚本会先校验目标仓库和
分支，校验成功后删除当前工程下的旧 `xllm/` 并重新 clone，避免旧分支、子模块
或未跟踪文件混入编译。目标仓库不可访问时会在删除前失败并保留旧目录。

注意：本地存在未提交修改、分支与远端无法 `--ff-only` 快进，或子模块存在
冲突性修改时，脚本会失败，不会强制覆盖本地代码。当前 xLLM 编译镜像在
`tools/prepare_xllm.sh` 中定义；如需更换镜像，需要调整该脚本的 `IMAGE` 配置。

### 7.2 单独编译 xLLM-Service

在 xTask 工程根目录执行：

```bash
bash tools/prepare_xllm_service.sh
```

也可以按同样方式查看终端日志或重定向到文件：

```bash
bash tools/prepare_xllm_service.sh > outputs/xllm_service_compile.log 2>&1
```

成功后会自动归档并原子更新：

```text
pkg/xllm/service_binary/current/xllm_master_serving
pkg/xllm/service_binary/current/librust_tokenizers.so
```

后续 PD 或 Service 路由用例会自动从 `current` 版本加载。Service 编译前会打印：

```text
[source-version] xLLM-Service repo: ...
[source-version] xLLM-Service branch: ...
[source-version] xLLM-Service commit: <commit-sha>
[source-version] xLLM-Service third_party/xllm commit: <commit-sha>
```

默认 Service 仓库地址为：

```text
git@coding.jd.com:xLLM_AI/xllm-service.git
```

指定仓库和分支：

```bash
# 位置参数优先于 XLLM_SERVICE_BRANCH
bash tools/prepare_xllm_service.sh release/v0.11.0

# 等价写法
# 仅示例，不能直接使用 <组织>/<项目>.git
XLLM_SERVICE_REPO_URL=git@coding.jd.com:<组织>/<项目>.git \
XLLM_SERVICE_BRANCH=release/v0.11.0 \
bash tools/prepare_xllm_service.sh
```

Service 的 `XLLM_SERVICE_UPDATE_SOURCE` 默认是 `1`：已有仓库会按目标分支
更新；仓库地址不一致时先校验再删除当前工程下的 `xllm-service/` 并重新 clone。
设置为 `0` 时只使用现有 checkout，不拉取、不切换分支，也不允许更换仓库：

```bash
XLLM_SERVICE_UPDATE_SOURCE=0 \
bash tools/prepare_xllm_service.sh release/v0.11.0
```

Service 的编译镜像可通过 `XLLM_SERVICE_BUILD_IMAGE` 覆盖；ETCD/etcdctl 不由
Service 编译脚本生成，仍按运行时依赖流程准备。

### 7.3 手动触发编译和用例执行

`cron_daily_build.sh` 支持 `build`、`service_build`、`test` 三个阶段，按命令
行顺序执行；任一阶段失败会停止后续阶段：

```bash
MAIN_ARGS="--case_name qwen3_8b_xllm_perf_i1024_o1024_bs1 --host_addr 11.87.191.75" \
bash tools/cron_daily_build.sh build service_build test
```

`MAIN_ARGS` 是传给 `main.py` 的参数，可以是单用例、用例组或其他受支持的参数。
也可以直接为两个编译阶段分别指定仓库和分支：

```bash
XLLM_REPO_URL=git@coding.jd.com:xLLM_AI/xllm.git \
XLLM_BRANCH=release/v0.11.0 \
XLLM_SERVICE_REPO_URL=git@coding.jd.com:xLLM_AI/xllm-service.git \
XLLM_SERVICE_BRANCH=release/v0.11.0 \
MAIN_ARGS="--case_name qwen3_8b_xllm_perf_i1024_o1024_bs1 --host_addr 11.87.191.75" \
bash tools/cron_daily_build.sh build service_build test
```

阶段可以任意组合并调整顺序，例如：

```bash
bash tools/cron_daily_build.sh build
bash tools/cron_daily_build.sh service_build
bash tools/cron_daily_build.sh test
bash tools/cron_daily_build.sh build test
```

### 7.4 定时编译和用例执行

定时任务由 `install_crontab.sh` 安装到当前用户的 crontab。当前用户需要具备
crontab 权限；脚本会优先使用普通用户 crontab，必要时尝试
`sudo crontab -u <用户>`。

只定时编译 xLLM 和 Service：

```bash
env CRON_TIME="01 22 * * *" \
STAGES="build service_build" \
bash tools/install_crontab.sh
```

只定时执行用例：

```bash
env CRON_TIME="01 22 * * *" \
STAGES="test" \
MAIN_ARGS="--case_name qwen3_8b_xllm_perf_i1024_o1024_bs1 --host_addr 11.87.191.75" \
bash tools/install_crontab.sh
```

定时编译后执行用例，并固定两个源码分支：

```bash
env CRON_TIME="01 22 * * *" \
STAGES="build service_build test" \
XLLM_BRANCH="release/v0.11.0" \
XLLM_SERVICE_BRANCH="release/v0.11.0" \
MAIN_ARGS="--case_name qwen3_8b_xllm_perf_i1024_o1024_bs1 --host_addr 11.87.191.75" \
bash tools/install_crontab.sh
```

`CRON_TIME` 使用标准五段格式“分 时 日 月 周”。例如 `01 22 * * *` 表示每天
22:01 执行。`STAGES` 可以是 `build`、`service_build`、`test` 的任意组合；安装
脚本会把 `XLLM_BRANCH`、`XLLM_SERVICE_BRANCH`、`MAIN_ARGS` 和
`MAIN_ARGS_LIST` 固化到 cron 行中。需要在 cron 中使用自定义仓库地址时，应在
安装后用 `crontab -e` 为任务行补充 `XLLM_REPO_URL` 和
`XLLM_SERVICE_REPO_URL`，因为仓库地址不会被当前安装脚本自动固化。

日志位置：

- xLLM 编译：`outputs/build.log`，历史文件为 `outputs/log/build_<时间>.log`。
- xLLM-Service 编译：`outputs/log/service_build_<时间>.log`。
- 框架总日志：`outputs/case_framework.log`，历史文件为
  `outputs/log/case_framework_<时间>.log`；包含阶段编排、编译输出和用例执行输出。
- xLLM 的 commit 会出现在 `build.log` 和框架日志；xLLM-Service 的 commit 只出现在
  Service 编译日志和框架日志。

用例结果、xLLM 运行日志和 benchmark 产物仍按具体用例配置归档到 `outputs/` 下
对应目录。查看当前定时任务：

```bash
sudo crontab -u <用户名> -l
```

取消本工具安装的定时任务：

```bash
bash tools/install_crontab.sh --uninstall
```

## 8. VBench Wan2.2 I2V

VBench 是独立的视频生成质量评测类型，不能把它写成 AISBench 的
`api_type`。VBench 用例应设置 `metrics: acc`（与普通精度用例一致，旧写法
`metrics: vbench` 仍兼容）和 `benchmark.benchmark_type: vbench`，并配上
`benchmark.vbench` 配置块，xTask 会为它创建独立的 runner 容器：

- xLLM 容器只负责启动 Wan2.2 I2V 推理服务并占用 NPU。
- VBench runner 只负责读取输入、调用 `/v1/video/generation`、保存 MP4 和执行质量评测。
- 普通 AISBench、EvalScope、文本/视觉精度和性能用例不会创建这个 runner，也不会走 VBench 分支。

结果分支判定在 `xtask/benchmarks/vbench/task_gate.py` 的 `is_vbench_task`：要求
`benchmark.benchmark_type: vbench` 且 `benchmark.vbench` 是映射块；`metrics` 本身
不参与判定，因此新写法 `metrics: acc` 和旧写法 `metrics: vbench` 都能被识别，
而普通精度/SWE-bench/性能用例因 `benchmark_type` 不同不会被误判。

可复制的完整示例是 `tests/e2e/xllm/examples/yaml_authoring/25_single_vbench_wan_i2v.yaml`；
107 条 mini 子集示例是 `30_single_vbench_wan_i2v_mini.yaml`（显式把 `full_info` 指向
mini JSON，不再设 `generation.limit`，并用顶层 `benchmark_timeout: 86400` 放大
docker exec 超时）。
VBench 工具、评测脚本及模型权重必须提前装入 runner 镜像或通过挂载提供；xTask 不会在测试期间
下载 VBench 数据集和权重。`camera_motion` 等维度还可能使用 Torch Hub 缓存，离线环境也应提前准备。
未配置 `image_home`/`image_name` 时，runner 默认使用
`quay.io/jd_xllm/xllm-ai:8.1.rc1-a3-openeuler22.03-py3.10`。

### 8.1 输入和输出路径

四个路径字段 `full_info`、`image_root`、`cache_dir`、`evaluation_script` 全部可以省略。
省略时使用统一默认布局：

```text
/export/home/models/xllm_test_060/vbench_data/     # 共享数据集（所有 vbench 用例只读复用）
├── full_info.json
└── data/                                          # full_info 中 image_name 对应的图片

<xTask 工程>/outputs/vbench_data/cache/             # 工程内共享缓存（容器内 /workspace/outputs/vbench_data/cache/）
                                                   # 维度模型、解码等可复用缓存

<xTask 工程>/xtask/benchmarks/vbench/               # 框架源码（容器内 /workspace/xtask/benchmarks/vbench/）
├── run_i2v.py                                     # 视频生成驱动
└── evaluate_vbench_i2v_bench.py                   # VBench 质量评测器（随仓库版本化）
```

`full_info` 是 VBench 原始 JSON，`image_root` 是图片根目录；二者可以写宿主机绝对路径，
也可以写当前 xTask 工程内的路径。若路径位于 xTask 工程挂载范围，脚本会自动换成容器内的
`/workspace/...` 路径。`cache_dir` 与 `evaluation_script` 同理，显式指定时必须保证 runner
对应的宿主机路径可见。

每次调用都会在 case 目录创建一个新的日期目录：

```text
outputs/<case_name>/
├── run_vbench_test.sh
└── YYYYMMDD_HHMMSS/
    ├── videos/*.mp4
    ├── *_full_info_videos.json
    ├── generation_metrics.json
    ├── *_eval_results.json
    └── *_final_score.json
```

`generation_metrics.json` 记录生成/断点续跑/失败统计；`*_eval_results.json` 是逐维度原始结果；
`*_final_score.json` 是 VBench 的 raw、归一化加权分数及 `quality_score`、`i2v_score`、
`total_score`。`temporal_flickering` 单独保留，不计入 Total Score。

### 8.2 VBench 配置字段

```yaml
metrics: acc
benchmark:
  benchmark_type: vbench
  vbench:
    mode: generate_and_evaluate   # generate / evaluate / generate_and_evaluate
    # 以下四个路径字段全部可省略，默认值见 8.1 节的共享布局
    full_info: /path/to/full_info.json
    image_root: /path/to/images
    cache_dir: /path/to/outputs/vbench_data/cache
    evaluation_script: /path/to/evaluate_vbench_i2v_bench.py
    resolution: 16-9
    name: wan2_2_xllm
    check: true
    dimensions: [i2v_subject, i2v_background, camera_motion]
    generation:
      size: "832*480"
      num_inference_steps: 40
      guidance_scale: 1.5
      num_frames: 81
      fps: 16
      seed: 42
      request_timeout: 1800
      limit: 107
    evaluation:
      baseline: null
      threshold: 0.10
```

`generate` 只生成视频并输出 manifest，`evaluate` 只读取已有的带 `video_list` 的 manifest，
`generate_and_evaluate` 先生成再评测。`limit` 仅用于试跑时限制样本条数；删除后按 full_info
全量执行。`dimensions` 省略时使用评测脚本默认维度。若配置 `evaluation.baseline`，xTask
会以 `total_score` 按 `threshold` 做基线校验；不配置 baseline 时只校验生成/评测结果完整性。

### 8.3 单机和多机规则

单机 VBench 仍只有一个 xLLM runtime 和一个 VBench runner。多机 VBench 的 xLLM runtime
按原有 `deployment.instances`/PD 拓扑启动，benchmark runner 由
`deployment.benchmark.runner_instance` 选定；case 输出目录只同步到 runner 所在 host，
不会把 MP4 和共享 `vbench_data` 缓存复制到其他 xLLM worker。远端 runner 完成后，xTask 会把其输出
下载回控制机，再执行统一结果归档和 baseline 校验。

`CLEAN_RESULTS=true` 时按现有生命周期清理 xLLM runtime、VBench runner 和相关服务；
`CLEAN_RESULTS=false` 时两类容器和服务都保留，便于调试，输出文件仍保留在 case 目录。

### 8.4 容器与运行环境（行为说明）

**容器环境变量全量透传**：`ContainerConfig.environment` 中配置的所有键值对（此前仅
proxy 相关变量生效）现在会原样注入容器。历史上写过但从未生效的环境变量配置，
从本版本起会真正进入容器环境——升级后请检查现有 YAML/引擎配置中是否有本不打算
生效的 `environment` 字段。proxy 变量（`http_proxy`/`https_proxy`/`no_proxy`/`NO_PROXY`）
仍走原有的特殊处理（自动补 `no_proxy` 白名单）。

**CANN 环境**：VBench runner 生成的 `run_vbench_test.sh` 会在启动时 source
`/usr/local/Ascend/ascend-toolkit/set_env.sh`（存在时），保证 `torch_npu` 在
import 链路（transformers 的 NPU 探测）中干净加载。评测维度模型本身仍默认跑 CPU。

**xLLM 共享内存**：xLLM 以 `--ipc=host` 运行，其 POSIX SHM 对象
（`/dev/shm/xllm_<port>_*`）在宿主机上按端口命名，且启动时会校验布局大小。
历史上崩溃残留的旧对象会让下次同端口启动报
`incompatible shared memory layout`。xTask 现在会在清理阶段自动删除本用例端口
对应的残留对象；多机场景下远端宿主机的残留对象无法自动清理，出现该报错时需手工删除。

**模型权重转换**：modelers.cn 的原生 Wan2.2-I2V-A14B 权重需转换为 diffusers 目录
结构后 xLLM 才能加载。转换工具已固化在
`xtask/benchmarks/vbench/tools/convert_wan22_diffusers.py`：

```bash
python3 xtask/benchmarks/vbench/tools/convert_wan22_diffusers.py \
    --model-root /export/home/models/wan2_2
```

脚本只转换 `high_noise_model/`、`low_noise_model/` 两个 transformer；
`model_index.json`、`scheduler/`、`text_encoder/`、`tokenizer/`、`vae/`
需从 HuggingFace `Wan2.2-I2V-A14B-Diffusers` 拉取放入同一目录，
并在根目录补 `config.json`（`transformers_version: null` 可跳过框架的
transformers 版本对齐安装）。

## 9. 外部已有服务（不启动 xLLM）

默认关闭。不写 `external_service`、写成 `false`，或写成 `enabled: false` 时，
行为与现有端到端用例完全相同：拉容器、编译/下载 xLLM、启动服务、再跑
benchmark。

开启后，xTask **不启动** xLLM、`xllm_master_serving`、etcd，也不占用被测机器上的
昇腾卡。它只做数据集准备、生成 benchmark 脚本、起 **client 容器**、对给定地址做
健康探测、跑测试并汇总结果。

```yaml
external_service: true          # 也可写成 mapping，见下
served_model_name: Qwen3-8B
model_path: /export/home/models/Qwen3-8B   # AISBench/EvalScope tokenizer；VBench 生成可不填
metrics: perf
dataset_name: gsm8k
benchmark:
  benchmark_type: aisbench
  perf: { ... }
deployment:
  benchmark:
    endpoint: http://10.20.30.40:8800
    health_path: /health
```

或：

```yaml
external_service:
  enabled: true
  endpoint: https://gateway.example/inference/svc-x
  health_path: /v1/models      # 空字符串表示跳过探测
  api_key: EMPTY               # EvalScope / ASR 客户端使用；AISBench 路径暂不透传
```

> AISBench 路径用域名 / https / 带子路径的 endpoint 见下文「AISBench / EvalScope」一段；
> 其中 `api_key` 目前未写入 AISBench 模型配置，网关要求鉴权时需另行处理。

| 字段 | 说明 |
| --- | --- |
| `external_service` | `true` / mapping。缺省或 `false` = 端到端。 |
| `endpoint` | 权威地址。解析顺序：`external_service.endpoint` → `host`+`port` → `deployment.benchmark.endpoint` → 顶层 `host_addr`+`port`。 |
| `health_path` | 拼在 **origin**（不含 API path）后探测。默认 `/health`；HTTP 2xx 即通过。空 = 跳过。 |
| `deployment.instances` / `xllm_service` 等 | 开启后忽略并打 warning，不会去那些 host 起服务容器。 |

**AISBench / EvalScope：** 仍起一台 engine 镜像作为 client，并下载对应 bench 工具；不执行 `download_xllm.sh`。EvalScope 使用 origin + path 作为 `--api-url` 前缀。AISBench 分两种写法：

- endpoint 是**纯 IP + http + 无子路径**（如 `http://11.87.191.75:8800`）时，沿用 `host_ip`/`host_port`，网关 path 不写入配置，脚本与旧版本逐字节一致。
- endpoint 是**域名、https 或带子路径**（如 `http://glm53-flash-test.jd.local:8800`、`https://gw.jd.local/inference/svc`）时，改写 AISBench 模型配置的 `url` 字段，`host_ip` 保持模板默认 `localhost`。原因是 AISBench 只用 `ipaddress.ip_address()` 校验 `host_ip`、仅额外放行 `localhost`，域名会在 build model 阶段以 `UTILS-CFG-002` 失败；而 `url` 非空时 AISBench 会忽略 `host_ip`/`host_port`，域名、https、子路径都可表达。该模式要求 `served_model_name` 非空（否则 AISBench 会去探测 `v1/models`）。

注意 `url` 模式下的已知限制：https 仅解决 scheme 拼接，内网私有 CA 仍需在 client 容器注入 `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`；若 https 网关必须经代理，AISBench 推理路径的 aiohttp session 未开 `trust_env`，需要另行处理。

**ASR / legacy_infer：** 只起客户端容器，不安装/启动 xLLM，不加载服务端模型权重
或 tokenizer；使用归档中的原客户端和评分器。`api: audio` 请求
`/v1/audio/transcriptions`，`api: chat` 请求 `/v1/chat/completions`；endpoint 经
解析为保留路径前缀的 `/v1` 基址，`external_service.api_key` 可传给客户端。
本地日期归档与 Excel 链接指向客户端归档主机，不能据此外部模式获取远端模型
启动日志或独立证明其 TP/CTC/Suffix 配置。完整参数、缓存与评分口径见第 3 节 ASR 小节。

**VBench / Wan2.2：** 只起 `benchmark.vbench.image_home` / `image_name` 指定的独立 runner，不创建 dummy xLLM 容器。默认请求 `http://<origin>/v1/video/generation`；若 `endpoint` 已带 path，则原样作为视频 API。`mode: evaluate` 不要求活服务、不做健康探测。NPU watchdog 在外部模式关闭。

**GEdit / image_edit：** 外部模式不自启 encoder/dit/judge 自服务容器；图像生成端点由
`endpoint` 解析结果追加 `/image/generation` 得到（见
`xtask/benchmarks/evalscope/generate_image_edit_script.py` 的 `_image_edit_endpoint`），
encoder 地址仍按 `benchmark.perf.encoder_url` / 环境变量 `GEDIT_ENCODER_URL` /
默认 `http://127.0.0.1:18007` 解析。

搜 `serving_extensiable_param`、卡号、PD 拓扑会被拒绝；只允许搜 benchmark 请求参数
（普通 LLM 如 `batch_size`，ASR 如 `concurrency`）。

完整示例：`26_external_aisbench_perf.yaml`、`27_external_vbench_wan_i2v.yaml`、
`40_external_joyai_asr_legacy_acc_perf.yaml`、`41_external_joyai_asr_legacy_repeats.yaml`。

## 10. 常用执行命令

普通单用例：

```bash
python main.py --case_name 01_single_aisbench_perf --host_addr 11.87.191.75
```

ASR 外部服务单轮及两轮归档验证（先修改 YAML 中 endpoint 为可用服务）：

```bash
python main.py --case_name 40_external_joyai_asr_legacy_acc_perf
python main.py --case_group 41_external_joyai_asr_legacy_repeats
```

第二条命令同时生成全局和用例组 Excel。ASR 自部署入口及准备条件见第 3 节。

多机普通用例的 IP 已经写在 YAML 中、需要编译yaml修改，不要用 `--host_addr` 试图整体替换多机拓扑：

```bash
python main.py --case_name 05_multihost_aisbench_perf
```

并发摸高和 exhaustive/TPE 搜参建议使用 `--case_group`。当 `tests/case_group/<名字>` 不存在时，框架会把这个名字当成单个基础 YAML：

```bash
python main.py --case_group 09_single_auto_assess \
  --assess_test_type auto_assess --target_ttft 1000 --target_tpot 50

python main.py --case_group 10_single_para_set_exhaustive \
  --assess_test_type para_set

python main.py --case_group 11_single_para_set_tpe \
  --assess_test_type para_set
```

搜索完成后，重点查看 `outputs/<case_group>/search/<基础用例名>/search_summary.yaml` 和 `search_trials.csv`。前者汇总最佳组合和成功/失败数量，后者一行记录一次实际测试。

## 11. 写完后的检查清单

- 先确认 `metrics`、`benchmark_type` 和对应的配置块一致：`metrics: perf` 配
  `benchmark.perf`（`benchmark_type: image_edit` 的 gedit 性能同样如此）；普通精度
  与 SWE-bench 新格式配 `benchmark.acc`；VBench 配 `benchmark.vbench`；ASR
  `metrics: acc,perf` 配 `benchmark.asr`（不拆成 acc/perf 配置块）。执行路由只看
  `benchmark_type`，`metrics: acc` 下是否走 SWE 分支由 `benchmark.acc` 的
  `mode`/`runner` 字段决定。
- 确认模型路径在所有自部署服务节点存在，数据资源路径在 benchmark runner 可见。
- ASR 确认原客户端/数据归档、清单数量、context、评分 profile、音频时长上限、
  并发/重复和门禁单位正确；不把 `joyai_legacy` 错误率冒充纯 CER/标准 WER，
  不将服务生命周期内重复音频的性能宣称为冷缓存。
- 单机通常只有一个 instance；普通部署用 `instances`，可跨 host 或在同一 host 拆成多个小 runtime；PD 只用 `pd_instances`。
- 多机每个 host 的 SSH、容器运行环境、NPU 卡号和端口都可用。
- 同一 host 上检查 `device_ids`、`local_port` 端口段、HCCL/协调端口和 PD transfer 端口没有冲突。
- 非 PD 多机的 `rank_start` 要按全局卡数累计；PD 可按逻辑实例分别累计或省略让框架计算。
- 搜索路径必须指向 YAML 中已经存在的字段；不要使用 `nodes[0]`、`device_ids[0]` 这种通用列表路径。
- 对接已有服务时打开 `external_service`，写清 endpoint；未打开时不要依赖 `deployment.benchmark.endpoint`。
- `para_set` 的 case group 内不要混用 exhaustive 和 TPE；每个组合的完整配置会保存在输出目录，旧用例不会被改写。
