# AISBench 协议一致性 + 受限解码评测设计文档

## 1. 背景与目标

大模型推理服务升级（新引擎版本、PD 分离架构、量化方案等）后，除传统性能/精度回归外，还需验证：

- **协议一致性**：OpenAI 兼容 API 响应结构（`finish_reason`、`role`、`tool_calls`、`reasoning_content` 等）是否符合规范；流式 SSE 增量拼接后与非流式结构是否一致。
- **受限解码**：JSON Schema、正则、工具调用、语法引导（guided_grammar）等约束是否仍然生效。
- **推理内容泄漏**：`enable_thinking=false` 场景下，思考标签（`<think>`、`<|begin_of_thought|>` 等）是否泄漏到正文中。

本功能全部在 xTask 侧实现，**不侵入 ais_bench 源码**，通过动态生成自定义模型配置和数据集配置接入 AISBench 运行时。

## 2. 整体架构

五层改造，自上而下：

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 脚本生成扩展 | `generate_aisbench_script.py` Step 6 | 受全局开关控制，生成协议测试 shell 段 |
| 协议感知 API 适配器 | `protocol_api.py` | 拦截响应、注入 per-case 参数、流式重组、泄漏检测 |
| 专用数据集 | `configs/datasets/protocol_constrained_gen.py` | 保留 dict 字段、剥离 ground_truth、显式转发 extra_params |
| 三层门禁评估器 | `protocol_constrained_evaluator.py` | 协议 → 语法 → 语义逐级校验 |
| A/B 退化对比 | `protocol_postprocess.py` | 基线 vs 待测通过率对比 + CI 门禁判定 |

数据流：

```
YAML (metrics: protocol + benchmark.protocol)
  → generate_aisbench_script Step 6（检查全局开关）
    → 生成 xtask_protocol_{api_type}.py 模型配置
    → 生成 protocol_constrained_gen 数据集配置（路径覆写）
    → AISBench 运行
      模型侧: create_protocol_aware_chat() 拦截响应，序列化完整 message
      数据集侧: extra_params 逐条注入请求体，ground_truth 剥离送评估器
      评估器侧: 反序列化预测，三层门禁打分
    → 结果 JSON
  → ProtocolABPostProcessor: A/B 对比 + CI 门禁
```

## 3. 全局开关

`config/xTask_config.json` 中 `ENV_CONFIG.enable_protocol_test`，默认 `false`：

```json
{
  "ENV_CONFIG": {
    "enable_protocol_test": false
  }
}
```

读取入口：`xtask/utils/config_parser.py` 的 `is_protocol_test_enabled()`。

行为：

- 开关关闭时，YAML 中即使配置了 `metrics: protocol` 和 `benchmark.protocol` 段，生成脚本只输出跳过注释和警告 echo，不生成任何协议测试命令。
- 开关关闭时，A/B 对比后处理也不执行。
- 开关全局唯一，避免不同环境行为不一致。

## 4. per-case 机制

每条 JSONL 样本独立控制请求参数和期望结果：

```json
{
  "id": "reasoning_002",
  "question": "什么是质数？请简要回答，不要思考过程。",
  "max_out_len": 512,
  "extra_params": {
    "extra_body": {
      "chat_template_kwargs": { "enable_thinking": false }
    }
  },
  "ground_truth": {
    "expected_finish_reason": "stop",
    "forbid_reasoning": true,
    "answer_regex": "质数|大于.*自然数|只能被.*1.*整除"
  }
}
```

- **`extra_params`**：支持 `tools`、`tool_choice`、`response_format`、`guided_json`、`guided_regex`、`guided_grammar`、`extra_body`。由 `ProtocolConstrainedInferencer` 显式转发到模型 `generate()` 调用，再由 `inject_protocol_fields()` 合并进请求体。
- **`ground_truth`**：经数据集 `output_column` 剥离，只送评估器，**不发送给模型**（`ProtocolConstrainedInferencer.infer()` 中 `data.pop("ground_truth")`）。

## 5. 协议感知 API 适配器

文件：`xtask/benchmarks/aisbench/protocol_api.py`

### 5.1 核心函数

| 函数 | 作用 |
| --- | --- |
| `inject_protocol_fields(payload, extra_params)` | 将 extra_params 合并进请求 payload |
| `extract_full_message(response_json)` | 从非流式响应提取完整 message 结构 |
| `assemble_sse_deltas(delta_chunks)` | 将流式 SSE delta 列表重组为完整 message |
| `check_reasoning_leak(content)` | 检测思考标签泄漏（`REASONING_LEAK_TAGS`） |
| `verify_stream_consistency(non_stream_msg, stream_msg)` | 校验流式/非流式结构一致性 |

### 5.2 ProtocolAwareMixin

混入 AISBench API 模型基类，重写 `generate()`：

1. 调用 `inject_protocol_fields()` 注入 per-case 参数。
2. 获取完整响应后，调用 `extract_full_message()` 提取 message 结构。
3. 将 message 序列化为 JSON 字符串作为预测输出（评估器端反序列化还原）。
4. 若 `stream_verify=True`，同时发起流式请求，调用 `assemble_sse_deltas()` 重组后与非流式结果做 `verify_stream_consistency()` 对比。

### 5.3 create_protocol_aware_chat 工厂

```python
def create_protocol_aware_chat(stream_verify: bool = False):
    """动态创建 ProtocolAwareMixin + 原生 API 类的组合类。"""
```

通过 `_get_base_classes()` 从 ais_bench 动态导入基类，运行时拼装，无需修改 ais_bench 源码。生成的模型配置文件 `xtask_protocol_{api_type}.py` 中 `type=create_protocol_aware_chat(stream_verify=...)`。

## 6. 专用数据集

文件：`xtask/benchmarks/aisbench/configs/datasets/protocol_constrained_gen.py`

### 6.1 ProtocolConstrainedDataset

继承 `CustomDataset`，重写 `load()`：逐行解析 JSONL，确保 `extra_params` 和 `ground_truth` 保持 dict 类型不被基类字符串化。

### 6.2 ProtocolConstrainedInferencer

继承 `GenInferencer`，重写 `infer()`：

- 深拷贝数据后 `pop("ground_truth")`，确保 gold 不发模型。
- 显式转发 `extra_params` 到模型调用。

### 6.3 reader/infer/eval 配置

```python
reader_cfg = dict(
    input_columns=['question', 'max_out_len', 'extra_params'],
    output_column='ground_truth',   # gold 只走评估器
)
eval_cfg = dict(
    evaluator=dict(type=ProtocolAndConstrainedEvaluator),
    pred_role='BOT',
)
```

### 6.4 内置数据集

`xtask/benchmarks/aisbench/datasets/protocol_constrained/` 下三组：

| 文件 | 用途 | 条数 |
| --- | --- | --- |
| `reasoning_split.jsonl` | 推理开关/泄漏检测 | 8 |
| `tool_call_bfcl.jsonl` | 工具调用协议 | 7 |
| `constrained_decoding.jsonl` | JSON Schema/正则/语法引导 | 9 |

## 7. 三层门禁评估器

文件：`xtask/benchmarks/aisbench/protocol_constrained_evaluator.py`

预测输出为 JSON 字符串（适配器序列化的完整 message），评估器先反序列化还原为 dict，再逐层校验：

### 7.1 第一层：协议校验 `check_protocol()`

校验响应结构本身是否合规：

- `finish_reason` 是否等于 `expected_finish_reason`（stop/length/tool_calls）。
- `role` 字段是否为 `assistant`。
- 若 `forbid_reasoning=true`，检查 `content` 中是否泄漏思考标签（`check_reasoning_leak()`）。
- 若 `require_reasoning=true`，检查是否存在 `reasoning_content` 字段或思考标签。

### 7.2 第二层：语法校验 `check_syntax()`

根据 `ground_truth` 中的约束类型分派：

| 约束键 | 校验方式 |
| --- | --- |
| `schema` | `_validate_json_schema()`：jsonschema 校验（需安装 jsonschema，未安装时跳过） |
| `regex` | `_validate_regex()`：re.fullmatch / re.search |
| `tool_calls` | `_validate_tool_calls()`：函数名、参数 JSON、调用顺序匹配 |
| `grammar` | 语法引导结果按正则近似校验 |

### 7.3 第三层：语义校验 `check_semantic()`

- `answer`：精确数值/字符串匹配。
- `answer_regex`：正则匹配答案内容。

### 7.4 汇总 `score_predictions()`

对每条样本输出三层各自通过/失败明细，汇总为：

```json
{
  "protocol_pass_rate": 1.0,
  "syntax_pass_rate": 0.857,
  "semantic_pass_rate": 0.714,
  "overall_pass_rate": 0.714,
  "details": [...]
}
```

`ProtocolAndConstrainedEvaluator` 作为 AISBench 评估器接入 `eval_cfg`。

## 8. A/B 退化对比

文件：`xtask/benchmarks/aisbench/protocol_postprocess.py`

用于对比基线版本与待测版本的协议测试结果，判定是否允许上线：

### 8.1 compare_ab_results()

对比两组结果的各层通过率，计算退化幅度。

### 8.2 evaluate_ci_gate() 门禁规则

| 层 | 规则 |
| --- | --- |
| 协议层 | **零容忍**：任何退化即 FAIL |
| 语法层 | **零容忍**：任何退化即 FAIL |
| 语义层 | 退化 ≤ 1.5% 通过，否则 FAIL |

### 8.3 generate_ab_report()

生成 Markdown 对比报告，含各层通过率、退化幅度、门禁结论。

### 8.4 ProtocolABPostProcessor

作为 xTask 后处理入口，读取基线和待测结果目录，执行对比并输出报告。

## 9. 脚本生成扩展

文件：`xtask/benchmarks/aisbench/generate_aisbench_script.py`

在 `generate_aisbench_script()` 中新增 Step 6，触发条件：`metrics` 含 `protocol` 且 `benchmark.protocol` 段存在。

### 9.1 模型配置生成

`create_protocol_model_config()` 基于 `api_type` 对应的原生模型配置，通过 AST 转换生成 `xtask_protocol_{api_type}.py`：

- `_ProtocolModelTypeOverride`：将 `type=原类名` 替换为 `type=create_protocol_aware_chat(stream_verify=...)`。
- `_add_protocol_import()`：插入 `from xtask.benchmarks.aisbench.protocol_api import create_protocol_aware_chat`。

生成后通过 sed 覆写 `host_ip`、`host_port`、`model`、`path`、`max_out_len`、`batch_size`、`request_rate`、`temperature` 及 `generation_kwargs`。

### 9.2 数据集配置覆写

复用 `create_dataset_config_override()` 将 `protocol_constrained_gen` 的数据集路径覆写为用户指定的 JSONL。

### 9.3 执行命令

```bash
python -m ais_bench --models xtask_protocol_vllm_api_general_chat \
  --datasets protocol_constrained_gen \
  --work-dir {output}/protocol_results
```

结果落盘至 `{output}/protocol_results`，模型配置副本一并拷贝。

## 10. 使用指南

### 10.1 开启全局开关

```json
// config/xTask_config.json
{ "ENV_CONFIG": { "enable_protocol_test": true } }
```

### 10.2 YAML 配置

参考示例：`tests/e2e/xllm/examples/yaml_authoring/36_single_aisbench_protocol.yaml`

```yaml
metrics: protocol
benchmark:
  benchmark_type: aisbench
  protocol:
    api_type: vllm_api_general_chat
    dataset_conf: protocol_constrained_gen
    dataset_path: xtask/benchmarks/aisbench/datasets/protocol_constrained/constrained_decoding.jsonl
    stream_verify: false
    batch_size: 8
    output_len: 4096
    num_prompts: 9
    generation_kwargs:
      temperature: 0
      ignore_eos: false
```

`benchmark.protocol` 配置项：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `api_type` | `vllm_api_general_chat` | 原生 AISBench API 模型类型 |
| `dataset_conf` | `protocol_constrained_gen` | 数据集配置名 |
| `dataset_path` | - | JSONL 数据集路径 |
| `stream_verify` | `false` | 是否额外发起流式请求做一致性校验 |
| `batch_size` | `8` | 并发数 |
| `output_len` | `4096` | 最大输出 token |
| `request_rate` | `0` | 请求速率（0=不限） |
| `num_prompts` | - | 采样条数（空=全量） |
| `generation_kwargs` | - | 透传给模型的生成参数 |

### 10.3 自定义数据集

按第 4 节格式编写 JSONL，每行包含 `question`、`max_out_len`、`extra_params`、`ground_truth`，通过 `dataset_path` 指定即可。

## 11. 文件清单

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `config/xTask_config.json` | 修改 | 新增 `ENV_CONFIG.enable_protocol_test` |
| `xtask/utils/config_parser.py` | 修改 | 新增 `is_protocol_test_enabled()` |
| `xtask/benchmarks/aisbench/generate_aisbench_script.py` | 修改 | Step 6 协议测试段 + AST 转换器 |
| `xtask/benchmarks/aisbench/protocol_api.py` | 新增 | 协议感知适配器 |
| `xtask/benchmarks/aisbench/protocol_constrained_evaluator.py` | 新增 | 三层门禁评估器 |
| `xtask/benchmarks/aisbench/protocol_postprocess.py` | 新增 | A/B 对比后处理 |
| `xtask/benchmarks/aisbench/configs/datasets/protocol_constrained_gen.py` | 新增 | 数据集配置 |
| `xtask/benchmarks/aisbench/datasets/protocol_constrained/*.jsonl` | 新增 | 内置数据集（3 组 24 条） |
| `tests/e2e/xllm/examples/yaml_authoring/36_single_aisbench_protocol.yaml` | 新增 | YAML 示例 |
| `tests/test_aisbench_protocol.py` | 新增 | 48 个单元测试 |