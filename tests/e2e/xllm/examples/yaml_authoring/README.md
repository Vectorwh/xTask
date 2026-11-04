# YAML 写作示例归档

这里是给新用例作者看的独立示例，不属于现有 case group，也不替换 `tests/e2e/xllm` 下的旧用例。

示例默认使用当前测试环境的模型路径和服务器：

- 单机示例：`11.87.191.75`
- 普通多机：`11.87.191.75`、`11.87.191.121`
- PD 拓扑：Prefill 使用 `75`、`121`，Decode 使用 `82`、`83`

运行前至少检查 `model_path`、各节点 IP、`device_ids`、端口和 SSH 配置。新用例建议复制最接近的示例后改名，不要直接修改历史用例。

ASR 的准备、YAML 配置、评分/门禁、归档和搜索说明统一维护在
`docs/xllm_yaml_authoring_guide.md` 第 3 节“JoyAI-ASR 原始评测”。
TP2 自部署基础用例位于 `tests/e2e/xllm/joyai/joyai_asr/`；40/41 是同一实现的
单轮/两轮外部服务示例，不存在 YAML 继承关系。

文件对应关系：

| 文件 | 用途 |
| --- | --- |
| `01`-`04` | 单机 AISBench/EvalScope 性能和精度 |
| `05`-`06` | 普通多机混部性能和精度 |
| `07`-`08` | `deployment.pd_instances` 方式的 PD 性能和精度 |
| `09` | 单机 `auto_assess` 并发摸高 |
| `10`-`11` | 单机 exhaustive/TPE 搜参 |
| `12`-`13` | 多机 auto_assess/para_set |
| `14` | 多机 PD 拓扑 + 参数 exhaustive 搜索 |
| `15` | PD 场景复用容器和 xLLM 服务做 auto_assess |
| `16` | PD 场景 exhaustive 参数搜索 |
| `17` | 单机 AISBench ShareGPT 多轮对话性能测试 |
| `18` | 单机 AISBench ShareGPT `last` 模式固定输入长度性能测试 |
| `19` | 普通服务在同一节点拆分为多个小 runtime |
| `20` | 多个 P/D 逻辑实例及同一逻辑实例的多个 node 共节点部署 |
| `21` | 多个普通独立实例共用一组 xllm-service + etcd 做统一请求分发 |
| `22` | 单机 AISBench ShareGPT `last` 模式多组 Prefix Cache 性能测试 |
| `23` | 单机 AISBench ShareGPT `last` 模式 `packed` 多组 Prefix Cache 长上下文性能测试 |
| `24` | 单机 AISBench GPQA Diamond 精度测试 |
| `25` | 单机 xLLM Wan2.2 I2V 视频生成与 VBench 质量评测 |
| `26` | 外部已有 OpenAI 兼容服务 + AISBench 性能（不启动 xLLM） |
| `27` | 外部已有 Wan2.2 + 独立 VBench runner 质量评测（不启动 xLLM） |
| `28` | `startup_environment`：按逻辑实例继承、按 node 整段替换的启动脚本 shell 行 |
| `40` | 外部 JoyaiASR 服务 + 原始客户端：完整 1162 条精度与性能评测（multipart/SSE，并发 2，不管理服务生命周期） |
| `41` | 外部 JoyaiASR 服务 + 两轮完整 1162 条评测：验证日期归档、多轮 CSV/统计和全局/用例组 Excel（并发 2） |
| `42` | Qwen3.6-35B-A3B 普通 LLM perf 回归抽测：复制现有 BS1 用例到本机 `.75`，保持 AISBench 请求、预热、MTP 和基线不变 |
