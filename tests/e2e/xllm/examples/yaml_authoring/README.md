# YAML 写作示例归档

这里是给新用例作者看的独立示例，不属于现有 case group，也不替换 `tests/e2e/xllm` 下的旧用例。

示例默认使用当前测试环境的模型路径和服务器：

- 单机示例：`11.87.191.75`
- 普通多机：`11.87.191.75`、`11.87.191.121`
- PD 拓扑：Prefill 使用 `75`、`121`，Decode 使用 `82`、`83`

运行前至少检查 `model_path`、各节点 IP、`device_ids`、端口和 SSH 配置。新用例建议复制最接近的示例后改名，不要直接修改历史用例。

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
