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
| `34_glm5.3_pd_split_service_reuse_containers_perf.yaml` | GLM5.3 2048/1024、bs1、1P1D，显式 Coordinator/Router 复用对应引擎节点容器 |

## PD 的 Coordinator / Router 配置

`tests/e2e/` 下的 PD 用例统一使用 `deployment.xllm_service` 中的显式
`coordinators`、`routers` 列表，写法参见 `34_glm5.3_pd_split_service_reuse_containers_perf.yaml`：

- Coordinator：HTTP/RPC 为 `28888/28889`，通常复用 Prefill 主节点容器。
- Router：HTTP/RPC 为 `38888/38889`，通常复用 Decode 主节点容器。
- `host` 与 `container_instance` 必须匹配本用例的 Engine node；后者是 YAML 中的 node 名称，不是逻辑 P/D 组名或 Docker 容器名。Service 不新增容器，也不计入 NPU/rank 拓扑。
- `deployment.benchmark.router_instance: router-0` 指定请求入口，健康检查路径为 `/readyz`；etcd 的地址及端口按各用例原配置保留，Engine 与两个 Service 角色共享该 etcd。
- 同机 PD 用例的 Router 仍绑定本机实际 Decode node。`14` 号示例的 Decode 节点会随拓扑候选变化，故 Router 固定复用 `75/prefill-0-node-0`，确保所有候选均可规划。

这些端口是单个用例的默认配置，不代表可以在同一服务器上并行启动多个用例；并行运行前应分别调整角色和 etcd 端口，并检查 NPU 占用。显式 C/R 列表面向新版 xllm-service，需要配套使用支持两个角色的二进制。
