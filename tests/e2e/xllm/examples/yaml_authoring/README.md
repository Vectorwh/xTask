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
| `34_glm5.3_pd_split_service_reuse_containers_perf.yaml` | GLM5.3 2048/1024、bs1、1P1D，显式 Coordinator/Router 复用对应引擎节点容器 |
| `40` | 外部 JoyaiASR 服务 + 原始客户端：完整 1162 条精度与性能评测（multipart/SSE，并发 2，不管理服务生命周期） |
| `41` | 外部 JoyaiASR 服务 + 两轮完整 1162 条评测：验证日期归档、多轮 CSV/统计和全局/用例组 Excel（并发 2） |
| `43` | 外部已有服务 + AISBench GPQA 精度，endpoint 用**域名**（验证 external_service 走 AISBench `url` 字段） |
| `42` | Qwen3.6-35B-A3B 普通 LLM perf 回归抽测：复制现有 BS1 用例到本机 `.75`，保持 AISBench 请求、预热、MTP 和基线不变 |

## PD 的 Coordinator / Router 配置

`tests/e2e/` 下的 PD 用例统一使用 `deployment.xllm_service` 中的显式
`coordinators`、`routers` 列表，写法参见 `34_glm5.3_pd_split_service_reuse_containers_perf.yaml`：

- Coordinator：HTTP/RPC 为 `28888/28889`，通常复用 Prefill 主节点容器。
- Router：HTTP/RPC 为 `38888/38889`，通常复用 Decode 主节点容器。
- `host` 与 `container_instance` 必须匹配本用例的 Engine node；后者是 YAML 中的 node 名称，不是逻辑 P/D 组名或 Docker 容器名。Service 不新增容器，也不计入 NPU/rank 拓扑。
- `deployment.benchmark.router_instance: router-0` 指定请求入口，健康检查路径为 `/readyz`；etcd 的地址及端口按各用例原配置保留，Engine 与两个 Service 角色共享该 etcd。
- 同机 PD 用例的 Router 仍绑定本机实际 Decode node。`14` 号示例的 Decode 节点会随拓扑候选变化，故 Router 固定复用 `75/prefill-0-node-0`，确保所有候选均可规划。

这些端口是单个用例的默认配置，不代表可以在同一服务器上并行启动多个用例；并行运行前应分别调整角色和 etcd 端口，并检查 NPU 占用。显式 C/R 列表面向新版 xllm-service，需要配套使用支持两个角色的二进制。

## JoyAI-Video（显式新增路径）

注意：编号 `43` 同时被 `43_external_aisbench_gpqa_acc_domain.yaml`（见上表）和
`43_external_joyai_video_t2v_quality.yaml`（见下表）使用。

| 示例 | 用途 |
| --- | --- |
| `43` | T2V 视频基础质量；要求无音轨的对应部署 |
| `44` | T2AV：VBench 视频质量＋音轨/音画时长完整性；不是 VABench 全量质量 |
| `45` | 正负提示词配对生成；未配置检测器时 `negative_control=not_scored` |
| `46` | 外部 FP32 embedding 生成；需真实特征清单和来源 |
| `47` | 服务内文本／外部 embedding 配对，记录解码画面和音频数值差异 |
| `48` | 独立 T2AV 性能，预热不计入测量，不执行评分器 |
| `50` | 本机16卡自部署文字直发 T2AV 冒烟，1条/2步；不作为质量/性能基线 |
| `51` | 本机自部署 T2AV 质量候选，3条/50步、VBench六维；无正式验收基线 |
| `52` | 本机自部署独立 T2AV 性能候选，3条×2轮/50步、1次预热、无评分器 |
| `53` | 原25的本机 Wan I2V 回归副本，保留原 benchmark/模型参数 |
| `54` | 原42的 Qwen3.6 perf 回归副本，使用空闲2/3卡与隔离端口，保留MTP、请求和基线 |
| `55` | 自部署JoyAI文字直发负向语义专项：12校准标签条件/24配对视频，GroundingDINO/CLAP离线评分，保留fail/inconclusive |

全部使用 `benchmark_type: vbench`、显式 `benchmark.vbench.task: t2v/t2av`。
不写 `task` 时仍走原 Wan I2V；没有模型名自动切换。运行前修改 endpoint、模型 ID、
音频采样率及评分模型路径。43/44 是自定义小集，不是官方标准全量分数；46/47
故意使用待提供的 embedding manifest 路径，不能拿合成特征做真实模型验收。
当前 `auto_assess/para_set` 不开放，使用显式并发/重复用例；完整说明见
`docs/xllm_yaml_authoring_guide.md` 的“JoyAI-Video T2V/T2AV”小节。

### 文本直发＋负向内容评分（49）

`49_external_joyai_video_negative_semantics.yaml` 使用纯文本 T2AV 配对生成，
独立 GroundingDINO（画面物体）／CLAP（音频事件）评分，不需要外部 embedding。
按 `xtask_negative_v1` 暂定回归规则给出 pass/fail/inconclusive；对照没有禁止
内容或样本不足时不判负向有效。12 个明确条件对应 24 个视频，含 6 个画面和
6 个音频条件。必须先准备示例中的离线评分模型目录与真实服务地址，禁止运行时
隐式下载模型。不改原45的请求级诊断、Wan I2V或其他模型门禁。
此规则不是 VBench/VABench 官方阈值，不覆盖动作、物理、口型和音画同步。
