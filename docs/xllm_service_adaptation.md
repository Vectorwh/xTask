# 新版 xLLM-Service / PD 运行适配

本次修改保留现有 YAML 中 Coordinator/Router 节点、端口、Engine 设备和 rank 分配。没有更新 xLLM/Service 远端源码、启动实际服务或执行编译。

## 1. Service 构建与静态 tokenizer

```bash
# 默认仍编译测试目标，不自动执行 CTest
bash tools/prepare_xllm_service.sh

# 快速构建：只在上游 setup.py 支持时传 --no-testing
XLLM_SERVICE_BUILD_TESTING=0 bash tools/prepare_xllm_service.sh
```

- 新 CMake 独立查找 spdlog：只更新直接子模块，并检查协议文件。
- 旧 CMake 依赖 xLLM 嵌套 spdlog：保留原递归更新/修复逻辑。
- 用 `readelf -d` 检查 Service 的实际 `DT_NEEDED`。静态 tokenizer 不要求 `.so`；动态 tokenizer 必须有对应 `.so`。
- 归档包含 tokenizer linkage、源码提交、构建选项、SHA256SUMS，完整产物落地后原子切换 `service_binary/current`。
- 安装时 current 或已提供的直接归档损坏则明确失败，不静默回退旧 Service。

## 2. 新 release pack

原 `pkg/xllm/{wheel,vendors,third_whl,runtime_libs,service_binary}` 流程默认保留。新版 upstream 产生 pack 时，xTask 同时保存到 `pkg/xllm/release_pack/<sha256>/`，原子更新 `current`。

```yaml
deployment:
  xllm_package:
    # 容器路径：整个 pkg 会按原有机制同步到相关节点
    release_pack: /workspace/pkg/xllm/release_pack/current/xllm_pack.tar.gz
    # 可选；未填写时要求镜像已经有 etcd 和 etcdctl
    etcd_bin_dir: /workspace/pkg/xllm/etcd_bin
```

也支持 `XLLM_RELEASE_PACK`、`XLLM_ETCD_BIN_DIR` 环境变量，显式 YAML 优先。没有独立 Engine wheel 且有本地 release_pack/current 时自动选新包；有原 wheel 时不强行切换。

新 pack 完整安装 wheel、vendors、三方 wheel、Mooncake runtime 和可选 Service，不混入旧 pkg 的 wheel 或运行库。新包不含 etcd，因此控制容器必须有独立 etcd 来源，缺失时不下载旧包兜底。普通 Engine 容器不要求 Service/etcd。

安装前验证 pack 布局、已有 SHA256 sidecar、Mooncake 内部校验和、runtime 与 wheel 的 SHA256 配对、CPU/Python ABI、可解析的 Torch/CANN 版本。源码/版本记录保存至容器 `/opt/xllm/installed-pack.json`。pip 最终判断 wheel 安装兼容性。没有元数据的 ABI/协议兼容性不能仅凭文件名完全证明，必须真实部署验证。

新旧包不会只按 `xllm_pack.tar.gz` 同名当成同一种格式。旧安装路径遇到新布局会明确提示选择新包，避免混装。

### 合并 Service（可选，不修改上游原始 pack）

```bash
python tools/xllm_release_pack.py publish \
  /path/to/xllm_pack.tar.gz pkg/xllm/release_pack \
  --service pkg/xllm/service_binary/current
```

构建 Service 时也可以设置 `XLLM_SERVICE_APPEND_PACK=/path/to/xllm_pack.tar.gz`，构建成功后发布合并包。

`service_build` 默认会自动把新 Service 合入当前 xLLM release pack：

```bash
XLLM_SERVICE_AUTO_APPEND_PACK=1   # 默认
```

选择输入 pack 的优先级为：

```text
XLLM_SERVICE_APPEND_PACK
  > pkg/xllm/release_pack/current/xllm_pack.tar.gz（auto append 开启且存在时）
  > 不合并，仅保留独立 Service 产物
```

关闭自动合并：

```bash
XLLM_SERVICE_AUTO_APPEND_PACK=0
```

该操作只更新 release pack 发布目录；默认用例仍走独立 `pkg/xllm/{wheel,vendors,third_whl,runtime_libs,service_binary}` 流程，只有显式配置 `deployment.xllm_package.release_pack` 或 `XLLM_RELEASE_PACK` 时才用 pack 安装。合并时重新生成整包校验值，临时目录验证成功后才原子切换 current，旧包仍保留。pack 仍不携带 etcd；运行控制面容器要求镜像预装 etcd，或显式配置 `XLLM_ETCD_BIN_DIR`。上游自行 append 后如果原 `.sha256` 已过期，应先按可信构建产物重新生成正确校验值；xTask 不会忽略错误 checksum。

## 3. HCCN 配置副本

```yaml
deployment:
  mode: pd_disaggregated
  hccn:
    mode: auto       # 默认；也可 strict / disabled
    source: /etc/hccn.conf
```

| 模式 | 行为 |
| --- | --- |
| auto（默认） | 对本次各张物理 NPU 只读查询；配置完整才挂载副本。无法查全则提示并保留工具 fallback，不挂 TLS-only 文件。 |
| strict | 所有本次设备地址必须可解析；地址缺失则在 Engine 启动前失败。 |
| disabled | 保持原运行容器配置，不做查询/挂载。 |

- 在每个真实 Engine 主机执行 `hccn_tool -i N -ip -g`，从不执行设备设置操作。
- 保留 TLS 等原始配置，补齐真实设备地址；已有地址与查询冲突则失败。
- 副本权限 0600，只读挂载到容器 `/etc/hccn.conf`。副本保存在 Engine 输出目录的同级 `.hccn/`，避免远端启动前同步脚本时被删。
- 不修改、清空、覆盖宿主配置。自动模式中不允许保留已有的 `/etc/hccn.conf` 直接 host bind，应改用 `source`；完全维持已有自定义挂载可显式 disabled。
- 构建 xLLM 默认检查宿主可见物理卡（`XLLM_BUILD_DEVICES=all`），也可显式列表如 `0,4`；`XLLM_HCCN_MODE=auto/strict/disabled` 控制构建模式。只生成临时副本，退出后删除。
- Service 独立 CPU 构建不再挂载宿主 HCCN。
- 两种 Engine 模板都从相同物理 ID 列表生成 ASCEND_RT_VISIBLE_DEVICES / ASCEND_VISIBLE_DEVICES，PD 显式启用 service routing。

## 4. 内部 token

优先级：`deployment.xllm_service.internal_api_token` → `XLLM_INTERNAL_TOKEN` → 每次运行随机 token。

- 一次运行共享同一个随机值，重试/评估复用不重新随机；新运行更换随机值。
- 随机值只存进程内缓存和运行对象，不写回 YAML/task_config；显式用户配置保持不变。
- Coordinator/Router 及 legacy Service 获得相同运行凭据；不假设所有版本的 Engine 都需要传该参数。
- 命令日志（包括 Base64 包装）、异常日志、结果报告和可选 bench-reporter 输出脱敏，不将 token 当普通 tokens 性能指标处理。
- 上游目前通过 CLI 接收 token，仍可能在有权限的 ps/proc 读取中可见。脱敏不等于网络加密或操作系统级秘密保护。
- 这是内部接口鉴权，不是 benchmark 公共推理 API Key，也不是 etcd 鉴权。

## 5. 停机与进程所有权

新版 split 流程：benchmark 客户端返回/停止发送 → 可选 drain 等待 → Router/Coordinator → Engine → 本次 owned etcd → 原有资源回收和容器清理。

```yaml
deployment:
  shutdown_grace_seconds: 15         # Engine 退出等待（原参数）
  container_stop_timeout_seconds: 15 # 容器退出等待（原参数）
  xllm_service:
    shutdown_grace_seconds: 15       # 各 Service / etcd TERM 等待，范围 1..300
    shutdown_drain_seconds: 0        # 可选额外排空宽限，范围 0..300
```

- drain_seconds 是有限宽限等待，不是虚构的服务器 drain API，不能宣称已准确统计所有外部在途请求。xTask 独占 benchmark 正常返回时不再发送请求；共享对外服务应另加实际 admission/drain 能力。
- 保留 legacy 单进程的原先 Engine-first 顺序，但将 legacy Service / etcd 的停止改为本次 owned PID+starttime+pidfd，不再全局按进程名 pkill。
- 启动部分成功、启动命令异常、Service 停止失败等情况下仍继续其他 owned 成员清理；Engine 等待抛异常也尝试停止 owned etcd。
- etcd 每运行独立数据目录，重试不直接 rm -rf 原目录，数据和日志保留。
- CLEAN_RESULTS=false 不执行排空/停进程/清容器，保留完整环境。
- 本阶段没有新增共享 external etcd 接管/HA，已有端口占用仍按原逻辑拒绝。

## 6. 验证范围

详见 `outputs/service_adaptation_20261114_11.87.191.75.md`。测试使用临时归档、假安装器或真实 CPU 子进程，不运行模型、不占 NPU。真实 HCCN 只执行读查询。

完整 Service 编译、跨 75/121 容器运行及真实 PD 请求/吞吐仍需后续验证；本次通过单测不等于这些实机验证已经完成。
