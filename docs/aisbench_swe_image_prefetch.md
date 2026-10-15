# SWE-bench x86 镜像预拉取（不占用 NPU）

仅当 x86 无法稳定连接 Docker Hub、SWE-bench 实例镜像尚未缓存时使用。
AISBench 会在启动 Agent 前检查实例镜像，缺失时尝试 `docker pull`；直接启动 xTask 用例可能使 xLLM/NPU
在网络下载期间空等。此工具只在指定的 x86 Docker 主机上添加所需镜像和 tag，
**不重启或修改共享 Docker daemon，不清理其他镜像/容器**。

## 推荐：主执行机一条命令准备 x86 镜像

在 xTask 工程根目录运行（不要启动 NPU/xLLM）：

```bash
python3 -m tools.prepare_aisbench_swe_images \
  --case-yaml tests/e2e/xllm/examples/yaml_authoring/35_single_aisbench_swebench_dsv4_verified_mini10.yaml
```

入口在主执行机读取用例 YAML、实际筛选的本地 parquet，生成镜像清单；使用已有的
xTask SSH 凭据将清单和三个辅助脚本传到 YAML 的 `runner.host`，然后在 x86 上依次
执行 **prefill（内部调用 resume）→ prefetch**。不需要手动指定或传送 `--images-file`，
也不会默认查找 `outputs/33.../diagnostics`。缓存保留在 x86 用户目录下，使用 `flock`
防止两个入口同时预拉取；不会在用例执行时隐式预拉取。

要求 x86 用户目录已有**核验过来源**、可执行的
`$HOME/xtask_swe/tools/crane-v0.22.1`，x86 主机有 `python3`、`requests` 和 `flock`。
如需其他路径，用 `--crane /x86/绝对路径`；本地 YAML 指向 x86 独有的数据集路径时，
用 `--dataset-path /主机可读的parquet或目录`。当前续传脚本固定使用
`docker.1ms.run` 镜像源和 `linux/amd64` 实例镜像，其他源需要先适配脚本。
**准备成功不代表评测通过**：准备完成后再单独运行 xTask 用例。

## 手动分步模式（需要排障时）

## 数据和前置条件

`verified_mini` 本地 parquet 的 `test` split 包含 50 个 `instance_id`。
根据 mini-swe-agent 的 `get_swebench_docker_image_name()`，
`django__django-11790` 对应的镜像为
`swebench/sweb.eval.x86_64.django_1776_django-11790:latest`；
其他实例同样将 `__` 替换成 `_1776_` 并转小写。

在主执行机用 **用例 YAML 和本地数据集**生成镜像列表，再将列表和以下三个预拉取工具复制到 x86 的**用户目录**：

- `tools/prefill_aisbench_swe_layers.py`
- `tools/resume_aisbench_swe_blob.py`
- `tools/prefetch_aisbench_swe_images.py`

```bash
# 在 xTask 项目根目录运行：自动读取 35 号 YAML 的 split、filter_spec，
# 并从本地默认 verified_mini parquet 生成实际选中的 10 个镜像。
python3 -m tools.generate_aisbench_swe_images \
  --case-yaml tests/e2e/xllm/examples/yaml_authoring/35_single_aisbench_swebench_dsv4_verified_mini10.yaml \
  --output "$HOME/xtask_swe/verified_mini10_images.txt"
```

生成器**不**默认读取 `outputs/33.../diagnostics`，也不修改三个预拉取脚本的原有 `--images-file` 用法。
换成 33 号 YAML 会生成完整 50 个镜像；数据集来源不是主执行机可读的默认路径时，
加 `--dataset-path /本机可读的parquet或目录`。生成器只读本地数据，不访问网络或 Docker；
必须将其产出的列表复制到 x86，然后把下方命令中的 `--images-file` 指向复制后的列表。
若找不到本地数据或筛选结果为空，会直接报错，不会使用历史列表。

在 x86 用户目录准备可信的 `crane` 命令。实际诊断使用的是
`google/go-containerregistry` 官方 v0.22.1 的 Linux x86_64 发布包，
从 GitHub 发布 API 获取资产 SHA-256，校验后才运行；
**不要从未经核验的地址下载并执行二进制文件**。
诊断时的可用镜像源是 x86 daemon 已配置的 `docker.1ms.run`；
环境变化时应换用受信任且可达的源。

## x86 侧执行顺序

下面路径仅为示例，先确保列表和脚本已复制到 `$HOME/xtask_swe/swe_image_prefetch/`：

```bash
ROOT="$HOME/xtask_swe/swe_image_prefetch"
CRANE="$HOME/xtask_swe/tools/crane-v0.22.1"
export HTTPS_PROXY=http://bamboo-proxy.jd.com:80
export HTTP_PROXY="$HTTPS_PROXY"
export NO_PROXY="docker.1ms.run,localhost,127.0.0.1,11.87.191.75"

# 针对长连接中断，按 HTTP Range 续传缺失层；只有大小和 SHA-256
# 与镜像清单一致的完整层才会进入此用户目录下的压缩层缓存。
python3 "$ROOT/prefill_aisbench_swe_layers.py" \
  --images-file "$ROOT/verified_mini_images.txt" \
  --cache "$ROOT/cache" \
  --blob-helper "$ROOT/resume_aisbench_swe_blob.py"

# 使用已缓存的压缩层生成 OCI/Docker 镜像、导入并标记为 AISBench 使用的名称。
python3 "$ROOT/prefetch_aisbench_swe_images.py" \
  --images-file "$ROOT/verified_mini_images.txt" \
  --crane "$CRANE" --root "$ROOT" \
  --docker-root "$(docker info --format '{{.DockerRootDir}}')" \
  --min-free-gb 100 --timeout 360 --retries 3
```

三个工具仅作用于 SWE-bench 预拉取操作，不修改 xTask 普通 AISBench、EvalScope、VBench 等流程。
第二步会跳过已有的规范镜像 tag，使用 `$ROOT/staging` 的临时 tar 并在单镜像导入后清理；
`$ROOT/cache` 是用户专用缓存。默认在共享 Docker 存储空间不足 100 GiB 时停止，
**不会自动 prune**。确认所需镜像已全部缓存后，再执行完整 xTask 用例。
