# 环境资产清单（日构建依赖的外部预置资产）

日构建用例除代码仓库与构建产物外，还依赖三类**外部预置资产**。历史教训
（2026-11-20）：新机器只拉仓库 + 跑构建，三类资产缺失导致 mmm 组 6 个用例
失败，且失败形态隐蔽（gedit 服务起不来 / VBench 质量维度全 NaN / ASR 直接
FileNotFoundError）。

## 资产一：xllm wheel（标准分发，自动）

| 项 | 内容 |
|---|---|
| 位置 | `pkg/xllm/xllm_npu*.whl` |
| 来源 | build 阶段产出（`tools/build_xllm_in_container.sh`） |
| 安装 | 容器内 `tools/download_xllm.sh` 自动 pip 安装 |
| 说明 | **无需任何手工操作**。普通用例与 gedit 用例（改走 wheel 链路后）都从这里取运行时 |

## 资产二：ASR 归档（joyai_asr 用例）

| 项 | 内容 |
|---|---|
| 位置 | `/export/home/models/xllm_test_060/datasets/joyai_asr/`：`data.zip`（111MB，音频数据集）；`mmm_bench/joyai_asr/`：`infer.zip`（740KB，WER 打分工具） |
| 来源 | 2026-11-03 打包，已由 jeby 机 asr_doc/ 迁入共享区 |
| 安装 | `generate_asr_script.py` 自动拷贝到用例输出目录；找不到时报 FileNotFoundError 并列出候选路径 |
| 兼容 | 旧工作区 `asr_doc/` 路径仍作 fallback（未配共享区的环境）；yaml 可用 `infer_archive`/`dataset_archive` 覆盖 |
| 失效症状 | `ASR infer_archive not found`，用例 0 秒失败 |

## 资产三：VBench 视频评测资产（wan2.2 I2V + joyai_video 两族共用）

| 项 | 内容 |
|---|---|
| 位置 | `/export/home/models/xllm_test_060/mmm_bench/Video/vbench2_beta_i2v/`（**唯一实体层**，目录名是稳定接口，勿重命名）：`full_info.json`（1118 条 I2V 提示词）+ `vbench2_i2v_mini_full_info.json`（107 条 mini）+ `data/`（2.5GB 参考图片）+ `clip/` + `pretrained_models/`（dino/clip/amt/raft/aesthetic/pyiqa/dreamsim/cotracker 全部评测权重） |
| 来源 | vbench2 官方资产（2026-11-20 由 `benchmark/vbench2/` 迁入，按模态归位 Video/） |
| 安装 | **零安装**。数据集路径由 yaml/`defaults.py` 直指实体层；评测权重的扁平 cache（`VBENCH_CACHE_DIR` = 实体层内 `vbench_cache/`）由 `ensure_cache_layout()` 在两族用例评测前自动重建 symlink 视图（幂等） |
| 共用关系 | wan2.2 用 full_info/data + dreamsim/cotracker + 6 个质量权重；joyai_video 用 6 个质量权重；被测模型（wan2_2 / joyai_video）各自独立 |
| 失效症状 | preflight 报 "VBench canonical asset tree" 缺失；缺权重时 t2av/i2v 评测维度 fail，实体被移动时视图自动重建仍可自愈 |

**教训记录**：历史上共享区曾用 `xllm_test_060/vbench_data` symlink 指向带日期快照
的目录，11/19 重命名共享区时 symlink 断链，两台机全部 vbench 用例同时挂。收敛
原则：实体层唯一化+稳定名、代码直指实体路径、视图可随时重建，**禁用指向快照
目录的绝对路径 symlink**。

## 资产四：JoyAI 负向校准环境（joyai_video_negative_calibrated 用例）

| 项 | 内容 |
|---|---|
| 位置 | `/export/home/models/xllm_test_060/mmm_bench/joyai_video_benchmark/`：`semantic_env/`（1.4G 独立 venv，python 3.11）、`grounding-dino-tiny/`（视觉违禁检测权重）、`clap-htsat-unfused/`（音频-文本对齐权重）、`wheelhouse/`（venv 离线轮仓）、`calibration/`（校准数据） |
| 安装 | 零安装。yaml 直接以 `python_exec: .../semantic_env/bin/python` 调用（venv 的路径解析相对自身，迁移目录后仍可用；`activate`/`pip` 脚本内的旧 shebang 不影响运行，用例不走它们） |
| 失效症状 | preflight 报 "JoyAI negative-calibration env" 缺失；用例负向检查直接失败 |

## gedit（image edit）二进制——已消灭的手工依赖

历史形态：gedit 用例直接执行工作区 `tmp/gedit_svc/xllm` 二进制（2026-10-28
在 jeby 机手工放置，之后无人维护版本）。新机器缺该文件时 encoder/dit 服务
`No such file or directory`，健康检查超时。

现状（改走 wheel 链路）：
1. scheduler 为每个 gedit 服务容器执行 `download_xllm.sh`（与普通用例的
   runtime deps 步骤一致），从 `pkg/xllm/` 安装最新构建的 wheel
2. 启动脚本运行时解析已安装 wheel 的二进制
   （`site-packages/xllm/xllm`），`tmp/gedit_svc/xllm` 降级为 fallback
3. 两者都没有时 fail fast，报错指明解决方式
4. yaml 仍可用 `xllm_binary` 钉死固定路径（自定义构建场景的逃生口）

收益：gedit 二进制始终跟随日构建最新版本（旧形态用 10 月二进制跑 11 月用例），
新机器零手工操作。

## 日构建体检

`tools/preflight_assets.py` 在 test 阶段开始时自动运行（只报告不阻断）：
wheel / ASR 归档 / VBench 模型三组检查，缺失时打印受影响用例清单。
手工执行：`python3 tools/preflight_assets.py`（加 `--strict` 缺失即非零退出）。

## 新环境初始化 checklist

1. 拉取 xTask 仓库
2. 确认 `/export/home/models` 挂载可见（依赖包根目录默认
   `/export/home/models/xllm_test_060`，可通过 config 的
   `DEPENDENCE_PACKAGE_PATH` 覆盖）
3. 跑一轮 build（自动产出 wheel）
4. 运行 `python3 tools/preflight_assets.py` 确认资产齐备
