#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VBench-I2V mini 视频测评脚本：10 个维度评测 + 集成总分计算。

严格按官方文档 vbench2_beta_i2v/README.md 的流程实现：
  - 第 5 节 Evaluation：对每个维度调用官方 compute_<dimension> 函数，
    3 个 i2v 维度（i2v_subject / i2v_background / camera_motion）来自
    VBench_stand/datasets/vbench2_beta_i2v（与上游 master 一致），
    7 个质量维度（subject_consistency / background_consistency /
    motion_smoothness / dynamic_degree / aesthetic_quality /
    imaging_quality / temporal_flickering）来自 ais_bench 适配过 NPU 的
    third_party/vbench（与本机 VBench 1.0 评测同一套实现）。
  - 第 6 节 How to Calculate Total Score：
      Normalized Score = (dim_score - min_val) / (max_val - min_val)
      加权平均得到 Quality Score 与 I2V Score，再合成 Total Score。
      公式常数（Min/Max、权重）直接 import 上游 scripts/constant.py，
      计算逻辑与上游 scripts/cal_i2v_final_score.py 完全一致。
      注意：temporal_flickering 会评测并报告，但不参与总分（上游
      TASK_INFO_I2V 总分公式中不含该维度，属官方行为）。

输入：生成脚本 produce 的 *_videos.json（官方 full_info 格式 + video_list
绝对路径，条目缺 video_list 自动跳过，与官方 load 函数行为一致）。

用法示例（先跑 --check 体检，再正式评测）：
    python3 evaluate_vbench_i2v_bench.py \
        --full_info /export/home/weinan5/wsd/xllm-wsd-wan2.2/vbench_i2v_benchmark/vbench2_i2v_mini_full_info_videos.json \
        --resolution 16-9 \
        --name wan2_2_i2v_mini \
        --output_dir ./i2v_eval_results \
        --check

    python3 evaluate_vbench_i2v_bench.py \
        --full_info /export/home/weinan5/wsd/xllm-wsd-wan2.2/vbench_i2v_benchmark/vbench2_i2v_mini_full_info_videos.json \
        --resolution 16-9 \
        --name wan2_2_i2v_mini \
        --output_dir ./i2v_eval_results

只重算总分（读已有 eval results，不重新评测）：
    python3 evaluate_vbench_i2v_bench.py --results_file ./i2v_eval_results/wan2_2_i2v_mini_eval_results.json
"""

import argparse
import gc
import importlib
import json
import math
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# 路径常量（本机环境）
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 支持环境变量覆盖，便于在另一台机器复用（迁移说明见 MIGRATION.md）：
#   VBENCH2_DATASETS -> 官方数据集根目录（其下含 vbench2_beta_i2v 子目录）
#   AIS_THIRD_PARTY  -> ais_bench 适配版 vbench 代码所在 third_party 目录
VBENCH2_DATASETS = os.environ.get('VBENCH2_DATASETS',
                                  '/export/home/weinan5/wsd/VBench_stand/datasets')
VBENCH2_ROOT = os.path.join(VBENCH2_DATASETS, 'vbench2_beta_i2v')
AIS_THIRD_PARTY = os.environ.get('AIS_THIRD_PARTY',
                                 '/export/home/weinan5/wsd/VBench_stand/benchmark/ais_bench/third_party')
# 评测依赖（dreamsim 及其闭包 open_clip/peft 等）已固化进 runner 镜像
# （Dockerfile 见 /tmp/vbench_image_build，镜像 8.1.rc1-a3-*）。
# 历史上的共享目录外挂方案（py_deps）已停用；若旧镜像环境再次缺失依赖，
# 恢复下方两段引用即可：
#   VBENCH_PY_DEPS = os.environ.get(
#       'VBENCH_PY_DEPS', os.path.join(VBENCH2_DATASETS, 'py_deps'))
# 评分模型缓存（VBENCH_CACHE_DIR 由容器环境注入；缺省回退到 ~/.cache/vbench
# 与上游 vbench2_beta_i2v/utils.py 的默认一致）
VBENCH_CACHE_DIR = os.environ.get('VBENCH_CACHE_DIR',
                                  os.path.join(os.path.expanduser('~'), '.cache', 'vbench'))
# 评分模型缓存的共享数据根不再用固定常量：调度器注入的 VBENCH_DATA_HOME 可能
# 指向 vbench2 datasets 根（benchmark/vbench2，与 vbench_data 共用变量名但布局
# 不同），运行时由 _resolve_data_home() 按候选顺序探测含 pretrained_models 的目录。

I2V_DIMS = ['i2v_subject', 'i2v_background', 'camera_motion']
QUALITY_DIMS = [
    'subject_consistency', 'background_consistency', 'motion_smoothness',
    'dynamic_degree', 'aesthetic_quality', 'imaging_quality',
    'temporal_flickering',
]
ALL_DIMS = I2V_DIMS + QUALITY_DIMS


def _resolve_data_home():
    """Locate the shared tree that actually contains pretrained_models/.

    ``VBENCH_DATA_HOME`` may be injected by the scheduler as the vbench2
    datasets root (``benchmark/vbench2``, for import paths) rather than the
    vbench_data symlink — the two share the variable name but not the layout.
    Probe candidates in order and return the first that holds the weights,
    or None when no shared data is available.
    """
    candidates = []
    env_home = os.environ.get('VBENCH_DATA_HOME')
    if env_home:
        candidates.append(env_home)
        candidates.append(os.path.join(env_home, 'vbench2_beta_i2v'))
    env_datasets = os.environ.get('VBENCH2_DATASETS')
    if env_datasets:
        candidates.append(os.path.join(env_datasets, 'vbench2_beta_i2v'))
        candidates.append(env_datasets)
    candidates.append('/xllm_test_060/vbench_data')
    candidates.append('/xllm_test_060/benchmark/vbench2/vbench2_beta_i2v')
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, 'pretrained_models')):
            return c
    return None


def _ensure_cache_layout():
    """Idempotently materialize the VBench cache layout from shared data.

    The evaluation dimensions resolve their scoring models (CLIP/DINO/RAFT/
    AMT/aesthetic/dreamsim) through ``VBENCH_CACHE_DIR``.  The layout is a set
    of symlinks into the shared ``vbench_data`` tree, so a fresh machine only
    needs the shared data copied — this rebuilds whatever is missing instead
    of requiring a manual cache setup step.  Existing entries are left
    untouched (resume-safe).
    """
    cache = VBENCH_CACHE_DIR
    src = _resolve_data_home()
    if src is None:
        # 共享数据不在（路径未配置或未拷贝），保持现状并提示
        print('[warn] 未找到含 pretrained_models 的 VBench 共享数据，跳过 cache 布局重建')
        return

    def _link(name, target):
        dst = os.path.join(cache, name)
        if os.path.lexists(dst):
            return
        if not os.path.exists(target):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(target, dst)

    pm = os.path.join(src, 'pretrained_models')
    # ais_bench vbench 侧的模型目录
    for model in ('aesthetic_model', 'amt_model', 'pyiqa_model', 'raft_model'):
        _link(model, os.path.join(pm, model))
    # CLIP
    for weight in ('ViT-B-32.pt', 'ViT-L-14.pt'):
        _link(os.path.join('clip_model', weight),
              os.path.join(src, 'clip', weight))
    # DINO（vbench 侧从 dino_model/ 直读）
    _link(os.path.join('dino_model', 'dino_vitbase16_pretrain.pth'),
          os.path.join(pm, 'torch', 'hub', 'checkpoints', 'dino_vitbase16_pretrain.pth'))
    _link(os.path.join('dino_model', 'facebookresearch_dino_main'),
          os.path.join(pm, 'torch', 'hub', 'facebookresearch_dino_main'))
    # dreamsim（i2v 维度）
    for weight in ('clip_vitb16_pretrain.pth.tar', 'dino_vitb16_pretrain.pth',
                   'open_clip_vitb16_pretrain.pth.tar'):
        _link(os.path.join('dreamsim_model', weight),
              os.path.join(pm, 'dreamsim_model', weight))
    _link(os.path.join('dreamsim_model', 'ensemble_lora'),
          os.path.join(pm, 'dreamsim_model', 'ensemble_lora'))
    # torch.hub（camera_motion 的 co-tracker、TORCH_HOME 指向这里）
    _link(os.path.join('torch', 'hub', 'facebookresearch_co-tracker_main'),
          os.path.join(pm, 'torch', 'hub', 'facebookresearch_co-tracker_main'))
    _link(os.path.join('torch', 'hub', 'facebookresearch_dino_main'),
          os.path.join(pm, 'torch', 'hub', 'facebookresearch_dino_main'))
    # torch.hub checkpoints：co-tracker 的 hubconf 以 pretrained=True 调
    # load_state_dict_from_url(_COTRACKER2_URL)，先查 TORCH_HOME/hub/checkpoints/
    # —— 缓存缺失时会转网络下载，离线环境直接挂死（SSL 握手无超时）。
    _link(os.path.join('torch', 'hub', 'checkpoints'),
          os.path.join(pm, 'torch', 'hub', 'checkpoints'))
    trusted = os.path.join(cache, 'torch', 'hub', 'trusted_list')
    if not os.path.exists(trusted):
        os.makedirs(os.path.dirname(trusted), exist_ok=True)
        with open(trusted, 'w', encoding='utf-8') as f:
            f.write('facebookresearch_co-tracker\nfacebookresearch_dino\n')


def _setup_paths():
    """让 vbench（ais_bench 适配版）、vbench2_beta_i2v（本地官方副本）、
    scripts/constant.py（总分常数）可被 import。"""
    # PYTHONPATH 只在解释器启动时读取，对本进程无效 —— 必须直接改 sys.path。
    paths = list(sys.path)
    for p in (AIS_THIRD_PARTY, VBENCH2_DATASETS, os.path.join(VBENCH2_ROOT, 'scripts')):
        if p not in paths:
            paths.insert(0, p)
    sys.path = paths


class _Cv2VideoReader:
    """cv2-backed drop-in replacement for ``decord.VideoReader``.

    The runner image pairs decord (old filter-graph API) with FFmpeg 9, where
    ``av_opt_set_array(..., "pixel_formats", ...)`` after buffersink init
    returns -22 (the option is no longer runtime-settable), so every decord
    ``VideoReader`` open fails.  ``temporal_flickering`` already reads via
    cv2 and works; this reuses that path for all other dimensions.
    """

    def __init__(self, path, width=None, height=None, num_threads=1):
        import cv2
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise IOError(f'cannot open video: {path}')
        self._width = width
        self._height = height
        self._frames = None

    def __len__(self):
        # CAP_PROP_FRAME_COUNT is metadata and can disagree with the frames
        # actually decodable (e.g. 81 vs 80 on the last partial frame).
        # Callers index range(len(reader)), so the count must match what
        # get_batch can actually serve: force-decode and count.
        self._load_all()
        return len(self._frames)

    def get_avg_fps(self):
        import cv2
        return self._cap.get(cv2.CAP_PROP_FPS)

    def _load_all(self):
        if self._frames is not None:
            return self._frames
        import cv2
        import numpy as np
        frames = []
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if self._width:
                frame = cv2.resize(frame, (self._width, self._height))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._frames = np.stack(frames) if frames else np.zeros((0, 2, 2, 3), dtype=np.uint8)
        return self._frames

    def get_batch(self, indices):
        import numpy as np
        frames = self._load_all()
        sel = frames[list(indices)]
        # decord's native bridge returns an object with .asnumpy()
        class _B:
            @staticmethod
            def asnumpy():
                return sel
        return _B()


def _patch_decord_if_broken():
    """Replace decord.VideoReader when the FFmpeg-9 incompatibility is present.

    Probes decord with a tiny real video only once; on DECORDError falls back
    to the cv2 implementation above.  No-op when decord works (e.g. a future
    image with a compatible FFmpeg).
    """
    import decord
    if getattr(decord, '_xTaskCv2Patch', False):
        return
    try:
        import tempfile, os as _os
        import cv2, numpy as np
        # 生成 3 帧纯色视频做探针
        probe = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        probe.close()
        w = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(probe.name, w, 10, (32, 32))
        for c in (64, 128, 192):
            vw.write(np.full((32, 32, 3), c, dtype=np.uint8))
        vw.release()
        try:
            decord.VideoReader(probe.name, num_threads=1)
        except Exception:
            decord.VideoReader = _Cv2VideoReader
            decord._xTaskCv2Patch = True
            print('[warn] decord 与镜像内 FFmpeg9 不兼容，已切换 cv2 后端读视频')
        finally:
            _os.unlink(probe.name)
    except Exception:
        # 探针本身失败（cv2 缺失等）时保持原状
        pass


def detect_device_str() -> str:
    import torch
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    if getattr(torch, 'npu', None) and torch.npu.is_available():
        return 'npu'
    return 'cpu'


# --------------------------------------------------------------------------
# 总分计算（与上游 scripts/cal_i2v_final_score.py 完全一致）
# --------------------------------------------------------------------------
def compute_total_score(results_dict):
    """results_dict: {dim: [score, ...]}（官方 eval results 格式）。

    返回 (upload_data, normalized_score, quality_score, i2v_score, total_score)。
    实现逐行对齐上游：
      get_nomalized_score() / get_i2v_quality_score() / get_i2v_score() / get_final_score()
    """
    from constant import (TASK_INFO_I2V, I2VKEY, NORMALIZE_DIC_I2V, DIM_WEIGHT_I2V,
                          I2V_LIST, I2V_QUALITY_LIST, I2V_QUALITY_WEIGHT, I2V_WEIGHT)

    # submission(): {dim 显示名: 分数}，缺失维度按 0 处理（上游行为）
    upload_data = {}
    for dim, value in results_dict.items():
        if dim not in I2VKEY:
            continue
        score = value[0] if isinstance(value, (list, tuple)) else value
        if score is None or (isinstance(score, float) and math.isnan(score)):
            continue
        upload_data[I2VKEY[dim]] = float(score)
    for key in TASK_INFO_I2V:
        if key not in upload_data:
            print(f'[warn] "{key}" 无有效分数，按 0 处理（上游 submission() 行为）')
            upload_data[key] = 0.0

    # get_nomalized_score()
    normalized_score = {}
    for key in TASK_INFO_I2V:
        min_val = NORMALIZE_DIC_I2V[key]['Min']
        max_val = NORMALIZE_DIC_I2V[key]['Max']
        normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = normalized_score[key] * DIM_WEIGHT_I2V[key]

    # get_i2v_quality_score() / get_i2v_score() / get_final_score()
    quality_score = (sum(normalized_score[k] for k in I2V_QUALITY_LIST)
                     / sum(DIM_WEIGHT_I2V[i] for i in I2V_QUALITY_LIST))
    i2v_score = (sum(normalized_score[k] for k in I2V_LIST)
                 / sum(DIM_WEIGHT_I2V[i] for i in I2V_LIST))
    final_score = (quality_score * I2V_QUALITY_WEIGHT + i2v_score * I2V_WEIGHT) \
        / (I2V_QUALITY_WEIGHT + I2V_WEIGHT)
    return upload_data, normalized_score, quality_score, i2v_score, final_score


def _sanitize(obj):
    """把结果里的 numpy 标量转成 python 标量，nan/inf 转 None，保证 json 合法。"""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def print_score_table(upload_data, normalized_score, quality_score, i2v_score, total_score):
    from constant import TASK_INFO_I2V, DIM_WEIGHT_I2V, NORMALIZE_DIC_I2V
    print('\n' + '=' * 70)
    print('VBench-I2V 总分（与官方 cal_i2v_final_score.py 相同口径）')
    print('=' * 70)
    print(f'{"维度":<36}{"原始分":>10}{"归一化×权重":>14}{"Min":>8}{"Max":>8}{"权重":>6}')
    for key in TASK_INFO_I2V:
        mn = NORMALIZE_DIC_I2V[key]['Min']
        mx = NORMALIZE_DIC_I2V[key]['Max']
        print(f'{key:<36}{upload_data[key]:>10.4f}{normalized_score[key]:>14.4f}'
              f'{mn:>8.3f}{mx:>8.3f}{DIM_WEIGHT_I2V[key]:>6.2f}')
    print('-' * 70)
    print(f'|     Quality Score|{quality_score:12.4f}|')
    print(f'|         I2V Score|{i2v_score:12.4f}|')
    print(f'|       Total Score|{total_score:12.4f}|')
    print('=' * 70)


# --------------------------------------------------------------------------
# 评测主流程
# --------------------------------------------------------------------------
def count_eligible(full_info, dim):
    n_entries = n_videos = 0
    for e in full_info:
        if dim in e.get('dimension', []) and 'video_list' in e:
            n_entries += 1
            vl = e['video_list'] if isinstance(e['video_list'], list) else [e['video_list']]
            n_videos += len(vl)
    return n_entries, n_videos


def check_preflight(full_info_path, dims, resolution):
    """只做体检：json 覆盖、图片与视频文件存在性，不加载任何模型。"""
    full_info = json.load(open(full_info_path, encoding='utf-8'))
    n_with = sum(1 for e in full_info if e.get('video_list'))
    print(f'full_info: {full_info_path}')
    print(f'entries: {len(full_info)}, with video_list: {n_with}')
    ok = True
    evaluable_dims = 0
    for dim in dims:
        n_entries, n_videos = count_eligible(full_info, dim)
        print(f'  {dim:<22} entries={n_entries:>4}  videos={n_videos:>4}')
        if n_videos == 0:
            # limit 抽样试跑时部分维度可能无覆盖；有覆盖的维度照常评，
            # 这里只降级为 warning，不判整体失败。
            print(f'    [warn] 该维度没有任何视频可评，跳过该维度')
        else:
            evaluable_dims += 1
    if evaluable_dims == 0:
        print('  [error] 没有任何维度有可评视频')
        ok = False
    # i2v 维度需要参考图（相对 cwd=VBench_stand/datasets 的官方路径）
    crop_dir = os.path.join(VBENCH2_ROOT, 'data', 'crop', resolution)
    if not os.path.isdir(crop_dir):
        print(f'[warn] 裁剪图目录不存在: {crop_dir}')
        ok = False
    missing_img = missing_vid = 0
    for e in full_info:
        if 'image_name' in e:
            img = os.path.join(crop_dir, e['image_name'])
            if not os.path.isfile(img):
                missing_img += 1
        if 'video_list' in e:
            for v in (e['video_list'] if isinstance(e['video_list'], list) else [e['video_list']]):
                if not os.path.isfile(v):
                    missing_vid += 1
    print(f'  缺失图片: {missing_img}, 缺失视频文件: {missing_vid}')
    if missing_img or missing_vid:
        ok = False
    print('check:', 'PASS' if ok else 'FAIL')
    return ok


def run_dimension(dim, full_info_path, device, submodules):
    """调用官方 compute_<dimension>，返回官方 results 元组。"""
    import torch
    if dim in I2V_DIMS:
        mod = importlib.import_module(f'vbench2_beta_i2v.{dim}')
    else:
        mod = importlib.import_module(f'vbench.{dim}')
    evaluate_func = getattr(mod, f'compute_{dim}')
    # dreamsim 初始化 ViT 时会 torch.hub.set_dir('./models') 且不还原，
    # 会污染后续维度（如 camera_motion）的 torch.hub 缓存目录 —— 这里保存并还原。
    hub_dir_before = torch.hub.get_dir()
    try:
        return evaluate_func(full_info_path, device, submodules)
    finally:
        torch.hub.set_dir(hub_dir_before)


def main():
    parser = argparse.ArgumentParser(
        description='VBench-I2V mini 视频测评（10 维度 + 总分计算，按官方 README 流程）')
    parser.add_argument('--full_info', type=str, default=None,
                        help='生成脚本产出的 *_videos.json（含 video_list，官方 full_info 格式）。')
    parser.add_argument('--resolution', type=str, default='16-9',
                        help='参考图裁剪比例（对应 data/crop/{resolution}，官方可选 1-1/8-5/7-4/16-9）。')
    parser.add_argument('--name', type=str, default='wan2_2_i2v_mini',
                        help='评测结果命名，输出 {name}_eval_results.json / {name}_final_score.json。')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'i2v_eval_results'),
                        help='结果输出目录。')
    parser.add_argument('--dims', nargs='*', default=ALL_DIMS,
                        choices=ALL_DIMS,
                        help='要评测的维度（默认全部 10 个）。')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'npu', 'cpu'],
                        help='评测设备（默认自动检测 NPU）。')
    parser.add_argument('--check', action='store_true',
                        help='只做体检（json/图片/视频文件检查），不评测。')
    parser.add_argument('--results_file', type=str, default=None,
                        help='跳过评测，直接对已有的 {name}_eval_results.json 重算总分。')
    args = parser.parse_args()

    _setup_paths()
    # 共享数据已就位时自动补齐评分模型 cache 布局（幂等，新机器免手工准备）
    _ensure_cache_layout()
    # decord/FFmpeg9 不兼容时切 cv2 后端（见 _patch_decord_if_broken）
    _patch_decord_if_broken()
    # 评测容器（VBench runner）不挂 Ascend toolkit，torch_npu autoload 会因
    # libhccl.so 缺失直接 raise。评测维度模型（CLIP/DINO/RAFT 等）可在 CPU
    # 跑，所以先关掉 autoload，下面再按设备探测结果可选地 import torch_npu。
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    try:
        import torch_npu  # noqa: F401  注册 npu 设备与 hccl 后端（必须先于 torch.device('npu')）
    except ImportError:
        pass
    if args.device == 'auto':
        args.device = detect_device_str()

    if args.results_file:
        with open(args.results_file, encoding='utf-8') as f:
            results_dict = json.load(f)
        upload_data, normalized_score, quality_score, i2v_score, total_score = \
            compute_total_score(results_dict)
        print_score_table(upload_data, normalized_score, quality_score, i2v_score, total_score)
        return 0

    if not args.full_info:
        parser.error('--full_info is required（或使用 --results_file 只重算总分）')
    full_info_path = os.path.abspath(args.full_info)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.check:
        return 0 if check_preflight(full_info_path, args.dims, args.resolution) else 1

    # 设备与分布式初始化（与 ais_bench VBenchEvalTask 相同方式）
    device_str = args.device
    from vbench.distributed import dist_init
    try:
        from vbench.distributed import dist_destroy
    except ImportError:
        # ais_bench 适配版未提供 dist_destroy，结束时无需显式销毁
        dist_destroy = None
    try:
        dist_init(device=device_str)
    except Exception as e:  # 分布式初始化失败（如端口/环境问题）退化为单进程直接跑
        print(f'[warn] dist_init 失败（{e}），按单进程继续')
    device = torch.device('npu:0' if device_str == 'npu' else 'cpu')
    print(f'eval device: {device_str}')

    # i2v 维度的官方 load 函数用相对路径（vbench2_beta_i2v/data/crop/...），
    # DreamSim 权重也缓存在 datasets/models/ 下 —— 统一 chdir 到 datasets。
    os.chdir(VBENCH2_DATASETS)
    print(f'cwd -> {VBENCH2_DATASETS}（官方 load_i2v_dimension_info 相对路径要求）')

    # 预检：各维度可评视频数
    full_info = json.load(open(full_info_path, encoding='utf-8'))
    print('per-dim eligible:')
    for dim in args.dims:
        n_entries, n_videos = count_eligible(full_info, dim)
        print(f'  {dim:<22} entries={n_entries:>4}  videos={n_videos:>4}')
        if n_videos == 0:
            print(f'    [warn] 无视频可评，该维度将得到 nan（总分按缺失处理为 0）')

    # i2v_background 官方实现 dreamsim(pretrained=True) 不带 device（默认 cuda），
    # 本机无 CUDA —— 在其 import 前把 device 参数默认值补上。
    # torch_npu 依赖 CANN 环境变量（libhccl.so 在 toolkit lib 下）：外层
    # run_vbench_test.sh 已 source set_env.sh，这里无需再处理；若直接裸跑
    # 脚本（无 CANN），下方 import torch_npu 的 try/except 会静默降级 CPU。
    import dreamsim as dreamsim_pkg
    _orig_dreamsim = dreamsim_pkg.dreamsim

    def _dreamsim_with_device(**kwargs):
        kwargs.setdefault('device', device_str)
        # dreamsim 默认 cache_dir='./models'（相对 cwd），离线环境必须显式指向
        # 共享缓存里的 dreamsim_model 目录（含 ensemble 4 个权重文件），
        # 否则 download_weights 会尝试联网下载。
        kwargs.setdefault('cache_dir', os.path.join(VBENCH_CACHE_DIR, 'dreamsim_model'))
        return _orig_dreamsim(**kwargs)

    dreamsim_pkg.dreamsim = _dreamsim_with_device

    # 子模块配置（模型 checkpoint 均在 /root/.cache/vbench 或 datasets/models 本地缓存）
    from vbench2_beta_i2v.utils import init_submodules as i2v_init_submodules
    from vbench.utils import init_submodules as quality_init_submodules
    i2v_subs = i2v_init_submodules([d for d in args.dims if d in I2V_DIMS],
                                   local=True, resolution=args.resolution)
    quality_subs = quality_init_submodules([d for d in args.dims if d in QUALITY_DIMS],
                                           local=True)
    submodules = {**i2v_subs, **quality_subs}

    results_dict = {}
    eval_results_path = os.path.join(output_dir, f'{args.name}_eval_results.json')
    # 断点续评：已有结果文件中不在 --dims 里的维度直接沿用，最后总分用合并后的全部维度
    if os.path.isfile(eval_results_path):
        try:
            with open(eval_results_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            reused = {d: v for d, v in saved.items() if d not in args.dims}
            results_dict.update(reused)
            if reused:
                print(f'reusing {len(reused)} saved dim(s) from previous run: {sorted(reused)}')
        except Exception:
            print('[warn] 读取既有结果文件失败，忽略（按全新评测处理）')
    for i, dim in enumerate(args.dims, start=1):
        print(f'\n[{i}/{len(args.dims)}] evaluating {dim} ...')
        # 预检里无视频的维度直接跳过：上游 compute_* 对空 video_list 会做
        # 0 除（如 subject_consistency 的 sim/cnt），官方全量数据不会触发，
        # limit 抽样时必须由调用方挡住。
        n_entries, n_videos = count_eligible(full_info, dim)
        if n_videos == 0:
            print(f'  [skip] 无视频可评，跳过（总分按缺失=0 处理）')
            continue
        try:
            results = run_dimension(dim, full_info_path, device, submodules[dim])
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[FAILED] {dim}: {e}（继续后续维度；总分计算时该维度按 0 处理，官方口径）')
            continue
        score = results[0]
        print(f'  {dim} score: {score}')
        results_dict[dim] = results
        # 增量落盘，中断可恢复
        with open(eval_results_path, 'w', encoding='utf-8') as f:
            json.dump(_sanitize(results_dict), f, indent=2, ensure_ascii=False)
        gc.collect()
        if device_str == 'npu':
            torch.npu.empty_cache()

    print(f'\neval results saved: {eval_results_path} '
          f'({len(results_dict)}/{len(args.dims)} dims)')

    # 总分（官方口径）
    upload_data, normalized_score, quality_score, i2v_score, total_score = \
        compute_total_score(results_dict)
    print_score_table(upload_data, normalized_score, quality_score, i2v_score, total_score)

    final_out = {
        'model_name': args.name,
        'raw_scores': {k: v for k, v in upload_data.items()},
        'normalized_weighted': {k: v for k, v in normalized_score.items()},
        'quality_score': quality_score,
        'i2v_score': i2v_score,
        'total_score': total_score,
        'note': '计算流程与官方 scripts/cal_i2v_final_score.py 完全一致；'
                'temporal_flickering 已评测但不参与总分（官方口径）；'
                '缺失/失败维度按 0 处理（官方 submission() 行为）。',
    }
    final_path = os.path.join(output_dir, f'{args.name}_final_score.json')
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize(final_out), f, indent=2, ensure_ascii=False)
    print(f'final score saved: {final_path}')

    try:
        if dist_destroy is not None:
            dist_destroy()
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())

