#!/usr/bin/env python3
"""Preflight asset check for daily builds.

Verifies the externally provisioned assets that benchmark cases depend on but
that the framework does not distribute itself.  Prints one line per asset
group with status and, on failure, which cases will break and how to fix.

Exit code is 0 even when assets are missing (report-only by default; pass
--strict to fail), so it can run at the top of a daily build without
blocking unrelated case groups.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FALLBACK_RESOURCE_ROOT = Path("/export/home/models/xllm_test_060")


def check_group(name: str, description: str, cases: str, missing: list[str]) -> bool:
    if not missing:
        print(f"[OK]   {name}: {description}")
        return True
    print(f"[MISS] {name}: {len(missing)} missing asset(s):")
    for item in missing[:5]:
        print(f"         - {item}")
    if len(missing) > 5:
        print(f"         ... and {len(missing) - 5} more")
    print(f"       affected cases: {cases}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when any asset group is missing")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    shared = FALLBACK_RESOURCE_ROOT
    failures = 0

    # ---- 1. xllm wheel (standard distribution, every xllm case) ----
    wheels = sorted((project / "pkg" / "xllm").glob("xllm_npu*.whl"))
    if not check_group(
        "xllm wheel",
        f"{len(wheels)} wheel(s) in pkg/xllm" if wheels else "no wheel in pkg/xllm",
        "all xllm cases (build stage output; run the build stage first)",
        [] if wheels else ["pkg/xllm/xllm_npu*.whl"],
    ):
        failures += 1

    # ---- 2. ASR assets (dataset + WER tools, split by role) ----
    asr_missing = [
        str(p) for p in (
            shared / "mmm_bench" / "joyai_asr" / "infer.zip",
            shared / "datasets" / "joyai_asr" / "data.zip",
        )
        if not p.is_file()
    ]
    if asr_missing:
        legacy = [str(project / "asr_doc" / Path(p).name) for p in asr_missing]
        asr_missing = [f"{p} (fallback: {l})" for p, l in zip(asr_missing, legacy)]
    if not check_group(
        "ASR assets",
        f"infer.zip (mmm_bench/joyai_asr) + data.zip (datasets/joyai_asr) under {shared}",
        "joyai_asr_xllm_tp2_legacy",
        asr_missing,
    ):
        failures += 1

    # ---- 3. VBench canonical asset tree ----
    # Single real copy serving both WAN I2V and JoyAI video cases: prompts
    # (full_info) + reference images (data/) + scoring weights (clip/ +
    # pretrained_models/).  The flat scoring-model cache is a symlink view
    # rebuilt on demand by ensure_cache_layout(), so only this tree must
    # exist on the host.
    vbench_root = shared / "mmm_bench" / "Video" / "vbench2_beta_i2v"
    vbench_canonical = [
        "full_info.json",
        "vbench2_i2v_mini_full_info.json",
        "data",
        "clip/ViT-B-32.pt",
        "clip/ViT-L-14.pt",
        "pretrained_models/aesthetic_model",
        "pretrained_models/amt_model",
        "pretrained_models/pyiqa_model",
        "pretrained_models/raft_model",
        "pretrained_models/dreamsim_model",
        "pretrained_models/torch/hub/checkpoints/dino_vitbase16_pretrain.pth",
    ]
    vbench_missing = [
        str(vbench_root / rel) for rel in vbench_canonical
        if not (vbench_root / rel).exists()
    ]
    if not check_group(
        "VBench canonical asset tree",
        f"full_info + data/ + pretrained_models under {vbench_root}",
        "wan2.2 vbench cases, joyai_video t2av/quality (scoring dims will fail)",
        vbench_missing,
    ):
        failures += 1

    # ---- 4. JoyAI video negative-calibration environment ----
    # Isolated venv + scoring weights for benchmark_semantics checks
    # (semantic_env is a venv; grounding-dino/clap are the vision/audio
    # detector weights referenced by the case yamls).
    jb_root = shared / "mmm_bench" / "joyai_video_benchmark"
    jb_missing = [
        str(p) for p in (
            jb_root / "semantic_env" / "bin" / "python3",
            jb_root / "grounding-dino-tiny" / "model.safetensors",
            jb_root / "clap-htsat-unfused" / "pytorch_model.bin",
        )
        if not p.exists()
    ]
    if not check_group(
        "JoyAI negative-calibration env",
        f"semantic_env + grounding-dino + clap under {jb_root}",
        "joyai_video_negative_calibrated (negative control checks)",
        jb_missing,
    ):
        failures += 1

    print()
    if failures:
        print(f"preflight: {failures} asset group(s) incomplete")
        if args.strict:
            return 1
        print("preflight: continuing (report-only mode)")
        return 0
    print("preflight: all asset groups ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
