#!/usr/bin/env python3
"""把 modelers.cn 原生 Wan2.2-I2V-A14B 权重转成 xLLM 需要的 diffusers 目录。

native key（modelers）                  -> diffusers key（xLLM）
blocks.N.self_attn.{q,k,v,o}            -> blocks.N.attn1.{to_q,to_k,to_v,to_out.0}
blocks.N.cross_attn.{q,k,v,o}           -> blocks.N.attn2.{to_q,to_k,to_v,to_out.0}
blocks.N.ffn.{0,2}                      -> blocks.N.ffn.net.{0.proj,2}
blocks.N.modulation                     -> blocks.N.scale_shift_table
text_embedding.{0,2}                    -> condition_embedder.text_embedder.{linear_1,linear_2}
time_embedding.{0,2}                    -> condition_embedder.time_embedder.{linear_1,linear_2}
time_projection.1                       -> condition_embedder.time_proj
head.head                               -> proj_out
head.modulation                         -> scale_shift_table

来源/目标都走 safetensors 零拷贝重命名（只改 header，不搬张量数据），
6 分片并成单文件输出到 transformer/ 与 transformer_2/。
"""
import json
import os
import re
import struct
import sys

# blocks.N.<prefix>.<suffix> 形式的块内映射
_BLOCK_RULES = [
    (re.compile(r"^self_attn\.q\.(.*)"), r"attn1.to_q.\1"),
    (re.compile(r"^self_attn\.k\.(.*)"), r"attn1.to_k.\1"),
    (re.compile(r"^self_attn\.v\.(.*)"), r"attn1.to_v.\1"),
    (re.compile(r"^self_attn\.o\.(.*)"), r"attn1.to_out.0.\1"),
    (re.compile(r"^self_attn\.norm_q\.(.*)"), r"attn1.norm_q.\1"),
    (re.compile(r"^self_attn\.norm_k\.(.*)"), r"attn1.norm_k.\1"),
    (re.compile(r"^cross_attn\.q\.(.*)"), r"attn2.to_q.\1"),
    (re.compile(r"^cross_attn\.k\.(.*)"), r"attn2.to_k.\1"),
    (re.compile(r"^cross_attn\.v\.(.*)"), r"attn2.to_v.\1"),
    (re.compile(r"^cross_attn\.o\.(.*)"), r"attn2.to_out.0.\1"),
    (re.compile(r"^cross_attn\.norm_q\.(.*)"), r"attn2.norm_q.\1"),
    (re.compile(r"^cross_attn\.norm_k\.(.*)"), r"attn2.norm_k.\1"),
    (re.compile(r"^ffn\.0\.(.*)"), r"ffn.net.0.proj.\1"),
    (re.compile(r"^ffn\.2\.(.*)"), r"ffn.net.2.\1"),
    # Wan native 的块内 norm3 对应 diffusers 的 norm2
    # （QK norm 用 rms_norm_across_heads 时 diffusers 无 norm1）
    (re.compile(r"^norm3\.(.*)"), r"norm2.\1"),
    (re.compile(r"^modulation$"), r"scale_shift_table"),
]

_ROOT_RULES = [
    (re.compile(r"^text_embedding\.0\.(.*)"), r"condition_embedder.text_embedder.linear_1.\1"),
    (re.compile(r"^text_embedding\.2\.(.*)"), r"condition_embedder.text_embedder.linear_2.\1"),
    (re.compile(r"^time_embedding\.0\.(.*)"), r"condition_embedder.time_embedder.linear_1.\1"),
    (re.compile(r"^time_embedding\.2\.(.*)"), r"condition_embedder.time_embedder.linear_2.\1"),
    (re.compile(r"^time_projection\.1\.(.*)"), r"condition_embedder.time_proj.\1"),
    (re.compile(r"^head\.head\.(.*)"), r"proj_out.\1"),
    (re.compile(r"^head\.modulation$"), r"scale_shift_table"),
]


def rename_key(k: str) -> str:
    m = re.match(r"^blocks\.(\d+)\.(.+)$", k)
    if m:
        idx, rest = m.group(1), m.group(2)
        for pattern, repl in _BLOCK_RULES:
            hit = pattern.match(rest)
            if hit:
                return f"blocks.{idx}." + hit.expand(repl)
        return k
    for pattern, repl in _ROOT_RULES:
        hit = pattern.match(k)
        if hit:
            return hit.expand(repl)
    return k


def read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def convert(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    shards = sorted(
        f for f in os.listdir(src_dir)
        if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
    )
    # 汇总所有分片的 header，重排成一个单文件
    parts = []
    total_tensors = 0
    for shard in shards:
        header, _ = read_header(os.path.join(src_dir, shard))
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        parts.append((shard, entries))
        total_tensors += len(entries)

    # diffusers 顺序：condition_embedder, patch_embedding, blocks, proj_out, scale_shift_table
    all_entries = {}
    for shard, entries in parts:
        for k, v in entries.items():
            all_entries[rename_key(k)] = v

    # 验证关键 key 是否齐全
    expect_prefixes = (
        "condition_embedder.", "patch_embedding.", "blocks.0.",
        "blocks.39.", "proj_out.", "scale_shift_table",
    )
    flat = "\n".join(sorted(all_entries))
    for p in expect_prefixes:
        assert p in flat, f"missing key prefix: {p}"
    print(f"{src_dir} -> {dst_dir}: {total_tensors} tensors, "
          f"renamed unique keys: {len(all_entries)}")

    # 单文件输出：把各分片数据段依次拼接，重写 header 的 data_offsets
    out_path = os.path.join(dst_dir, "diffusion_pytorch_model.safetensors")
    offset = 0
    new_header = {}
    with open(out_path, "wb") as out:
        for shard, entries in parts:
            _, header_end = read_header(os.path.join(src_dir, shard))
            file_size = os.path.getsize(os.path.join(src_dir, shard))
            data_len = file_size - header_end
            with open(os.path.join(src_dir, shard), "rb") as f:
                f.seek(header_end)
                remaining = data_len
                while remaining > 0:
                    chunk = f.read(min(1 << 24, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)
            # 重算该分片内每个 key 的 offset
            for k, v in entries.items():
                new_key = rename_key(k)
                start, end = v["data_offsets"]
                new_header[new_key] = {
                    "dtype": v["dtype"],
                    "shape": v["shape"],
                    "data_offsets": [offset + start, offset + end],
                }
            offset += data_len
        header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
        # 需要回填 header 长度：先写占位再重写
        final = struct.pack("<Q", len(header_bytes)) + header_bytes + out.getvalue()[8:] if False else None
    # 上面的流式写法难以回填，改为先算好 header 再写
    os.remove(out_path)
    # pass 1: 计算 header（需要知道每个 key 的最终 offset —— 顺序按分片排列）
    ordered_keys = []
    offset = 0
    for shard, entries in parts:
        _, header_end = read_header(os.path.join(src_dir, shard))
        data_len = os.path.getsize(os.path.join(src_dir, shard)) - header_end
        for k, v in entries.items():
            start, end = v["data_offsets"]
            ordered_keys.append((rename_key(k), v["dtype"], v["shape"],
                                 offset + start, offset + end))
        offset += data_len
    new_header = {
        k: {"dtype": dt, "shape": sh, "data_offsets": [s, e]}
        for k, dt, sh, s, e in ordered_keys
    }
    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for shard, _ in parts:
            _, header_end = read_header(os.path.join(src_dir, shard))
            with open(os.path.join(src_dir, shard), "rb") as f:
                f.seek(header_end)
                while True:
                    chunk = f.read(1 << 24)
                    if not chunk:
                        break
                    out.write(chunk)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"  written: {out_path} ({size_gb:.1f} GB)")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="modelers.cn 原生 Wan2.2-I2V-A14B -> diffusers 目录转换"
                    "（high/low noise 两个 transformer 各转一份，6 分片并 1）")
    parser.add_argument("--model-root", required=True,
                        help="模型根目录，其下含 high_noise_model/ low_noise_model/")
    parser.add_argument("--high-src", default=None,
                        help="覆盖 high 噪声源目录（默认 <model-root>/high_noise_model）")
    parser.add_argument("--low-src", default=None,
                        help="覆盖 low 噪声源目录（默认 <model-root>/low_noise_model）")
    args = parser.parse_args()

    root = args.model_root
    convert(args.high_src or os.path.join(root, "high_noise_model"),
            os.path.join(root, "transformer"))
    convert(args.low_src or os.path.join(root, "low_noise_model"),
            os.path.join(root, "transformer_2"))
    print(
        "注意：本脚本只转换 transformer 权重。完整 diffusers 目录还需\n"
        "  model_index.json / scheduler/ / text_encoder/ / tokenizer/ / vae/\n"
        "（从 HF Wan2.2-I2V-A14B-Diffusers 拉取后放入同一 model-root），\n"
        "并在根目录补 config.json（transformers_version: null 即可跳过框架的\n"
        "transformers 版本对齐安装）。"
    )
