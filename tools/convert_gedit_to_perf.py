"""把 GEdit-Bench 数据集转换为 gedit perf JSONL。

输入：datasets ``load_from_disk`` 格式的 GEdit-Bench（HF datasets 保存，
字段含 ``instruction`` 与 ``input_image``（PIL Image 或可转换对象），
默认路径 /xllm_test_060/benchmark/ais_bench/datasets/GEdit-Bench，
可用 ``--dataset-path`` 覆盖）。

输出目录结构：

    <output-dir>/
        gedit_perf.jsonl     # 每行 {"prompt", "images"(绝对路径), "size",
                             #        "num_inference_steps", "guidance_scale"}
        images/              # input_image 逐条另存的 PNG
            000000.png
            000001.png
            ...

行格式与 xtask/plugins/evalscope/gedit_api.py 的 build_request 约定一致：
``images`` 存图片绝对路径（jsonl 不存 base64，评测时按需读盘编码）。
``size`` / ``num_inference_steps`` / ``guidance_scale`` 由本工具 CLI 参数
注入到每一行；``seed`` 不写入（由 wrapper/数据集行缺省 42 兜底）。

用法示例：

    python tools/convert_gedit_to_perf.py \
        --output-dir outputs/gedit_perf \
        --limit 20 --size 1328*1328 \
        --num-inference-steps 50 --guidance-scale 3.5
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GEdit-Bench (datasets load_from_disk) to gedit perf jsonl."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/xllm_test_060/benchmark/ais_bench/datasets/GEdit-Bench",
        help="GEdit-Bench dataset root (datasets load_from_disk format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gedit_perf",
        help="Output directory; writes gedit_perf.jsonl and images/ under it",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max examples to convert; 0 means all",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1328*1328",
        help="Image size injected into each jsonl line (W*H)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="num_inference_steps injected into each jsonl line",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="guidance_scale injected into each jsonl line",
    )
    return parser.parse_args()


def convert(dataset_path, output_dir, limit=0, size="1328*1328",
            num_inference_steps=50, guidance_scale=3.5):
    """执行转换，返回写出的 jsonl 行数。"""
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "gedit_perf.jsonl")
    total = len(dataset) if limit <= 0 else min(limit, len(dataset))

    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for index in range(total):
            example = dataset[index]
            instruction = example.get("instruction")
            if not isinstance(instruction, str) or not instruction:
                raise ValueError(
                    f"GEdit-Bench example {index} has no valid 'instruction' field"
                )
            image = example.get("input_image")
            if image is None:
                raise ValueError(
                    f"GEdit-Bench example {index} has no 'input_image' field"
                )

            # 另存 PNG（带序号文件名），jsonl 只存绝对路径
            image_name = f"{index:06d}.png"
            image_path = os.path.abspath(os.path.join(images_dir, image_name))
            if hasattr(image, "save"):
                # HF datasets 的 Image 特征返回 PIL.Image（已加载）
                if getattr(image, "mode", "RGB") != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, format="PNG")
            else:
                # bytes 情况（datasets 未安装 PIL 解码器扩展时）
                with open(image_path, "wb") as img_file:
                    img_file.write(image)

            record = {
                "prompt": instruction,
                "images": [image_path],
                "size": size,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[convert_gedit_to_perf] wrote {written} lines to {jsonl_path}")
    return written


def main():
    args = parse_args()
    convert(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        limit=args.limit,
        size=args.size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
